import base64
from typing import Dict, Any, List
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.schema import Document, MediaResource
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock, MessageRole
from llama_index.core.tools import BaseTool, FunctionTool
from prompts.prompts import description_prompt, legal_prompt, clarifications_prompt, consistency_prompt, raw_text_extraction_prompt
from raptor.raptor_setup import RaptorSetup
from datetime import datetime
from utils.output_saver import OutputSaver
from utils.text_extractor import TextExtractor
from utils.logo_product_matcher import LogoProductMatcher
import os
from pathlib import Path
import re
import calendar

class Tools:
    """Collection des outils disponibles pour l'analyse de publicité"""
    def __init__(self, llm: AzureOpenAI, raptor: RaptorSetup):
        self.llm = llm
        self.raptor = raptor
        self._tools = self._create_tools()
        self.vision_result = None
        self.legislation = None
        self.raw_text = None
        self.output_saver = OutputSaver()
        self.text_extractor = TextExtractor()
        self.extracted_text = None
        self.logo_product_matcher = LogoProductMatcher()
        self.product_logo_inconsistencies = []
        # Dictionnaire pour référence correcte des jours de la semaine
        self.correct_days = {
            "lundi": "lundi",
            "mardi": "mardi", 
            "mercredi": "mercredi", 
            "jeudi": "jeudi", 
            "vendredi": "vendredi", 
            "samedi": "samedi", 
            "dimanche": "dimanche"
        }
    
    def _create_tools(self) -> list[BaseTool]:
        """Crée la liste des outils disponibles pour l'agent"""
        return [
            FunctionTool.from_defaults(
                fn=self.extract_raw_text_for_agent,
                name="extract_raw_text",
                description="Extrait le texte brut d'une image publicitaire sans aucune modification ou correction. À utiliser en PREMIER, avant toute autre analyse.",
            ),
            FunctionTool.from_defaults(
                fn=self.analyze_vision,
                name="analyze_vision",
                description="Analyse une image publicitaire et fournit une description détaillée structurée. Utilisez cet outil APRÈS l'extraction du texte brut.",
            ),
            FunctionTool.from_defaults(
                fn=self.verify_consistency,
                name="verify_consistency",
                description="Vérifie la cohérence des informations (orthographe, adresse, téléphone, email, url) après l'analyse visuelle.",
            ),
            FunctionTool.from_defaults(
                fn=self.verify_product_logo_consistency,
                name="verify_product_logo_consistency",
                description="Vérifie la cohérence entre les logos et les produits mentionnés dans la publicité.",
            ),
            FunctionTool.from_defaults(
                fn=self.verify_dates,
                name="verify_dates",
                description="Vérifie la cohérence des dates et des jours de la semaine mentionnés dans la publicité. Vérifie également si les dates sont futures ou passées.",
            ),
            FunctionTool.from_defaults(
                fn=self.search_legislation,
                name="search_legislation",
                description="Recherche la législation applicable en fonction de la description de l'image. À utiliser après analyze_vision.",
            ),
            FunctionTool.from_defaults(
                fn=self.get_clarifications,
                name="get_clarifications",
                description="Obtient des clarifications spécifiques sur des aspects de la publicité en se basant sur la vision et la législation.",
            ),
            FunctionTool.from_defaults(
                fn=self.analyze_compliance,
                name="analyze_compliance",
                description="Analyse finale de la conformité de la publicité en combinant tous les résultats précédents.",
            ),
        ]
    
    @property
    def tools(self) -> list[BaseTool]:
        """Retourne la liste des outils disponibles"""
        return self._tools

    def analyze_vision(self, image_path: str) -> str:
        """
        Analyse une image publicitaire avec GPT-4o
        Args:
            image_path: Chemin vers l'image à analyser
        Returns:
            str: Description détaillée structurée de l'image
        """
        print(f"\n🖼️ Analyse de l'image: {image_path}")
        
        # Vérifier si l'analyse a déjà été initialisée (par l'extraction de texte brut)
        if not self.output_saver.is_analysis_in_progress():
            self.output_saver.start_new_analysis(image_path)
        
        with open(image_path, "rb") as image_file:
            img_data = base64.b64encode(image_file.read())
        
        self._last_image_data = img_data  # Garder l'image en mémoire
        image_document = Document(image_resource=MediaResource(data=img_data))
        
        # Préparer un prompt qui inclut le texte brut déjà extrait
        enhanced_prompt = description_prompt
        if hasattr(self, 'raw_text') and self.raw_text:
            enhanced_prompt = f"""Le texte brut suivant a déjà été extrait de l'image. Utilisez-le comme référence pour votre analyse mais NE LE RECOPIEZ PAS intégralement:

TEXTE BRUT DÉJÀ EXTRAIT:
----------
{self.raw_text}
----------

{description_prompt}"""
        
        msg = ChatMessage(
            role=MessageRole.USER,
            blocks=[
                TextBlock(text=enhanced_prompt),
                ImageBlock(image=image_document.image_resource.data),
            ],
        )

        response = self.llm.chat(messages=[msg])
        result = str(response)
        
        # Supprimer le préfixe "assistant:" s'il est présent
        if result.startswith("assistant:"):
            result = result[len("assistant:"):].strip()
            
        self.vision_result = result
        
        self.output_saver.save_vision_result(self.vision_result)
        
        return self.vision_result

    def check_weekday_spelling(self, text: str) -> list:
        """
        Vérifie l'orthographe des jours de la semaine dans le texte
        
        Args:
            text: Texte à vérifier
            
        Returns:
            list: Liste des erreurs d'orthographe détectées au format JSON
        """
        if not text:
            return []
        
        # Liste des orthographes correctes
        self.correct_days = {
            "lundi": "lundi",
            "mardi": "mardi",
            "mercredi": "mercredi",
            "jeudi": "jeudi",
            "vendredi": "vendredi",
            "samedi": "samedi",
            "dimanche": "dimanche"
        }
        
        # Liste des fautes d'orthographe courantes
        common_misspellings = {
            "lumdi": "lundi",
            "marid": "mardi",
            "mecredis": "mercredi",
            "mercredis": "mercredi",
            "jedi": "jeudi",
            "jeudis": "jeudi",
            "venredi": "vendredi",
            "venredis": "vendredi",
            "vendedi": "vendredi",
            "vendredy": "vendredi",
            "samedis": "samedi",
            "dimanches": "dimanche"
        }
        
        # Erreurs détectées
        errors = []
        
        # Vérification des orthographes incorrectes
        for misspelling, correction in common_misspellings.items():
            # Utiliser une expression régulière pour rechercher le mot avec frontières de mot
            pattern = re.compile(r'\b' + re.escape(misspelling) + r'\b', re.IGNORECASE)
            matches = pattern.finditer(text)
            
            for match in matches:
                misspelled_text = match.group(0)
                errors.append({
                    "text": misspelled_text,
                    "correction": correction if misspelled_text.islower() else correction.capitalize(),
                    "position": match.start(),
                    "context": text[max(0, match.start()-15):min(len(text), match.end()+15)]
                })
        
        # Vérification supplémentaire pour les problèmes de majuscule au milieu de phrase
        for day, correct_spelling in self.correct_days.items():
            # Rechercher le jour en minuscule au début d'une phrase ou après une ponctuation
            pattern = re.compile(r'(?:^|[.!?]\s+)' + re.escape(day) + r'\b', re.IGNORECASE)
            matches = pattern.finditer(text)
            
            for match in matches:
                found_day = match.group(0).strip().lstrip('.!?').strip()
                # Vérifier si le premier caractère est en minuscule alors qu'il devrait être en majuscule
                if found_day and found_day[0].islower():
                    errors.append({
                        "text": found_day,
                        "correction": found_day.capitalize(),
                        "position": match.start(),
                        "context": text[max(0, match.start()-15):min(len(text), match.end()+15)],
                        "type": "majuscule_manquante"
                    })
            
            # Rechercher le jour en majuscule au milieu d'une phrase
            pattern = re.compile(r'(?<![.!?]\s+)\b' + re.escape(day).capitalize() + r'\b')
            matches = pattern.finditer(text)
            
            for match in matches:
                # Vérifier que ce n'est pas en début de ligne ou de phrase
                if match.start() > 0 and text[match.start()-1].isalpha():
                    errors.append({
                        "text": match.group(0),
                        "correction": match.group(0).lower(),
                        "position": match.start(),
                        "context": text[max(0, match.start()-15):min(len(text), match.end()+15)],
                        "type": "majuscule_inappropriee"
                    })
        
        return errors

    def calculate_levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calcule la distance de Levenshtein (distance d'édition) entre deux chaînes.
        Mesure le nombre minimum d'opérations (insertions, suppressions, substitutions)
        nécessaires pour transformer une chaîne en une autre.
        
        Args:
            s1: Première chaîne
            s2: Deuxième chaîne
            
        Returns:
            int: Distance de Levenshtein
        """
        if len(s1) < len(s2):
            return self.calculate_levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def check_spelling(self, text: str) -> list:
        """
        Vérifie l'orthographe de tous les mots dans un texte
        
        Args:
            text: Texte brut extrait de l'image
            
        Returns:
            list: Liste des fautes d'orthographe détectées
        """
        print("\n🔍 Vérification de l'orthographe de tous les mots...")
        
        if not text:
            return []
            
        # Liste pour stocker les erreurs trouvées
        spelling_errors = []
        
        try:
            # Vérification des jours de la semaine (gardons cette partie spécifique)
            weekday_errors = self.check_weekday_spelling(text)
            spelling_errors.extend(weekday_errors)
            
            # Rechercher les erreurs communes dans le dictionnaire de corrections
            common_errors = {
                # Mots généraux
                "publicite": "publicité",
                "telefone": "téléphone", 
                "telphone": "téléphone",
                "adrese": "adresse",
                "addresse": "adresse",
                "mangez": "manger",
                "bouger": "bouger",
                "ofre": "offre",
                "promocion": "promotion",
                "remise": "remise",
                "prix": "prix",
                "magazin": "magasin",
                "boutique": "boutique",
                "ouver": "ouvert",
                "fermé": "fermé",
                
                # Erreurs courantes en français
                "chauque": "chaque",
                "plusieur": "plusieurs", 
                "baucoup": "beaucoup",
                "maintenent": "maintenant",
                "aujourdhui": "aujourd'hui",
                "corespond": "correspond",
                "necessite": "nécessite",
                "malheuresement": "malheureusement",
                "definitivment": "définitivement",
                "notament": "notamment",
                "apareil": "appareil",
                "aparence": "apparence",
                "apartement": "appartement",
                "apeller": "appeler",
                "aplication": "application",
                "aprendre": "apprendre",
                "apres": "après",
                "ateur": "auteur",
                "batiment": "bâtiment",
                "bibliotheque": "bibliothèque",
                "biensur": "bien sûr",
                "bientot": "bientôt",
                "bizare": "bizarre",
                "boujour": "bonjour",
                "ceuillir": "cueillir",
                "chaqu'un": "chacun",
                "cherir": "chérir",
                "conaitre": "connaître",
                "conaissance": "connaissance",
                "defois": "des fois",
                "deja": "déjà",
                "dailleur": "d'ailleurs",
                "dailleurs": "d'ailleurs",
                "daccord": "d'accord",
                "dificulté": "difficulté",
                "excelent": "excellent",
                "exercise": "exercice",
                "interresant": "intéressant",
                "mechant": "méchant",
                "parcontre": "par contre",
                "peutetre": "peut-être",
                "plusieures": "plusieurs",
                "pres": "près",
                "repondre": "répondre",
                "reponce": "réponse",
                "vraiement": "vraiment",
                "voyanture": "voyante"
            }
            
            # Dictionnaire de mots français corrects pour référence
            french_words = set(common_errors.values())
            french_words.update(self.correct_days.values())
            # Ajouter d'autres mots français courants
            french_words.update([
                "le", "la", "les", "un", "une", "des", "et", "ou", "à", "au", "aux", "du", "des",
                "ce", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses",
                "notre", "nos", "votre", "vos", "leur", "leurs", "je", "tu", "il", "elle", "nous",
                "vous", "ils", "elles", "qui", "que", "quoi", "dont", "où", "comment", "pourquoi",
                "quand", "être", "avoir", "faire", "dire", "aller", "voir", "savoir", "pouvoir",
                "vouloir", "falloir", "valoir", "prendre", "mettre", "passer", "devoir", "venir",
                "tenir", "mais", "ou", "et", "donc", "or", "ni", "car", "si", "grand", "petit",
                "beau", "joli", "bon", "mauvais", "vieux", "jeune", "nouveau", "dernier", "premier",
                "tout", "chaque", "certain", "même", "autre", "tel", "quel", "quelque", "ceci", "cela",
                "rien", "personne", "quelqu'un", "chacun", "aucun", "nul", "tous", "plusieurs"
            ])
            
            # Rechercher les erreurs communes dans le texte
            for incorrect, correct in common_errors.items():
                matches = re.finditer(r'\b' + re.escape(incorrect) + r'\b', text.lower())
                for match in matches:
                    original_word = text[match.start():match.end()]
                    if original_word.lower() == incorrect:  # Vérifier que c'est bien le mot incorrect
                        # Déterminer la casse du mot correct en fonction du mot original
                        if original_word.isupper():
                            corrected = correct.upper()
                        elif original_word[0].isupper():
                            corrected = correct.capitalize()
                        else:
                            corrected = correct
                            
                        spelling_errors.append({
                            "incorrect": original_word,
                            "correct": corrected,
                            "position": match.start()
                        })
            
            # Analyser tous les mots du texte pour trouver d'autres erreurs potentielles
            words = re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', text)  # Mots d'au moins 3 lettres
            
            for word in words:
                # Ignorer les mots déjà identifiés comme incorrects
                if any(error["incorrect"].lower() == word.lower() for error in spelling_errors):
                    continue
                
                word_lower = word.lower()
                
                # Ignorer les mots corrects déjà connus
                if word_lower in french_words:
                    continue
                
                # Rechercher le mot le plus proche dans notre dictionnaire
                min_distance = float('inf')
                closest_word = None
                
                # Ne vérifier que les mots de longueur similaire pour optimiser
                for correct_word in french_words:
                    # Ne considérer que les mots dont la longueur est proche
                    if abs(len(word_lower) - len(correct_word)) > 2:
                        continue
                    
                    # Calculer la distance de Levenshtein
                    distance = self.calculate_levenshtein_distance(word_lower, correct_word)
                    
                    # Normaliser la distance par rapport à la longueur du mot
                    normalized_distance = distance / max(len(word_lower), len(correct_word))
                    
                    # Si la distance est faible (mot similaire) et meilleure que précédemment
                    if normalized_distance < 0.25 and distance < min_distance:
                        min_distance = distance
                        closest_word = correct_word
                
                # Si on a trouvé un mot similaire, considérer comme une erreur
                if closest_word and min_distance <= 2:  # Max 2 opérations d'édition
                    # Déterminer la casse du mot correct en fonction du mot original
                    if word.isupper():
                        corrected = closest_word.upper()
                    elif word[0].isupper():
                        corrected = closest_word.capitalize()
                    else:
                        corrected = closest_word
                    
                    # Trouver la position dans le texte
                    positions = [m.start() for m in re.finditer(r'\b' + re.escape(word) + r'\b', text)]
                    for position in positions:
                        spelling_errors.append({
                            "incorrect": word,
                            "correct": corrected,
                            "position": position,
                            "confidence": 1 - (min_distance / max(len(word), len(closest_word)))
                        })
            
            # Tri par position dans le texte et suppression des doublons
            spelling_errors.sort(key=lambda x: x["position"])
            
            # Supprimer les doublons (garder seulement la première occurrence)
            unique_errors = []
            seen = set()
            for error in spelling_errors:
                key = (error["incorrect"].lower(), error["position"])
                if key not in seen:
                    seen.add(key)
                    unique_errors.append(error)
            
            spelling_errors = unique_errors
            
            # Afficher le résultat
            if spelling_errors:
                print(f"⚠️ {len(spelling_errors)} fautes d'orthographe détectées:")
                for error in spelling_errors[:10]:  # Limiter l'affichage aux 10 premières erreurs
                    confidence_info = f" (confiance: {error['confidence']:.2%})" if 'confidence' in error else ""
                    print(f"  - '{error['incorrect']}' devrait être '{error['correct']}'{confidence_info}")
                if len(spelling_errors) > 10:
                    print(f"  - ... et {len(spelling_errors) - 10} autres fautes")
            else:
                print("✅ Aucune faute d'orthographe détectée")
                
            return spelling_errors
            
        except Exception as e:
            print(f"❌ Erreur lors de la vérification orthographique: {str(e)}")
            return spelling_errors  # Retourner les erreurs trouvées jusqu'à présent

    def verify_consistency(self, vision_result: str) -> str:
        """
        Vérification de la cohérence des informations sur l'image
        
        Args:
            vision_result: Résultat de l'analyse visuelle
            
        Returns:
            str: Rapport de vérification des cohérences
        """
        print("\n🔍 Vérification de la cohérence...")
        
        if not hasattr(self, 'raw_text') or not self.raw_text:
            raise ValueError("Vous devez d'abord extraire le texte brut de l'image")
        
        # Stocker les résultats précédents
        self.vision_result = vision_result
        
        # Obtenir la date actuelle au format français
        current_date = datetime.now().strftime("%d/%m/%Y")
        
        # Vérifier l'orthographe des jours de la semaine
        spelling_errors = self.check_weekday_spelling(self.raw_text)
        
        # Vérifier les prix et réductions
        price_errors = self.check_price_consistency(self.raw_text)
        
        # Préparer la section des erreurs d'orthographe
        spelling_errors_section = ""
        if spelling_errors:
            spelling_errors_section = "\nERREURS D'ORTHOGRAPHE DÉTECTÉES (à inclure dans votre analyse):\n"
            for i, error in enumerate(spelling_errors, 1):
                if "type" in error and error["type"] == "majuscule_manquante":
                    spelling_errors_section += f"{i}. ERREUR DE MAJUSCULE: '{error['text']}' devrait s'écrire '{error['correction']}' en début de phrase\n"
                elif "type" in error and error["type"] == "majuscule_inappropriee":
                    spelling_errors_section += f"{i}. ERREUR DE MAJUSCULE: '{error['text']}' devrait s'écrire '{error['correction']}' en milieu de phrase\n"
                else:
                    spelling_errors_section += f"{i}. ERREUR D'ORTHOGRAPHE: '{error['text']}' devrait s'écrire '{error['correction']}' (contexte: '{error['context']}')\n"
        
        # Préparer la section des erreurs de prix
        price_errors_section = ""
        if price_errors:
            price_errors_section = "\nINCOHÉRENCES DE PRIX DÉTECTÉES (à inclure obligatoirement dans votre analyse):\n"
            for i, error in enumerate(price_errors, 1):
                if error["type"] == "prix_supérieur" or error["type"] == "prix_supérieur_avec_réduction":
                    price_errors_section += f"{i}. ERREUR CRITIQUE: Le prix après réduction ({error['prix_réduit']}€) est SUPÉRIEUR au prix initial ({error['prix_initial']}€) dans '{error['texte_original']}'\n"
                elif error["type"] == "calcul_incorrect":
                    price_errors_section += f"{i}. ERREUR DE CALCUL: Pour une réduction de {error['pourcentage_réduction']}% sur {error['prix_initial']}€, le prix affiché est {error['prix_affiché']}€ alors qu'il devrait être {error['prix_calculé']}€\n"
                elif error["type"] == "prix_barré_incohérent":
                    price_errors_section += f"{i}. ERREUR CRITIQUE: Le prix réduit ({error['prix_réduit']}€) est SUPÉRIEUR au prix barré initial ({error['prix_initial']}€)\n"
        
        # Préparer la section du texte brut
        raw_text_section = "TEXTE BRUT EXTRAIT DE L'IMAGE (référence exacte pour la vérification):\n\n"
        raw_text_section += self.raw_text + "\n\n"
        
        # Créer le prompt final
        enhanced_prompt = f"""{raw_text_section}{spelling_errors_section}{price_errors_section}

{consistency_prompt.format(vision_result=vision_result, current_date=current_date)}"""
        
        msg = ChatMessage(
            role=MessageRole.USER,
            blocks=[
                TextBlock(text=enhanced_prompt),
                ImageBlock(image=self._last_image_data),
            ],
        )
        
        response = self.llm.chat(messages=[msg])
        result = str(response)
        
        # Supprimer le préfixe "assistant:" s'il est présent
        if result.startswith("assistant:"):
            result = result[len("assistant:"):].strip()
        
        # Stocker les erreurs pour une utilisation ultérieure dans analyze_compliance
        self.spelling_errors = spelling_errors
        self.price_errors = price_errors
            
        # Sauvegarder le résultat
        self.output_saver.save_consistency_check(result)
            
        return result

    def verify_dates(self, text: str, **kwargs) -> dict:
        """
        Vérifie les dates mentionnées dans le texte et leur cohérence.
        Détecte les incohérences entre jours de la semaine et dates.
        
        Args:
            text: Le texte à analyser
            
        Returns:
            dict: Résultat de l'analyse des dates
        """
        print("🔍 Vérification des dates dans le texte")
        
        # Récupérer la date actuelle
        current_date = datetime.now()
        
        # Année en cours pour vérification des dates
        current_year = 2025
        
        # Dictionnaire normalisant les jours de la semaine
        jours_semaine = {
            'lundi': 0, 'mardi': 1, 'mercredi': 2, 'jeudi': 3, 
            'vendredi': 4, 'samedi': 5, 'dimanche': 6
        }
        
        # Jours avec accents
        jours_semaine_accents = {
            'lundi': 0, 'mardi': 1, 'mercredi': 2, 'jeudi': 3, 
            'vendredi': 4, 'samedi': 5, 'dimanche': 6
        }
        
        # Map des nombres en jours
        map_numero_jour = {
            0: "lundi", 1: "mardi", 2: "mercredi", 3: "jeudi",
            4: "vendredi", 5: "samedi", 6: "dimanche"
        }
        
        # --- NOUVELLE LOGIQUE POUR DATES EN TOUTES LETTRES ET PLAGES ---
        mois_fr = [
            "janvier", "février", "fevrier", "mars", "avril", "mai", "juin", "juillet", "août", "aout", "septembre", "octobre", "novembre", "décembre", "decembre"
        ]
        mois_map = {m: i+1 for i, m in enumerate([
            "janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"
        ])}
        mois_map["fevrier"] = 2
        mois_map["aout"] = 8
        mois_map["decembre"] = 12
        
        # Chercher les jours listés (ex: "vendredi et samedi")
        jours_regex = r"(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)(?:\s*et\s*(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche))*"
        jours_matches = re.findall(jours_regex, text, flags=re.IGNORECASE)
        jours_list = []
        for match in jours_matches:
            jours_list.extend([j for j in match if j])
        jours_list = [j.lower() for j in jours_list if j]
        
        # Chercher les dates en toutes lettres (ex: "27 septembre", "28 septembre")
        date_lettres_regex = r"(\d{1,2})\s*(janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)"
        date_lettres_matches = re.findall(date_lettres_regex, text, flags=re.IGNORECASE)
        dates_list = []
        for jour_num, mois in date_lettres_matches:
            mois_num = mois_map[mois.lower()]
            dates_list.append((int(jour_num), mois_num, mois))
        
        # Si on a autant de jours que de dates, on les associe dans l'ordre
        couples = []
        if len(jours_list) == len(dates_list) and len(jours_list) > 0:
            for i in range(len(jours_list)):
                couples.append((jours_list[i], dates_list[i]))
        # Si un seul mois pour plusieurs jours/dates, on fait toutes les combinaisons
        elif len(jours_list) > 0 and len(dates_list) > 0:
            for jour in jours_list:
                for date in dates_list:
                    couples.append((jour, date))
        # Sinon, on continue avec la logique classique
        
        # Vérifier la cohérence pour chaque couple jour/date
        date_errors = []
        for jour, (jour_num, mois_num, mois_nom) in couples:
            try:
                date_obj = datetime(current_year, mois_num, jour_num)
                weekday_num = date_obj.weekday()
                mentioned_weekday_num = jours_semaine[jour.lower()]
                if weekday_num != mentioned_weekday_num:
                    correct_day = map_numero_jour[weekday_num]
                    date_errors.append({
                        "date": f"{jour_num} {mois_nom}",
                        "stated_day": jour,
                        "correct_day": correct_day,
                        "description": f"Le {jour} {jour_num} {mois_nom} {current_year} est un {correct_day}, pas un {jour}."
                    })
                if date_obj.date() < current_date.date():
                    date_errors.append({
                        "date": f"{jour_num} {mois_nom}",
                        "type": "expired",
                        "description": f"La date {jour_num} {mois_nom} {current_year} est déjà passée (date actuelle: {current_date.strftime('%d/%m/%Y')})."
                    })
            except Exception as e:
                date_errors.append({
                    "date": f"{jour_num} {mois_nom}",
                    "stated_day": jour,
                    "error": str(e),
                    "description": f"Erreur d'analyse de date: {str(e)}"
                })
        # --- FIN NOUVELLE LOGIQUE ---
        
        # Sauvegarder les erreurs pour utilisation ailleurs
        self.weekday_errors = date_errors
        
        # Résultat de l'analyse
        result = {
            "dates_found": couples,
            "weekday_errors": date_errors,
            "total_errors": len(date_errors)
        }
        
        if date_errors:
            error_descriptions = "\n".join([f"- {err['description']}" for err in date_errors])
            result["summary"] = f"⚠️ {len(date_errors)} incohérence(s) détectée(s) dans les dates:\n{error_descriptions}"
        else:
            result["summary"] = "✓ Aucune incohérence détectée dans les dates."
        
        # Sauvegarder le résultat
        if hasattr(self, 'output_saver'):
            self.output_saver.save_output('dates_verification', result)
        
        return result

    def search_legislation(self, vision_result: str) -> str:
        """
        Recherche la législation applicable
        Args:
            vision_result: Résultat de l'analyse visuelle
        Returns:
            str: Législation applicable
        """
        print("\n🔍 Recherche de législation...")
        print(f"Vision result utilisé pour la recherche: {vision_result[:200]}...")
        
        try:
            # Rechercher dans la base de connaissances
            raw_legislation = self.raptor.search(vision_result)
            print(f"\nLégislation brute trouvée: {raw_legislation[:200]}...")
            
            # Stocker la législation brute
            self.legislation = raw_legislation
            
            # Utiliser le query engine pour synthétiser la réponse
            query = f"""Analyser et synthétiser la législation suivante dans le contexte de cette publicité :
            
            CONTEXTE PUBLICITAIRE :
            {vision_result}
            
            LÉGISLATION TROUVÉE :
            {raw_legislation}
            """
            
            synthesis = self.raptor.query(query)
            print(f"\nSynthèse de la législation: {synthesis[:200]}...")
            
            self.output_saver.save_legislation(synthesis)
            
            return synthesis
            
        except Exception as e:
            print(f"\n❌ Erreur lors de la recherche de législation: {str(e)}")
            # En cas d'erreur, utiliser la législation brute si disponible
            if raw_legislation:
                return raw_legislation
            raise

    def get_clarifications(self, questions_text: str) -> str:
        """
        Obtient des clarifications spécifiques en analysant l'image
        Args:
            questions_text: Questions spécifiques nécessitant des clarifications
        Returns:
            str: Réponses aux questions de clarification
        """
        print("\n❓ Obtention des clarifications...")
        
        if not self.vision_result or not self.legislation:
            raise ValueError("L'analyse visuelle et la recherche de législation doivent être effectuées d'abord")
        
        # Initialiser l'historique des clarifications si nécessaire
        if not hasattr(self, '_clarifications_history'):
            self._clarifications_history = set()
        
        # Vérifier si la question a déjà été posée
        if questions_text in self._clarifications_history:
            print("⚠️ Cette clarification a déjà été demandée")
            return "Cette question a déjà été posée. Veuillez demander des clarifications sur d'autres aspects ou passer à l'analyse de conformité."
        
        # Ajouter la question à l'historique
        self._clarifications_history.add(questions_text)
        
        # Créer le message multimodal avec l'image
        msg = ChatMessage(
            role=MessageRole.USER,
            blocks=[
                TextBlock(text=clarifications_prompt.format(questions_text=questions_text)),
                ImageBlock(image=self._last_image_data),
            ],
        )
        
        print("\nEnvoi de l'image et des questions au LLM...")
        response = self.llm.chat(messages=[msg])
        result = str(response)
        
        # Supprimer le préfixe "assistant:" s'il est présent
        if result.startswith("assistant:"):
            result = result[len("assistant:"):].strip()
        
        self.output_saver.save_clarifications(result)
        
        return result

    def analyze_compliance(self, 
                        vision_result: str = None, 
                        consistency_result: str = None, 
                        legislation_result: str = None,
                        product_logo_analysis: str = None,
                        dates_verification: dict = None,
                        raw_text: str = None,
                        **kwargs) -> str:
        """
        Analyse la conformité de la publicité en fonction des différentes analyses
        
        Args:
            vision_result: Résultat de l'analyse visuelle
            consistency_result: Résultat de la vérification de cohérence
            legislation_result: Résultat de la recherche de législation
            product_logo_analysis: Résultat de l'analyse de cohérence produit/logo
            dates_verification: Résultat de la vérification des dates (format dictionnaire)
            raw_text: Texte brut extrait de l'image
            
        Returns:
            str: Rapport de conformité
        """
        
        # Récupérer les résultats d'analyse précédents si non fournis
        if not vision_result and hasattr(self, 'vision_result'):
            vision_result = self.vision_result
            
        if not consistency_result and hasattr(self, 'consistency_result'):
            consistency_result = self.consistency_result
            
        if not legislation_result and hasattr(self, 'legislation_result'):
            legislation_result = self.legislation_result
            
        if not product_logo_analysis and hasattr(self, 'product_logo_analysis'):
            product_logo_analysis = self.product_logo_analysis
            
        if not dates_verification and hasattr(self, 'weekday_errors'):
            dates_verification = {
                "weekday_errors": self.weekday_errors,
                "total_errors": len(self.weekday_errors) if self.weekday_errors else 0
            }
            
        if not raw_text and hasattr(self, 'raw_text'):
            raw_text = self.raw_text
            
        # Initialiser toutes les variables utilisées dans le prompt pour éviter les erreurs Python
        date_info = ""
        price_errors_info = ""
        weekday_errors_info = ""
        price_errors_summary = ""
        date_errors_summary = ""
        weekday_errors_summary = ""
        non_transformed_products = False
        products_list = []

        # Liste complète des mentions légales nutritionnelles PNNS à surveiller
        pnns_mentions = [
            "mangerbouger.fr",
            "pour votre santé, mangez au moins cinq fruits et légumes par jour",
            "pour votre santé, évitez de manger trop gras, trop sucré, trop salé",
            "pour votre santé, pratiquez une activité physique régulière",
            "pour votre santé, évitez de grignoter entre les repas",
            "pour votre santé, évitez de consommer trop de sel",
            "pour votre santé, limitez les produits sucrés",
            "pour votre santé, limitez les produits gras",
            "pour votre santé, limitez les produits salés",
            "pour votre santé, limitez la consommation d'alcool"
        ]

        # Texte à analyser pour la mention et les produits : raw_text si possible, sinon vision_result
        texte_a_analyser = raw_text if raw_text and not raw_text.startswith('ERREUR') else vision_result if vision_result else ""

        # Vérifier la présence d'au moins une mention PNNS dans le texte analysé
        mentions_pnns_trouvees = [m for m in pnns_mentions if m in texte_a_analyser.lower()]
        mention_pnns_presente = len(mentions_pnns_trouvees) > 0

        # Détection des produits transformés et non transformés
        produits_transformes = False
        tous_non_transformes = False
        produits_detectes = []
        if hasattr(self, 'logo_product_matcher') and texte_a_analyser:
            products = self.logo_product_matcher.extract_products_from_text(texte_a_analyser)
            produits_detectes = products
            if products:
                tous_non_transformes = self.logo_product_matcher.is_non_transformed_product(products)
                # On considère qu'il y a des produits transformés si la liste n'est pas tous non transformés
                produits_transformes = not tous_non_transformes

        # Liste des non-conformités spécifiques à ajouter
        non_conformites = []
        # Cas 1 : au moins un produit transformé, mention PNNS absente
        if produits_transformes and not mention_pnns_presente:
            non_conformites.append(
                "Absence de mention légale nutritionnelle obligatoire (PNNS) alors que des produits transformés sont présents."
            )
        # Cas 2 : tous non transformés, mention PNNS présente
        if tous_non_transformes and mention_pnns_presente:
            produits_str = ", ".join(produits_detectes) if produits_detectes else "(non détectés)"
            non_conformites.append(
                f"Non-conformité : la mention légale nutritionnelle (PNNS) est présente alors qu'aucun produit transformé n'est détecté (ex : {produits_str}). Cette mention ne doit pas figurer pour des produits non transformés."
            )
        # Cas 3 : mixte (au moins un transformé et un non transformé), la mention PNNS est obligatoire, ne pas signaler la présence
        # (déjà couvert par la logique ci-dessus)

        # Vérification de la présence du numéro RCS et du site internet dans le texte analysé
        rcs_present = bool(re.search(r"\bRCS\b", texte_a_analyser, re.IGNORECASE))
        site_present = bool(re.search(r"\bhttps?://|www\.[a-z0-9\-]+\.[a-z]{2,}\b", texte_a_analyser, re.IGNORECASE))
        # Exclure www.mangerbouger.fr du site internet de l'entreprise
        site_present = site_present and not re.search(r"www\.mangerbouger\.fr", texte_a_analyser, re.IGNORECASE)
        # Signaler explicitement l'absence de RCS et/ou de site internet
        if not rcs_present:
            non_conformites.append("Absence de numéro RCS : la publicité doit comporter le numéro RCS de l'entreprise.")
        if not site_present:
            non_conformites.append("Absence de site internet de l'entreprise : aucun site internet spécifique à l'annonceur n'est mentionné.")

        # --- Point d'entrée pour la vérification avancée via RAG ---
        if legislation_result:
            rag_non_conformities = self.check_rag_legislation(legislation_result, texte_a_analyser, produits_detectes)
            non_conformites.extend(rag_non_conformities)
        # --- Fin point d'entrée RAG ---

        # Prompt de base pour l'analyse de conformité
        prompt = f"""Analyse complète de la conformité de cette publicité selon la législation publicitaire:

DONNÉES D'ANALYSE VISUELLE:
{vision_result}

VÉRIFICATION DE COHÉRENCE:
{consistency_result}

LÉGISLATION APPLICABLE:
{legislation_result}

ANALYSE DE COHÉRENCE PRODUIT/LOGO:
{product_logo_analysis}

{date_info}
{price_errors_info}
{weekday_errors_info}

TEXTE BRUT EXTRAIT:
{raw_text if raw_text else "Non disponible"}

{price_errors_summary}
{date_errors_summary}
{weekday_errors_summary}

RAPPEL IMPORTANT:
- NE PAS recommander inutilement d'ajouter une adresse pour l'établissement si ce n'est pas obligatoire
- L'adresse de l'établissement N'EST PAS OBLIGATOIRE pour les publicités standards
- RETIRER toute recommandation d'ajout d'adresse qui ne soit pas légalement requise
- LE NUMÉRO DE TÉLÉPHONE N'EST PAS OBLIGATOIRE pour les publicités standards
- NE PAS inclure de section "RECOMMANDATIONS" dans le rapport final
- Se concentrer UNIQUEMENT sur les non-conformités légales réelles
- Pour les viandes, les étoiles (★,☆,✩,✪) indiquent la qualité de la viande et NE SONT PAS des astérisques nécessitant un renvoi
- VÉRIFIER ATTENTIVEMENT les mentions d'origine des produits:
  * "pêché en Loire Atlantique" ou similaire pour des produits de viande = NON-CONFORMITÉ MAJEURE
  * "Le Porc Français" pour des produits qui ne sont pas du porc = NON-CONFORMITÉ MAJEURE 
  * "Le Bœuf Français" pour des produits qui ne sont pas du bœuf = NON-CONFORMITÉ MAJEURE
  * Toute incohérence entre l'origine déclarée et le type de produit = NON-CONFORMITÉ MAJEURE
- POUR LES DATES:
  * NE PAS recommander d'ajouter l'année aux dates - ce n'est PAS nécessaire
  * Pour vérifier la cohérence des dates sans année mentionnée, utiliser l'année en cours (2025)
  * Vérifier UNIQUEMENT la cohérence entre jour de la semaine et date (ex: si "Vendredi 12/05" est cohérent en 2025)
  * NE PAS considérer l'absence d'année dans une date comme une non-conformité
"""

        # Initialiser la variable prompt_reminder
        prompt_reminder = ""

        # Ajouter l'information sur les produits non transformés si détectés
        if non_transformed_products:
            produits_detectes = ", ".join(products_list)
            prompt_reminder += f"""
INFORMATION CRITIQUE SUR LES PRODUITS NON TRANSFORMÉS:
- Des produits non transformés ont été détectés: {produits_detectes}
- Les produits non transformés (viande fraîche, poisson frais, fruits et légumes frais) sont EXEMPTÉS de la mention www.mangerbouger.fr
- NE PAS signaler l'absence de mention www.mangerbouger.fr comme une non-conformité
- NE PAS recommander d'ajouter la mention www.mangerbouger.fr dans ce cas
"""
        
        # RAPPEL CRITIQUE concernant les erreurs de prix
        if price_errors_info:
            prompt_reminder += """
RAPPEL CRITIQUE SUR LES PRIX:
- CONSIDÉRER COMME NON-CONFORMITÉ MAJEURE tout prix après réduction supérieur au prix initial
- INCLURE OBLIGATOIREMENT les erreurs de prix détectées dans la liste des non-conformités
- EXPLIQUER avec précision le calcul correct qui aurait dû être fait
"""
        
        # RAPPEL CRITIQUE concernant les erreurs de jours/dates
        if weekday_errors_info or date_info:
            prompt_reminder += """
RAPPEL CRITIQUE SUR LES DATES:
- CONSIDÉRER COMME NON-CONFORMITÉ MAJEURE toute incohérence entre date et jour de la semaine
- INCLURE OBLIGATOIREMENT les erreurs de dates détectées dans la liste des non-conformités
- SPÉCIFIER le jour correct qui correspond à chaque date mentionnée
- VÉRIFIER LA COHÉRENCE avec l'année en cours (2025) pour les dates sans année
- NE PAS recommander d'ajouter l'année aux dates - ce n'est PAS nécessaire
"""
        
        prompt += prompt_reminder
        
        # Ajouter les non-conformités spécifiques dans le prompt final
        if non_conformites:
            prompt += "\n\nNON-CONFORMITÉS SPÉCIFIQUES DÉTECTÉES :\n"
            for nc in non_conformites:
                prompt += f"- {nc}\n"
        
        # Utiliser le LLM pour analyser la conformité
        response = self.llm.complete(prompt)
        result = str(response)
        
        # Supprimer le préfixe "assistant:" s'il est présent
        if result.startswith("assistant:"):
            result = result[len("assistant:"):].strip()
        
        # Sauvegarder le résultat
        if hasattr(self, 'output_saver'):
            try:
                self.output_saver.save_output('compliance_analysis', result)
            except AttributeError:
                # Utiliser une autre méthode de sauvegarde si save_output n'existe pas
                self.output_saver.save_compliance_analysis(result)
        
        return result

    def extract_text_from_image(self, image_path: str, mode: str = "docling", ocr_engine: str = "tesseract") -> str:
        """
        Extrait le texte visible dans une image publicitaire
        
        Args:
            image_path: Chemin vers l'image à analyser
            mode: Mode d'extraction ('docling', 'pytesseract', 'easyocr')
            ocr_engine: Moteur OCR à utiliser avec Docling ('tesseract', 'easyocr', 'rapidocr')
            
        Returns:
            str: Texte extrait de l'image
        """
        print(f"\n🔤 Extraction du texte de l'image avec {mode}: {image_path}")
        
        # Configurer les options d'extraction selon le mode
        options = {}
        if mode == "docling":
            try:
                # Options avancées pour l'extraction Docling
                extracted_text = self.text_extractor.extract_text_with_docling(
                    image_path, 
                    ocr_engine=ocr_engine,
                    custom_options=options
                )
            except Exception as e:
                print(f"⚠️ Erreur avec Docling: {str(e)}. Essai d'une méthode alternative...")
                # Fallback vers une autre méthode
                extracted_text = self.text_extractor.extract_text(image_path, fallback=True)
        elif mode == "pytesseract":
            extracted_text = self.text_extractor.extract_text_with_pytesseract(image_path)
        elif mode == "easyocr":
            extracted_text = self.text_extractor.extract_text_with_easyocr_direct(image_path)
        else:
            print(f"⚠️ Mode {mode} non supporté, utilisation de la méthode générique")
            extracted_text = self.text_extractor.extract_text(image_path, fallback=True)
        
        # Si le texte est vide, afficher un avertissement
        if not extracted_text or len(extracted_text.strip()) < 5:
            print("⚠️ Attention: Très peu ou pas de texte extrait de l'image.")
        else:
            print(f"✅ Texte extrait ({len(extracted_text)} caractères)")
            
        # Sauvegarder des métadonnées supplémentaires pour l'analyse
        metadata = {
            "mode": mode,
            "ocr_engine": ocr_engine if mode == "docling" else "N/A",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "char_count": len(extracted_text),
            "success": bool(extracted_text and len(extracted_text.strip()) > 5)
        }
        
        # Sauvegarder le résultat dans les sorties
        self.output_saver.save_text_extraction(extracted_text, mode)
        self.extracted_text = extracted_text
        
        return extracted_text 

    def extract_raw_text_with_vision(self, image_path: str) -> str:
        """
        Extrait le texte brut d'une image en utilisant un fallback OCR robuste (Tesseract/EasyOCR)
        Args:
            image_path: Chemin de l'image
        Returns:
            str: Texte brut extrait
        """
        print(f"\n📝 Extraction du texte brut pour l'agent (fallback OCR): {image_path}")
        try:
            # Utiliser l'OCR local (Tesseract/EasyOCR) avec fallback
            raw_text = self.text_extractor.extract_text(image_path, fallback=True)
            if raw_text:
                print("\n💾 Texte brut sauvegardé dans le fichier JSON principal")
                return raw_text
            else:
                return "Aucun texte extrait"
        except Exception as e:
            print(f"❌ Erreur lors de l'extraction de texte brut: {str(e)}")
            return f"ERREUR: {str(e)}"

    def extract_raw_text_for_agent(self, image_path: str) -> str:
        """
        Extrait le texte brut d'une image publicitaire pour l'agent ReACT
        
        Args:
            image_path: Chemin vers l'image à analyser
            
        Returns:
            str: Texte brut extrait
        """
        print(f"\n📝 Extraction du texte brut pour l'agent: {image_path}")
        
        try:
            # Vérifier que l'image existe
            if not os.path.exists(image_path):
                error_msg = f"❌ Image non trouvée: {image_path}"
                print(error_msg)
                return error_msg
            
            # Initialiser une nouvelle analyse - Important: doit être fait AVANT d'essayer de sauvegarder des résultats
            self.output_saver.start_new_analysis(image_path)
            
            # Utiliser GPT Vision pour l'extraction
            result = self.extract_raw_text_with_vision(image_path)
            
            # Vérifier que le résultat n'est pas vide
            if not result or len(result.strip()) < 10:
                print("⚠️ Texte extrait trop court ou vide, mais continuons l'analyse")
            
            # Sauvegarder dans les données de l'analyse
            self.raw_text = result
            
            # Sauvegarder dans l'output_saver
            self.output_saver.save_raw_text(result)
            
            print("✅ Extraction de texte brut réussie")
            return result
            
        except Exception as e:
            error_msg = f"❌ Erreur lors de l'extraction du texte brut: {str(e)}"
            print(error_msg)
            # Même en cas d'erreur, on continue l'analyse
            print("⚠️ Continuez avec l'analyse visuelle malgré l'erreur d'extraction")
            return error_msg 

    def verify_product_logo_consistency(self, vision_result: str = None) -> str:
        """
        Vérifie la cohérence entre les logos et les produits mentionnés dans la publicité
        
        Args:
            vision_result: Résultat de l'analyse visuelle (optionnel)
            
        Returns:
            str: Rapport de vérification des cohérences logo-produit
        """
        print("\n🔍 Vérification de la cohérence entre logos et produits...")
        
        if not vision_result and not self.vision_result:
            raise ValueError("L'analyse visuelle doit être effectuée d'abord")
            
        vision_content = vision_result if vision_result else self.vision_result
        
        # Texte complet à analyser (combinaison de l'analyse visuelle et du texte brut)
        full_text = vision_content
        if hasattr(self, 'raw_text') and self.raw_text:
            full_text = full_text + "\n\n" + self.raw_text
        
        # Extraire les produits et logos mentionnés dans le texte
        products = self.logo_product_matcher.extract_products_from_text(full_text)
        logos = self.logo_product_matcher.extract_logos_from_text(full_text)
        
        # Vérifier la cohérence
        inconsistencies = self.logo_product_matcher.check_product_logo_consistency(logos, products)
        self.product_logo_inconsistencies = inconsistencies
        
        # Construire le rapport
        if not inconsistencies:
            result = "VÉRIFICATION PRODUITS/LOGOS : COHÉRENT\n\nAucune incohérence détectée entre les logos et les produits mentionnés."
            if logos:
                result += f"\n\nLogos détectés ({len(logos)}) : {', '.join(logos)}"
            if products:
                result += f"\n\nProduits détectés ({len(products)}) : {', '.join(products)}"
        else:
            result = "VÉRIFICATION PRODUITS/LOGOS : NON COHÉRENT\n\nIncohérences détectées :\n"
            for i, inconsistency in enumerate(inconsistencies, 1):
                result += f"{i}. Le logo '{inconsistency['logo']}' n'est pas compatible avec les produits suivants : {', '.join(inconsistency['products'])}\n"
                result += f"   → Catégories autorisées pour ce logo : {', '.join(inconsistency['allowed_categories'])}\n\n"
            
            result += "RECOMMANDATION : Corriger ces incohérences en remplaçant les logos par des logos appropriés pour les produits concernés."
        
        # Sauvegarder le résultat
        self.output_saver.save_product_logo_consistency(result)
        
        return result 

    def check_price_consistency(self, text: str) -> list:
        """
        Vérifie la cohérence des prix et des réductions dans une publicité.
        Détecte notamment les prix après réduction supérieurs ou égaux aux prix initiaux.
        
        Args:
            text: Texte brut extrait de l'image
            
        Returns:
            list: Liste des incohérences de prix détectées
        """
        print("\n💰 Vérification de la cohérence des prix et réductions...")
        
        if not text:
            return []
            
        # Liste pour stocker les erreurs trouvées
        price_errors = []
        
        # Chercher des motifs de prix et de réductions
        import re
        
        # 1. Recherche directe de prix avant/après réduction
        # Format courant: prix initial X€ -> prix réduit Y€
        price_pairs = re.findall(r'(\d+[,.]\d+|\d+)(\s*€|\s*EUR|\s*euros?)(?:\s*[-–—>→]+\s*)(\d+[,.]\d+|\d+)(\s*€|\s*EUR|\s*euros?)', text)
        
        for match in price_pairs:
            try:
                # Extraire et normaliser les prix (remplacer virgule par point)
                initial_price = float(match[0].replace(',', '.'))
                reduced_price = float(match[2].replace(',', '.'))
                
                # Vérifier si le prix réduit est supérieur ou égal au prix initial
                if reduced_price >= initial_price:
                    error_info = {
                        "type": "prix_supérieur",
                        "prix_initial": initial_price,
                        "prix_réduit": reduced_price,
                        "différence": reduced_price - initial_price,
                        "pourcentage": 100 * (reduced_price - initial_price) / initial_price if initial_price > 0 else 0,
                        "texte_original": f"{match[0]}{match[1]} -> {match[2]}{match[3]}"
                    }
                    price_errors.append(error_info)
                    print(f"⚠️ Prix incohérent: {error_info['texte_original']} - Le prix après réduction est plus élevé que le prix initial")
            except ValueError:
                # Ignorer si la conversion en float échoue
                continue
        
        # 2. Recherche de prix avec pourcentage de réduction explicite
        # Format courant: prix X€ -Y% -> prix Z€
        discount_patterns = re.findall(r'(\d+[,.]\d+|\d+)(\s*€|\s*EUR|\s*euros?)\s*[-–—]\s*(\d+)(\s*%)\s*(?:[-–—>→]+)\s*(\d+[,.]\d+|\d+)(\s*€|\s*EUR|\s*euros?)', text)
        
        for match in discount_patterns:
            try:
                initial_price = float(match[0].replace(',', '.'))
                discount_pct = float(match[2])
                final_price = float(match[4].replace(',', '.'))
                
                # Calculer le prix réduit attendu
                expected_price = round(initial_price * (1 - discount_pct/100), 2)
                
                # Tolérance d'arrondi (1 centime)
                tolerance = 0.01
                
                # Vérifier si le prix affiché est supérieur au prix initial
                if final_price >= initial_price:
                    error_info = {
                        "type": "prix_supérieur_avec_réduction",
                        "prix_initial": initial_price,
                        "pourcentage_réduction": discount_pct,
                        "prix_affiché": final_price,
                        "prix_calculé": expected_price,
                        "différence": final_price - initial_price,
                        "texte_original": f"{match[0]}{match[1]} -{match[2]}% -> {match[4]}{match[5]}"
                    }
                    price_errors.append(error_info)
                    print(f"⚠️ Prix incohérent: {error_info['texte_original']} - Le prix après réduction est plus élevé que le prix initial")
                
                # Vérifier si le prix affiché correspond au calcul de la réduction
                elif abs(final_price - expected_price) > tolerance:
                    error_info = {
                        "type": "calcul_incorrect",
                        "prix_initial": initial_price,
                        "pourcentage_réduction": discount_pct,
                        "prix_affiché": final_price,
                        "prix_calculé": expected_price,
                        "différence": final_price - expected_price,
                        "texte_original": f"{match[0]}{match[1]} -{match[2]}% -> {match[4]}{match[5]}"
                    }
                    price_errors.append(error_info)
                    print(f"⚠️ Calcul de réduction incorrect: {error_info['texte_original']} - Le prix devrait être {expected_price}€")
            except ValueError:
                continue
        
        # 3. Recherche de prix barrés suivis de prix réduits
        # Format courant: prix barré X€ prix Y€
        # Détecter les mots suggérant un prix barré: ancien prix, avant, était à, etc.
        barred_price_patterns = [
            r'(?:ancien\s+prix|prix\s+normal|avant|était\s+à|prix\s+habituel)(?:\s*:)?\s*(\d+[,.]\d+|\d+)(\s*€|\s*EUR|\s*euros?)(?:[^€]*?)(\d+[,.]\d+|\d+)(\s*€|\s*EUR|\s*euros?)',
            r'(\d+[,.]\d+|\d+)(\s*€|\s*EUR|\s*euros?)(?:\s*(?:au lieu de|barré|remplacé par))(?:[^€]*?)(\d+[,.]\d+|\d+)(\s*€|\s*EUR|\s*euros?)'
        ]
        
        for pattern in barred_price_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # L'ordre peut varier selon le pattern, déterminer quel prix est l'initial et lequel est le réduit
                    prices = [float(match[0].replace(',', '.')), float(match[2].replace(',', '.'))]
                    currencies = [match[1], match[3]]
                    
                    # Dans le premier pattern, le prix initial est le premier; dans le second, c'est le deuxième
                    if pattern == barred_price_patterns[0]:
                        initial_price, reduced_price = prices
                        initial_currency, reduced_currency = currencies
                    else:
                        reduced_price, initial_price = prices
                        reduced_currency, initial_currency = currencies
                    
                    # Vérifier si le prix réduit est supérieur ou égal au prix initial
                    if reduced_price >= initial_price:
                        error_info = {
                            "type": "prix_barré_incohérent",
                            "prix_initial": initial_price,
                            "prix_réduit": reduced_price,
                            "différence": reduced_price - initial_price,
                            "pourcentage": 100 * (reduced_price - initial_price) / initial_price if initial_price > 0 else 0,
                            "texte_original": match[0] + match[1] + "..." + match[2] + match[3]
                        }
                        price_errors.append(error_info)
                        print(f"⚠️ Prix barré incohérent: le prix réduit {reduced_price}{reduced_currency} est supérieur au prix initial {initial_price}{initial_currency}")
                except ValueError:
                    continue
        
        # Résumé des résultats
        if price_errors:
            print(f"⚠️ {len(price_errors)} incohérences de prix détectées")
        else:
            print("✅ Aucune incohérence de prix détectée")
            
        return price_errors 

    def check_rag_legislation(self, legislation_text, ad_text, products):
        """
        Analyse la législation extraite du RAG et détecte les non-conformités spécifiques enrichies.
        Args:
            legislation_text (str): Texte de la législation extraite (RAG)
            ad_text (str): Texte de la publicité à analyser
            products (list): Liste des produits détectés
        Returns:
            list: Liste de non-conformités détectées
        """
        non_conformities = []
        if not legislation_text or not ad_text:
            return non_conformities

        import re

        # 1. Mentions obligatoires globales
        mentions_obligatoires = re.findall(r"mention obligatoire ?: ([^\n\r]+)", legislation_text, re.IGNORECASE)
        for mention in mentions_obligatoires:
            if mention.lower() not in ad_text.lower():
                non_conformities.append(f"Mention légale obligatoire absente : '{mention}'")

        # 2. Mentions interdites globales
        mentions_interdites = re.findall(r"mention interdite ?: ([^\n\r]+)", legislation_text, re.IGNORECASE)
        for mention in mentions_interdites:
            if mention.lower() in ad_text.lower():
                non_conformities.append(f"Mention légale interdite présente : '{mention}'")

        # 3. Mentions obligatoires conditionnelles par produit
        # Exemple de pattern : "mention obligatoire pour (.+) : (.+)"
        cond_obligatoires = re.findall(r"mention obligatoire pour ([^:]+) ?: ([^\n\r]+)", legislation_text, re.IGNORECASE)
        for condition, mention in cond_obligatoires:
            for prod in products:
                if condition.lower() in prod.lower() and mention.lower() not in ad_text.lower():
                    non_conformities.append(f"Mention obligatoire pour '{condition}' absente alors que le produit '{prod}' est présent : '{mention}'")

        # 4. Mentions interdites conditionnelles par produit
        cond_interdites = re.findall(r"mention interdite pour ([^:]+) ?: ([^\n\r]+)", legislation_text, re.IGNORECASE)
        for condition, mention in cond_interdites:
            for prod in products:
                if condition.lower() in prod.lower() and mention.lower() in ad_text.lower():
                    non_conformities.append(f"Mention interdite pour '{condition}' présente alors que le produit '{prod}' est présent : '{mention}'")

        # 5. Mentions à formuler exactement
        exact_mentions = re.findall(r"mention exacte ?: ([^\n\r]+)", legislation_text, re.IGNORECASE)
        for mention in exact_mentions:
            if mention.lower() not in ad_text.lower():
                non_conformities.append(f"La mention obligatoire doit être formulée exactement ainsi : '{mention}'")

        # 6. (Extension possible : position, taille, couleur, etc. à partir de la vision)
        # À implémenter selon les besoins et les capacités de la vision

        return non_conformities 