from typing import Dict, List, Tuple, Optional, Set
import re

class LogoProductMatcher:
    """
    Classe pour vérifier la cohérence entre les logos et les produits mentionnés
    """
    
    def __init__(self):
        """Initialise la base de données des correspondances logo-produit"""
        self.logo_product_mapping = {
            # Format: "nom du logo": ["catégories de produits autorisées"]
            "le porc français": ["porc", "cochon", "charcuterie"],
            "porc français": ["porc", "cochon", "charcuterie"],  # Variante sans "le"
            "le bœuf français": ["bœuf", "bovin", "boeuf", "taureau", "vache"],
            "bœuf français": ["bœuf", "bovin", "boeuf", "taureau", "vache"],  # Variante sans "le"
            "le veau français": ["veau"],
            "veau français": ["veau"],  # Variante sans "le"
            "l'agneau français": ["agneau", "mouton", "brebis"],
            "agneau français": ["agneau", "mouton", "brebis"],  # Variante sans "l'"
            "viandes de france": ["viande"],
            "volaille française": ["volaille", "poulet", "dinde", "canard", "oie"],
            "pêché en loire atlantique": ["poisson", "fruit de mer", "crustacé"],
            "pêche française": ["poisson", "fruit de mer", "crustacé"],
            "label rouge": ["tous"],  # Label Rouge peut être utilisé pour de nombreux produits
            "aoc": ["tous"],          # AOC/AOP peut être utilisé pour de nombreux produits
            "aop": ["tous"],
            "bio": ["tous"],
            "agriculture biologique": ["tous"],
            "igp": ["tous"],          # IGP peut être utilisé pour de nombreux produits
        }
        
        # Mots clés pour détecter les produits
        self.product_keywords = {
            "porc": ["porc", "cochon", "porcin", "filet mignon", "jambon", "échine", "saucisse", "côte de porc", "lard", "travers"],
            "bœuf": ["bœuf", "bovin", "boeuf", "taureau", "vache", "entrecôte", "rumsteck", "bavette", "hampe", "steak", "langue bovine", "viande bovine", "langue"],
            "veau": ["veau"],
            "agneau": ["agneau", "mouton", "brebis", "gigot", "côtelette d'agneau"],
            "volaille": ["volaille", "poulet", "dinde", "canard", "oie", "pintade", "chapon", "poule", "fermier de janzé"],
            "poisson": ["poisson", "saumon", "truite", "cabillaud", "thon", "lotte", "sole", "dorade", "bar", "merlu"],
            "crustacé": ["fruit de mer", "huître", "moule", "crevette", "crustacé", "langoustine", "homard"],
            "produit laitier": ["fromage", "lait", "yaourt", "beurre", "crème", "emmental", "camembert", "brie"],
            "boulangerie": ["pain", "baguette", "croissant", "brioche", "pâtisserie", "gâteau", "tarte"],
            "alcool": ["vin", "bière", "spiritueux", "alcool", "whisky", "champagne", "rhum"],
            "viande": ["viande", "charcuterie"]
        }
        
        # Liste des catégories de produits connus
        self.product_categories = []
        for cat_keywords in self.product_keywords.values():
            self.product_categories.extend(cat_keywords)
        
        # Mapping des produits spécifiques à leur catégorie
        self.specific_product_mapping = {
            "langue": "bœuf",
            "langue bovine": "bœuf",
            "filet mignon": "porc",
            "poulet fermier": "volaille",
            "fermier de janzé": "volaille"
        }
        
        # Liste des catégories de produits non transformés exemptés de mentions légales comme mangerbouger.fr
        self.non_transformed_products = {
            "viandes fraîches": ["viande fraîche", "viande crue", "viande non transformée"],
            "poissons frais": ["poisson frais", "poisson cru", "poisson non transformé", "noix de saint-jacques", "coquille saint-jacques"],
            "fruits frais": ["fruit", "fruit frais", "pomme", "poire", "banane", "orange", "citron", "fraise", "framboise", "raisin", "melon", "pastèque", "ananas"],
            "légumes frais": ["légume", "légume frais", "carotte", "pomme de terre", "salade", "tomate", "concombre", "poivron", "oignon", "ail"]
        }
    
    def is_logo_compatible_with_product(self, logo: str, product: str) -> bool:
        """
        Vérifie si un logo est compatible avec un produit
        
        Args:
            logo: Nom du logo ou label
            product: Nom du produit
            
        Returns:
            bool: True si compatible, False sinon
        """
        # Normaliser les entrées
        logo_norm = logo.lower().strip()
        product_norm = product.lower().strip()
        
        # Si le logo n'est pas dans notre base, on ne peut pas vérifier
        if logo_norm not in self.logo_product_mapping:
            return True  # Par défaut, on considère que c'est compatible
        
        allowed_categories = self.logo_product_mapping[logo_norm]
        
        # Si "tous" est dans les catégories autorisées, le logo est compatible avec tous les produits
        if "tous" in allowed_categories:
            return True
        
        # Vérifier d'abord si le produit est dans le mapping spécifique
        if product_norm in self.specific_product_mapping:
            category = self.specific_product_mapping[product_norm]
            # Vérifier si cette catégorie est compatible avec le logo
            return any(cat in allowed_categories for cat in [category, product_norm])
        
        # Vérifier si le produit est dans une catégorie autorisée
        for category in allowed_categories:
            if category in product_norm or product_norm in category:
                return True
        
        return False
    
    def check_product_logo_consistency(self, logos: List[str], products: List[str]) -> List[Dict]:
        """
        Vérifie la cohérence entre les logos et produits détectés
        
        Args:
            logos: Liste des logos identifiés
            products: Liste des produits mentionnés
            
        Returns:
            List[Dict]: Liste des incohérences détectées avec explication
        """
        inconsistencies = []
        
        # Associer les produits spécifiques à leurs catégories générales
        extended_products = products.copy()
        product_categories = {}  # Dictionnaire pour stocker le mapping produit -> catégorie
        
        for product in products:
            product_norm = product.lower().strip()
            if product_norm in self.specific_product_mapping:
                # Ajouter sa catégorie générale
                extended_products.append(self.specific_product_mapping[product_norm])
                product_categories[product_norm] = self.specific_product_mapping[product_norm]
            
            # Détection explicite des produits bovins
            if "bœuf" in product_norm or "boeuf" in product_norm or "bovin" in product_norm:
                product_categories[product_norm] = "bœuf"
            
            # Détection explicite des produits de volaille
            if "poulet" in product_norm or "volaille" in product_norm or "dinde" in product_norm:
                product_categories[product_norm] = "volaille"
            
            # Détection explicite des produits porcins
            if "porc" in product_norm or "cochon" in product_norm:
                product_categories[product_norm] = "porc"
                
            # Cas spécifique: "langue de boeuf"
            if "langue" in product_norm and ("bœuf" in product_norm or "boeuf" in product_norm or "bovin" in product_norm):
                product_categories[product_norm] = "bœuf"
            elif "langue" in product_norm and "porc" not in product_norm and "cochon" not in product_norm:
                # Par défaut, si "langue" est mentionnée sans précision, on la considère comme bovine
                product_categories[product_norm] = "bœuf"
        
        # Maintenant, vérifier les incohérences entre chaque logo et tous les produits
        for logo in logos:
            logo_norm = logo.lower().strip()
            if logo_norm not in self.logo_product_mapping:
                continue  # Ignorer les logos inconnus
                
            # Déterminer la catégorie du logo
            logo_category = None
            if "porc" in logo_norm:
                logo_category = "porc"
            elif "bœuf" in logo_norm or "boeuf" in logo_norm:
                logo_category = "bœuf"
            elif "volaille" in logo_norm:
                logo_category = "volaille"
            
            incompatible_products = []
            for product in extended_products:
                product_norm = product.lower().strip()
                
                # Cas spécial 1: Langue de boeuf avec logo porc
                if ("langue" in product_norm and (
                    "bœuf" in product_norm or "boeuf" in product_norm or "bovin" in product_norm)) and "porc" in logo_norm:
                    incompatible_products.append(product)
                    continue
                
                # Cas spécial 2: Langue (par défaut considérée comme bovine) avec logo porc
                if "langue" in product_norm and "porc" not in product_norm and "porc" in logo_norm:
                    incompatible_products.append(product)
                    continue
                
                # Cas spécial 3: Si le produit a une catégorie identifiée qui ne correspond pas au logo
                if product_norm in product_categories:
                    product_cat = product_categories[product_norm]
                    if logo_category and product_cat != logo_category and logo_category != "tous":
                        # Vérification spécifique pour logo Porc avec produit bœuf
                        if "porc" in logo_norm and product_cat == "bœuf":
                            incompatible_products.append(product)
                            continue
                        # Vérification spécifique pour logo Bœuf avec produit porc
                        if ("bœuf" in logo_norm or "boeuf" in logo_norm) and product_cat == "porc":
                            incompatible_products.append(product)
                            continue
                
                # Vérification générale de compatibilité
                if not self.is_logo_compatible_with_product(logo, product):
                    incompatible_products.append(product)
            
            # Éliminer les doublons
            incompatible_products = list(set(incompatible_products))
            
            if incompatible_products:
                inconsistencies.append({
                    "logo": logo,
                    "products": incompatible_products,
                    "allowed_categories": self.logo_product_mapping.get(logo_norm, []),
                    "explanation": f"Le logo '{logo}' n'est pas compatible avec les produits suivants: {', '.join(incompatible_products)}"
                })
        
        return inconsistencies
    
    def extract_products_from_text(self, text: str) -> List[str]:
        """
        Extrait les catégories de produits mentionnées dans un texte avec une analyse plus approfondie
        
        Args:
            text: Texte à analyser
            
        Returns:
            List[str]: Liste des produits identifiés
        """
        found_products = set()
        text_lower = text.lower()
        
        # Recherche de base par mot-clé simple
        for category in self.product_categories:
            if category in text_lower:
                found_products.add(category)
        
        # Recherche spécifique pour les produits particuliers
        # Langue bovine - explicitement catégorisée comme bœuf
        if re.search(r'langue\s+bovine', text_lower) or re.search(r'langue\s+de\s+b[oœe]uf', text_lower) or re.search(r'viande\s+bovine\s+langue', text_lower):
            found_products.add("langue bovine")
            found_products.add("langue")
        
        # Viande bovine - explicitement catégorisée comme bœuf
        if re.search(r'viande\s+bovine', text_lower):
            found_products.add("viande bovine")
        
        # Filet mignon est spécifiquement du porc
        if re.search(r'filet\s+mignon', text_lower):
            found_products.add("filet mignon")
        
        # Poulet fermier (notamment de Janzé)
        if re.search(r'poulet\s+fermier', text_lower) or re.search(r'poulet\s+de\s+janzé', text_lower) or re.search(r'fermier\s+de\s+janzé', text_lower):
            found_products.add("poulet fermier")
            found_products.add("fermier de janzé")
        
        # Recherche dans chaque ligne pour détecter des produits spécifiques
        lines = text.lower().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Détection de la "langue bovine"
            if ("langue" in line and ("bovine" in line or "bœuf" in line or "boeuf" in line)):
                found_products.add("langue bovine")
                found_products.add("langue")
            
            # Détection de "viande bovine"
            if "viande bovine" in line:
                found_products.add("viande bovine")
            
            # Si une ligne contient juste "langue", sans contexte, la considérer comme produit bovin par défaut
            if "langue" in line and not any(word in line for word in ["chat", "étrangère", "parler", "discuter"]):
                found_products.add("langue")
            
            # Détection du "filet mignon"
            if "filet mignon" in line:
                found_products.add("filet mignon")
            
            # Détection du poulet fermier de Janzé
            if "poulet fermier" in line or "fermier de janzé" in line:
                found_products.add("poulet fermier")
                found_products.add("fermier de janzé")
        
        return list(found_products)
    
    def extract_logos_from_text(self, text: str) -> List[str]:
        """
        Extrait les logos et labels mentionnés dans un texte avec une détection améliorée
        
        Args:
            text: Texte à analyser
            
        Returns:
            List[str]: Liste des logos identifiés
        """
        found_logos = set()
        text_lower = text.lower()
        
        # Recherche de base par mot-clé
        for logo in self.logo_product_mapping.keys():
            if logo in text_lower:
                found_logos.add(logo)
        
        # Recherche par expressions régulières pour les variantes courantes
        patterns = [
            (r'le\s+porc\s+français', "le porc français"),
            (r'porc\s+français', "porc français"),
            (r'le\s+b[oœe]uf\s+français', "le bœuf français"),
            (r'b[oœe]uf\s+français', "bœuf français"),
            (r'le\s+veau\s+français', "le veau français"),
            (r'veau\s+français', "veau français"),
            (r'l[\'\"]agneau\s+français', "l'agneau français"),
            (r'agneau\s+français', "agneau français"),
            (r'volaille\s+française', "volaille française"),
            (r'label\s+rouge', "label rouge")
        ]
        
        for pattern, logo_name in patterns:
            if re.search(pattern, text_lower):
                found_logos.add(logo_name)

        # Analyse du contexte par ligne pour détecter les logos dans leur contexte
        lines = text.lower().split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Détection par mots-clés
            if "porc" in line and "français" in line:
                found_logos.add("le porc français")
            
            if any(beef in line for beef in ["bœuf", "boeuf", "bovin"]) and "français" in line:
                found_logos.add("le bœuf français")
            
            if "volaille" in line and "français" in line:
                found_logos.add("volaille française")
            
            if "label rouge" in line:
                found_logos.add("label rouge")
            
            # Traitement des lignes adjacentes pour associer logos et produits
            if i > 0 and i < len(lines) - 1:
                prev_line = lines[i-1].strip()
                next_line = lines[i+1].strip()
                
                # Détection de logos dans les lignes adjacentes aux produits
                if "origine france" in line.lower() or "origine" in line.lower():
                    # Chercher le produit dans la même ligne ou les lignes adjacentes
                    context_lines = [prev_line, line, next_line]
                    context_text = " ".join(context_lines).lower()
                    
                    if "porc" in context_text or "filet mignon" in context_text:
                        found_logos.add("le porc français")
                    
                    if "bœuf" in context_text or "bovin" in context_text or "langue" in context_text or "viande bovine" in context_text:
                        found_logos.add("le bœuf français")
                    
                    if "poulet" in context_text or "volaille" in context_text or "fermier de janzé" in context_text:
                        found_logos.add("volaille française")
        
        return list(found_logos)
    
    def is_non_transformed_product(self, products: List[str]) -> bool:
        """
        Détermine si les produits listés sont des produits non transformés exemptés de la mention mangerbouger.fr
        (viande fraîche, poisson frais, fruits et légumes frais, etc.)
        Args:
            products: Liste des produits à vérifier
        Returns:
            bool: True si tous les produits sont non transformés, False sinon
        """
        import re
        # Si aucun produit n'est fourni, fallback : regarder dans le texte complet
        if not products:
            return False
        text = " ".join(products).lower()
        # Marqueurs explicites de produits transformés
        transformed_markers = [
            "transformé", "préparé", "cuisiné", "appertisé", "conserve", "plat préparé",
            "sauté", "mijoté", "surgelé", "cuit", "pané", "mariné"
        ]
        if any(marker in text for marker in transformed_markers):
            print("[LOGO_PRODUCT_MATCHER] Produit transformé détecté via marqueur.")
            return False
        # Synonymes et variantes pour produits non transformés
        non_transformed_patterns = [
            r"ananas(\s+frais)?",
            r"noix\s+de\s+saint[ -]?jacques",
            r"coquille(s)?\s+saint[ -]?jacques",
            r"poisson(s)?\s+frais",
            r"poisson(s)?\s+cru(s)?",
            r"fruit(s)?\s+frais",
            r"légume(s)?\s+frais",
            r"viande(s)?\s+fraîche(s)?",
            r"viande(s)?\s+crue(s)?",
            r"viande(s)?\s+non\s+transformée(s)?",
            r"pomme", r"poire", r"banane", r"orange", r"citron", r"fraise", r"framboise", r"raisin", r"melon", r"pastèque",
            r"carotte", r"pomme\s+de\s+terre", r"salade", r"tomate", r"concombre", r"poivron", r"oignon", r"ail"
        ]
        for pattern in non_transformed_patterns:
            if re.search(pattern, text):
                print(f"[LOGO_PRODUCT_MATCHER] Produit non transformé détecté via pattern : {pattern}")
                return True
        # Vérifications spécifiques par type de produit (comme avant)
        if "viande" in text and not any(processed in text for processed in ["charcuterie", "jambon", "saucisse", "saucisson", "pâté"]):
            print("[LOGO_PRODUCT_MATCHER] Viande fraîche détectée.")
            return True
        if "poisson" in text and not any(processed in text for processed in ["fumé", "pané", "conserve"]):
            print("[LOGO_PRODUCT_MATCHER] Poisson frais détecté.")
            return True
        meat_keywords = ["bœuf", "bovin", "boeuf", "veau", "agneau", "volaille", "poulet", "dinde", "canard"]
        processed_keywords = ["préparé", "transformé", "cuisiné", "charcuterie"]
        if any(meat in text for meat in meat_keywords) and not any(processed in text for processed in processed_keywords):
            print("[LOGO_PRODUCT_MATCHER] Produit de boucherie non transformé détecté.")
            return True
        print("[LOGO_PRODUCT_MATCHER] Aucun produit non transformé détecté.")
        return False