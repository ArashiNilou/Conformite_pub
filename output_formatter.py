import json
import os
from datetime import datetime
from pathlib import Path
from src.utils.logo_product_matcher import LogoProductMatcher

class OutputFormatter:
    """
    Classe pour formater les sorties d'analyse publicitaire en différents formats
    (HTML, PDF, etc.) à partir des fichiers JSON générés
    """
    
    def __init__(self):
        self.output_dir = Path("outputs")
        self.reports_dir = Path("rapports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.logo_product_matcher = LogoProductMatcher()
    
    def format_json_to_html(self, json_path):
        """
        Convertit un fichier JSON d'analyse en rapport HTML synthétique
        
        Args:
            json_path: Chemin vers le fichier JSON d'analyse
        
        Returns:
            str: Chemin vers le fichier HTML généré
        """
        try:
            # Charger le JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extraire le nom de fichier de l'image analysée
            filename = Path(json_path).stem
            if filename.startswith("analyse_"):
                filename = filename[8:]  # Enlever le préfixe "analyse_"
            
            # Séparer le timestamp s'il est présent
            if "_202" in filename:  # Si contient une date au format 20250513...
                parts = filename.split('_')
                timestamp_parts = [p for p in parts if p.startswith('202')]
                if timestamp_parts:
                    # Garder uniquement le nom de l'image sans les timestamps
                    non_timestamp_parts = [p for p in parts if not p.startswith('202')]
                    filename = '_'.join(non_timestamp_parts)
            
            # Créer le rapport HTML
            html_content = self._generate_html_template(data, filename)
            
            # Sauvegarder dans le dossier rapports
            output_path = self.reports_dir / f"rapport_{filename}.html"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"✅ Rapport HTML généré : {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Erreur lors de la génération du rapport HTML : {str(e)}")
            return None
    
    def format_all_json_to_html(self):
        """
        Convertit tous les fichiers JSON d'analyse en rapports HTML
        
        Returns:
            list: Liste des chemins vers les fichiers HTML générés
        """
        generated_files = []
        json_files = list(self.output_dir.glob("analyse_*.json"))
        
        print(f"🔄 Conversion de {len(json_files)} fichiers JSON en rapports HTML...")
        
        for json_file in json_files:
            html_path = self.format_json_to_html(json_file)
            if html_path:
                generated_files.append(html_path)
        
        print(f"✅ {len(generated_files)} rapports HTML générés dans {self.reports_dir}")
        return generated_files
    
    def _generate_html_template(self, data, filename):
        """
        Génère le contenu HTML à partir des données JSON
        
        Args:
            data: Données d'analyse au format JSON
            filename: Nom du fichier pour le titre
            
        Returns:
            str: Contenu HTML
        """
        # Extraire les informations pertinentes du JSON
        raw_text = data.get("raw_text", "")
        vision_result = data.get("vision_analysis", "")
        consistency = data.get("consistency_check", "")
        product_logo = data.get("product_logo_consistency", "")
        legislation = data.get("legislation", "")
        compliance = data.get("compliance_analysis", "")
        
        # Analyser le texte pour détecter les problèmes
        problemes = []
        messages_non_conformite = ["non conforme", "non-conforme", "erreur critique"]
        if any(message.lower() in compliance.lower() for message in messages_non_conformite):
            problemes.append("Non-conformité légale détectée")
        
        if "basse résolution" in raw_text.lower() or "illisible" in raw_text.lower():
            problemes.append("Image partiellement illisible ou en basse résolution")
        
        # Vérifier si les produits sont non transformés avant d'ajouter l'absence de mangerbouger.fr aux problèmes
        if "mangerbouger.fr" in legislation.lower() and "mangerbouger.fr" not in vision_result.lower():
            products = self.logo_product_matcher.extract_products_from_text(vision_result)
            if not self.logo_product_matcher.is_non_transformed_product(products):
                problemes.append("Absence de mention www.mangerbouger.fr")
        
        if "astérisque" in compliance.lower() and "sans renvoi" in compliance.lower():
            problemes.append("Astérisques sans renvoi")
            
        # Déterminer l'évaluation globale
        evaluation_globale = "CONFORME"
        if problemes:
            if any("Non-conformité" in p for p in problemes):
                evaluation_globale = "NON CONFORME"
            else:
                evaluation_globale = "PARTIELLEMENT CONFORME"
        
        # Préparer les observations
        observations = self._extraire_observations(data)
        
        # Générer le HTML
        date_generation = datetime.now().strftime("%d/%m/%Y")
        
        html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport d'Analyse - {filename}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background-color: #f2f2f2;
            padding: 12px;
            text-align: left;
            border: 1px solid #ddd;
            font-weight: bold;
        }}
        td {{
            padding: 12px;
            border: 1px solid #ddd;
            vertical-align: top;
        }}
        .evaluation {{
            font-weight: bold;
            padding: 5px;
            border-radius: 4px;
            display: inline-block;
        }}
        .pertinent {{
            color: #27ae60;
            background-color: #eafaf1;
        }}
        .ameliorer {{
            color: #f39c12;
            background-color: #fef9e7;
        }}
        .probleme {{
            color: #e74c3c;
            background-color: #fdedec;
        }}
        .check-icon {{
            color: #27ae60;
            margin-right: 5px;
        }}
        .warning-icon {{
            color: #f39c12;
            margin-right: 5px;
        }}
        .error-icon {{
            color: #e74c3c;
            margin-right: 5px;
        }}
        .summary {{
            background-color: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 20px 0;
        }}
        .verdict {{
            font-size: 1.2em;
            text-align: center;
            padding: 15px;
            margin: 20px 0;
            font-weight: bold;
            border-radius: 5px;
        }}
        .conforme {{
            background-color: #d5f5e3;
            color: #27ae60;
        }}
        .partiellement-conforme {{
            background-color: #fef9e7;
            color: #f39c12;
        }}
        .non-conforme {{
            background-color: #fdedec;
            color: #e74c3c;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 10px;
            border-top: 1px solid #eee;
            text-align: center;
            font-size: 0.9em;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Rapport d'Analyse Publicitaire - {filename}</h1>
        
        <div class="verdict {evaluation_globale.lower().replace(' ', '-')}">
            VERDICT : {evaluation_globale}
        </div>
        
        <div class="summary">
            <h3>Synthèse de l'analyse</h3>
            <p>{self._generer_synthese(data, problemes)}</p>
        </div>

        <table>
            <tr>
                <th width="20%">Aspect analysé</th>
                <th width="60%">Observations</th>
                <th width="20%">Évaluation</th>
            </tr>
"""
        
        # Ajouter chaque aspect
        for aspect, details in observations.items():
            html += f"""
            <tr>
                <td>{aspect}</td>
                <td>{details['observation']}</td>
                <td><span class="evaluation {details['classe']}"><span class="{details['icon_class']}">{details['icon']}</span> {details['evaluation']}</span></td>
            </tr>"""
        
        # Fermer la table et ajouter les recommandations
        html += f"""
        </table>

        <div class="summary">
            <h3>Recommandations</h3>
            <ol>
                {self._generer_recommandations(data, problemes)}
            </ol>
        </div>

        <div class="footer">
            <p>Rapport généré le {date_generation} | Système d'analyse publicitaire Légalité Pubs</p>
        </div>
    </div>
</body>
</html>"""
        
        return html
    
    def _extraire_observations(self, data):
        """
        Extrait les observations structurées à partir des données d'analyse
        
        Args:
            data: Données d'analyse au format JSON
            
        Returns:
            dict: Observations structurées par aspect
        """
        vision_result = data.get("vision_analysis", "")
        compliance = data.get("compliance_analysis", "")
        raw_text = data.get("raw_text", "")
        
        # Initialiser les observations avec des valeurs par défaut
        observations = {
            "Contenu / Message": {
                "observation": self._extraire_contenu_message(vision_result),
                "evaluation": "Pertinent",
                "classe": "pertinent",
                "icon": "✓",
                "icon_class": "check-icon"
            },
            "Produits": {
                "observation": self._extraire_produits(vision_result),
                "evaluation": "Bien ciblé",
                "classe": "pertinent",
                "icon": "✓",
                "icon_class": "check-icon"
            },
            "Design graphique": {
                "observation": self._extraire_design(vision_result),
                "evaluation": "Efficace",
                "classe": "pertinent",
                "icon": "✓",
                "icon_class": "check-icon"
            },
            "Lisibilité / Typographie": {
                "observation": self._extraire_lisibilite(raw_text, vision_result),
                "evaluation": "Bonne",
                "classe": "pertinent",
                "icon": "✓",
                "icon_class": "check-icon"
            },
            "Mentions légales": {
                "observation": self._extraire_mentions_legales(vision_result, compliance),
                "evaluation": "Conformes",
                "classe": "pertinent",
                "icon": "✓",
                "icon_class": "check-icon"
            },
            "Informations pratiques": {
                "observation": self._extraire_infos_pratiques(vision_result),
                "evaluation": "Complètes",
                "classe": "pertinent",
                "icon": "✓",
                "icon_class": "check-icon"
            }
        }
        
        # Actualiser les évaluations en fonction des problèmes détectés
        if "basse résolution" in raw_text.lower() or "illisible" in raw_text.lower():
            observations["Lisibilité / Typographie"]["evaluation"] = "À améliorer"
            observations["Lisibilité / Typographie"]["classe"] = "ameliorer"
            observations["Lisibilité / Typographie"]["icon"] = "⚠"
            observations["Lisibilité / Typographie"]["icon_class"] = "warning-icon"
        
        if "non conforme" in compliance.lower():
            observations["Mentions légales"]["evaluation"] = "Non conformes"
            observations["Mentions légales"]["classe"] = "probleme"
            observations["Mentions légales"]["icon"] = "✗"
            observations["Mentions légales"]["icon_class"] = "error-icon"
            
        if "coordonnées" in compliance.lower() and ("incomplètes" in compliance.lower() or "manquantes" in compliance.lower()):
            observations["Informations pratiques"]["evaluation"] = "Incomplètes"
            observations["Informations pratiques"]["classe"] = "ameliorer"
            observations["Informations pratiques"]["icon"] = "⚠"
            observations["Informations pratiques"]["icon_class"] = "warning-icon"
        
        return observations
    
    def _extraire_contenu_message(self, vision_result):
        """Extrait l'observation sur le contenu et le message"""
        # Extraire des phrases pertinentes de l'analyse
        contenu = "Message publicitaire clair et bien structuré."
        
        # Si des mots-clés spécifiques sont présents dans la vision_result
        if "authenticité" in vision_result.lower():
            contenu = "Message d'authenticité mis en avant."
        if "tradition" in vision_result.lower():
            contenu += " Valeurs de tradition bien communiquées."
        if "naturel" in vision_result.lower() or "nature" in vision_result.lower():
            contenu += " Accent sur le caractère naturel du produit."
            
        # Vérifier si des problèmes orthographiques sont mentionnés
        if "orthographe" in vision_result.lower() and "faute" in vision_result.lower():
            contenu += " Quelques fautes d'orthographe relevées."
            
        return contenu
    
    def _extraire_produits(self, vision_result):
        """Extrait l'observation sur les produits"""
        produits = "Produits clairement identifiés."
        
        # Extraire des descriptions spécifiques de produits
        if "yaourt" in vision_result.lower():
            produits = "Yaourts et produits laitiers clairement identifiés."
        if "viande" in vision_result.lower():
            produits = "Produits de boucherie clairement identifiés."
        if "fruit" in vision_result.lower() and "légume" in vision_result.lower():
            produits = "Fruits et légumes clairement identifiés."
            
        return produits
    
    def _extraire_design(self, vision_result):
        """Extrait l'observation sur le design graphique"""
        design = "Design graphique efficace, avec une bonne hiérarchie visuelle."
        
        # Mentions spécifiques au design
        if "couleur" in vision_result.lower():
            if "efficace" in vision_result.lower():
                design = "Utilisation efficace des couleurs, bonne hiérarchie visuelle."
            elif "agressive" in vision_result.lower():
                design = "Couleurs vives, parfois agressives. Hiérarchie visuelle correcte."
                
        # Mentions des logos
        if "logo" in vision_result.lower():
            if "bien intégré" in vision_result.lower():
                design += " Logos bien intégrés dans le design global."
            elif "trop petit" in vision_result.lower():
                design += " Logos présents mais trop petits."
                
        return design
    
    def _extraire_lisibilite(self, raw_text, vision_result):
        """Extrait l'observation sur la lisibilité/typographie"""
        lisibilite = "Texte bien lisible, police de taille adaptée."
        
        # Problèmes de lisibilité mentionnés
        if "basse résolution" in raw_text.lower() or "illisible" in raw_text.lower():
            lisibilite = "Certaines parties du texte sont en basse résolution ou difficilement lisibles."
            
        if "trop petit" in vision_result.lower() and "mention" in vision_result.lower():
            lisibilite += " Mentions légales en caractères trop petits."
            
        if "police" in vision_result.lower() and "adaptée" in vision_result.lower():
            lisibilite += " Police globalement adaptée pour le texte principal."
            
        return lisibilite
    
    def _extraire_mentions_legales(self, vision_result, compliance):
        """Extrait l'observation sur les mentions légales"""
        mentions = "Mentions légales présentes et conformes."
        
        if "www.mangerbouger.fr" in compliance.lower():
            if "absent" in compliance.lower() or "manquant" in compliance.lower():
                mentions = "Absence de la mention obligatoire www.mangerbouger.fr."
            
        if "astérisque" in compliance.lower() and "sans renvoi" in compliance.lower():
            mentions += " Astérisques présents sans renvoi correspondant."
            
        if "RCS" in compliance.lower():
            if "absent" in compliance.lower() or "manquant" in compliance.lower():
                mentions += " Numéro RCS manquant."
                
        return mentions
    
    def _extraire_infos_pratiques(self, vision_result):
        """Extrait l'observation sur les informations pratiques"""
        infos = "Coordonnées et informations pratiques complètes."
        
        # Analyse des coordonnées
        coordonnees = []
        if "site internet" in vision_result.lower() or "www" in vision_result.lower():
            coordonnees.append("site internet")
        if "adresse" in vision_result.lower():
            if "absent" in vision_result.lower() or "manquant" in vision_result.lower():
                infos = "Adresse physique manquante."
            else:
                coordonnees.append("adresse")
        if "téléphone" in vision_result.lower():
            if "absent" in vision_result.lower() or "manquant" in vision_result.lower():
                infos += " Numéro de téléphone manquant."
            else:
                coordonnees.append("téléphone")
                
        # Si des coordonnées sont présentes
        if coordonnees:
            infos = f"Coordonnées présentes : {', '.join(coordonnees)}."
            
        # Informations sur les dates
        if "date" in vision_result.lower():
            if "visible" in vision_result.lower():
                infos += " Dates bien visibles."
            elif "manquant" in vision_result.lower():
                infos += " Dates manquantes ou peu visibles."
                
        return infos
    
    def _generer_synthese(self, data, problemes):
        """
        Génère un texte de synthèse pour le rapport
        
        Args:
            data: Données d'analyse
            problemes: Liste des problèmes détectés
            
        Returns:
            str: Texte de synthèse
        """
        vision_result = data.get("vision_analysis", "")
        
        # Déterminer le type de produit
        type_produit = "produit"
        if "yaourt" in vision_result.lower():
            type_produit = "yaourts"
        elif "viande" in vision_result.lower() or "boucherie" in vision_result.lower():
            type_produit = "produits de boucherie"
        elif "fruit" in vision_result.lower() and "légume" in vision_result.lower():
            type_produit = "fruits et légumes"
            
        # Message de base
        if not problemes:
            return f"La publicité pour les {type_produit} est conforme aux exigences légales. Le message marketing est clair et les informations requises sont présentes."
        else:
            synthese = f"La publicité pour les {type_produit} présente "
            
            if len(problemes) == 1:
                synthese += f"un problème: {problemes[0].lower()}. "
            else:
                synthese += f"plusieurs problèmes: {', '.join([p.lower() for p in problemes[:-1]])} et {problemes[-1].lower()}. "
                
            synthese += "Des corrections sont nécessaires pour assurer la conformité légale."
                
            return synthese
    
    def _generer_recommandations(self, data, problemes):
        """
        Génère la liste des recommandations HTML
        
        Args:
            data: Données d'analyse
            problemes: Liste des problèmes détectés
            
        Returns:
            str: HTML des recommandations
        """
        recommandations_html = ""
        vision_result = data.get("vision_analysis", "")
        
        # Générer des recommandations en fonction des problèmes
        for probleme in problemes:
            if "mangerbouger.fr" in probleme.lower():
                # Vérifier si les produits détectés sont non transformés
                # Dans ce cas, ne pas ajouter la recommandation
                products = self.logo_product_matcher.extract_products_from_text(vision_result)
                if not self.logo_product_matcher.is_non_transformed_product(products):
                    recommandations_html += "<li><strong>Légal (URGENT) :</strong> Ajouter la mention www.mangerbouger.fr qui doit occuper au moins 7% de la surface publicitaire.</li>\n"
            elif "résolution" in probleme.lower() or "illisible" in probleme.lower():
                recommandations_html += "<li><strong>Qualité :</strong> Améliorer la résolution de l'image, particulièrement pour les mentions légales.</li>\n"
            elif "astérisque" in probleme.lower():
                recommandations_html += "<li><strong>Légal :</strong> Ajouter les renvois pour chaque astérisque utilisé dans la publicité.</li>\n"
            elif "Non-conformité" in probleme:
                recommandations_html += "<li><strong>Conformité :</strong> Corriger les non-conformités légales mentionnées dans l'analyse.</li>\n"
                
        # Recommandations standard si aucun problème spécifique n'est détecté
        if not recommandations_html:
            recommandations_html = """
                <li><strong>Optimisation :</strong> Bien que conforme, l'efficacité publicitaire pourrait être améliorée avec un appel à l'action plus direct.</li>
                <li><strong>Visibilité :</strong> Envisager d'augmenter légèrement la taille des mentions légales tout en conservant leur lisibilité.</li>
            """
            
        return recommandations_html

# Fonction pour utilisation en ligne de commande
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Générer des rapports HTML à partir des analyses JSON')
    parser.add_argument('--all', action='store_true', help='Convertir tous les fichiers JSON')
    parser.add_argument('--file', help='Chemin vers un fichier JSON spécifique')
    args = parser.parse_args()
    
    formatter = OutputFormatter()
    
    if args.all:
        formatter.format_all_json_to_html()
    elif args.file:
        formatter.format_json_to_html(args.file)
    else:
        print("Erreur: spécifiez soit --all pour tous les fichiers, soit --file pour un fichier spécifique")

if __name__ == "__main__":
    main() 