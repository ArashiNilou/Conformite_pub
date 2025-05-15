# Légalité Pubs - Analyse de Conformité Publicitaire

Outil d'analyse automatisée de publicités pour vérifier leur conformité légale, utilisable en ligne de commande.

## Description

Cet outil permet de :
- Analyser des publicités (images, PDF) pour vérifier leur conformité légale
- Détecter les erreurs de prix, les mentions légales manquantes
- Vérifier la lisibilité et la qualité des informations
- Générer un rapport détaillé avec les non-conformités
- Exporter les résultats d'analyse

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/ArashiNilou/Conformite_pub.git
cd Conformite_pub

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

```bash
# Analyser une publicité via la ligne de commande
python src/main.py --files "chemin/vers/votre/image.jpg"

# Analyser toutes les publicités d'un dossier
python src/main.py --dir "chemin/vers/dossier"

# Voir toutes les options disponibles
python src/main.py --help
```

Options principales :
- `--files` : Chemin vers le fichier publicitaire à analyser (requis si --dir n'est pas utilisé)
- `--dir` : Chemin vers un répertoire contenant les fichiers à analyser (requis si --files n'est pas utilisé)
- `--test_text_extraction` : Active uniquement le test d'extraction de texte sans analyse complète
- `--extract_raw_text` : Extrait le texte brut des images sans corrections
- `--mode` : Mode d'extraction de texte (docling, gpt4v, azure_cv)
- `--ocr` : Moteur OCR à utiliser avec Docling (tesseract, easyocr, rapidocr, tesseract_api)
- `--method` : Méthode d'extraction de texte brut (tesseract, easyocr, auto, gpt_vision)

### Exemples

Analyser un fichier simple sans espaces dans le nom :
```bash
python src/main.py --files publicite.jpg
```

Analyser un fichier avec des espaces dans le nom (utilisez des guillemets) :
```bash
python src/main.py --files "dossier/Nom avec espaces.jpg"
```

Analyser plusieurs fichiers à la fois :
```bash
python src/main.py --files "publicite1.jpg" "publicite2.png" "publicite3.pdf"
```

Analyser tous les fichiers d'un dossier :
```bash
python src/main.py --dir "dossier_publicites"
```

> **Note importante** : Si le nom du fichier ou le chemin contient des espaces, entourez-le de guillemets pour éviter des erreurs.

## Fonctionnalités principales

- Analyse du contenu visuel et textuel
- Détection des erreurs de prix (notamment les prix barrés incorrects)
- Vérification des mentions légales obligatoires
- Contrôle de la lisibilité des textes
- Génération de rapports détaillés 

## Résultats d'analyse

L'outil analyse 5 éléments clés :
1. Mentions légales
2. Lisibilité/Typographie
3. Logo
4. Astérisques
5. Dates

Pour chaque élément, l'outil fournit :
- Des observations détaillées
- Une évaluation (Conforme, Non conforme, À corriger, etc.)
- Des recommandations pour améliorer la conformité

Les résultats sont sauvegardés dans le dossier `outputs`.

## Technologies utilisées

- Python
- OpenAI GPT-4o
- Traitement d'images
- Extraction de texte
- Analyse de conformité automatisée
