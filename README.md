# Légalité Pubs - Analyse de Conformité Publicitaire

Application d'analyse automatisée de publicités pour vérifier leur conformité légale.

## Description

Cette application permet de :
- Analyser des publicités (images, PDF) pour vérifier leur conformité légale
- Détecter les erreurs de prix, les mentions légales manquantes
- Vérifier la lisibilité et la qualité des informations
- Générer un rapport détaillé avec les non-conformités

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
# Lancer l'application Streamlit
streamlit run streamlit_app.py
```

Dans l'interface, vous pouvez :
1. Télécharger une image ou un PDF publicitaire
2. Lancer l'analyse
3. Consulter les résultats dans un tableau avec des indicateurs visuels
4. Télécharger le rapport complet au format HTML

## Fonctionnalités principales

- Analyse du contenu visuel et textuel
- Détection des erreurs de prix (notamment les prix barrés incorrects)
- Vérification des mentions légales obligatoires
- Contrôle de la lisibilité des textes
- Génération de rapports détaillés

## Technologies utilisées

- Python
- Streamlit
- OpenAI GPT-4o
- Traitement d'images
- Extraction de texte
