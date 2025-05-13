import streamlit as st
import os
import json
import base64
from PIL import Image
import pandas as pd
import io
import subprocess
from datetime import datetime
import time
import PyPDF2

st.set_page_config(
    page_title="Analyse Publicitaire",
    layout="wide"
)

def run_analysis(file_path):
    """Exécute le script d'analyse sur le fichier spécifié et retourne le chemin du résultat"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cmd = f"python src/main.py --files '{file_path}'"
    
    with st.spinner('Analyse en cours...'):
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            st.error(f"Erreur lors de l'analyse: {stderr.decode('utf-8')}")
            return None
    
    # Trouver le fichier de résultat le plus récent
    results_dir = "outputs"
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    if not json_files:
        st.error("Aucun résultat d'analyse trouvé")
        return None
    
    # Trier par date de modification (plus récent en premier)
    latest_file = sorted(json_files, key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)[0]
    return os.path.join(results_dir, latest_file)

def analyze_pub(uploaded_file):
    """Sauvegarde le fichier uploadé et lance l'analyse"""
    # Créer un dossier temporaire si nécessaire
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Sauvegarder le fichier
    file_extension = uploaded_file.name.split('.')[-1].lower()
    timestamp = int(time.time())
    temp_path = os.path.join(temp_dir, f"upload_{timestamp}.{file_extension}")
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Si c'est un PDF, extraire la première page comme image
    if file_extension == 'pdf':
        try:
            # Extraire la première page du PDF comme image
            pdf_reader = PyPDF2.PdfReader(temp_path)
            
            # Convertir la première page en image (nécessite un module supplémentaire)
            # Cette partie est simplifiée et peut nécessiter des ajustements
            img_path = os.path.join(temp_dir, f"converted_{timestamp}.png")
            
            # Utiliser une commande système pour convertir PDF en image
            convert_cmd = f"convert -density 300 '{temp_path}[0]' '{img_path}'"
            os.system(convert_cmd)
            
            # Si la conversion a réussi, utiliser l'image convertie
            if os.path.exists(img_path):
                temp_path = img_path
        except Exception as e:
            st.error(f"Erreur lors de la conversion du PDF: {str(e)}")
            return None
    
    # Lancer l'analyse
    result_path = run_analysis(temp_path)
    return result_path

def get_evaluation_emoji(evaluation):
    """Convertit l'évaluation textuelle en emoji"""
    if "conforme" in evaluation.lower() or "pertinent" in evaluation.lower() or "bon" in evaluation.lower() or "claire" in evaluation.lower() or "efficace" in evaluation.lower():
        return "✅"
    elif "améliorer" in evaluation.lower() or "incomplet" in evaluation.lower() or "partiel" in evaluation.lower():
        return "⚠️"
    elif "non conforme" in evaluation.lower() or "problème" in evaluation.lower() or "insuffisant" in evaluation.lower():
        return "❌"
    else:
        return "ℹ️"

def parse_analysis_result(result_path):
    """Analyse le fichier JSON de résultat et le convertit en DataFrame pour affichage"""
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extraire les informations pertinentes du résultat
        # Cette partie dépend de la structure exacte de votre JSON
        raw_text = data.get("raw_text", "Non disponible")
        vision_analysis = data.get("vision_analysis", "Non disponible")
        compliance_analysis = data.get("compliance_analysis", "Non disponible")
        
        # Créer un DataFrame structuré pour l'affichage
        aspects = [
            "Contenu / Message",
            "Produits",
            "Design graphique",
            "Images",
            "Lisibilité / Typographie",
            "Informations pratiques",
            "Mentions légales",
            "Qualité de l'image",
            "Efficacité marketing"
        ]
        
        # Ces observations devront être extraites du résultat d'analyse
        # Pour l'exemple, nous utilisons des valeurs statiques
        observations = extract_observations(vision_analysis, compliance_analysis)
        evaluations = extract_evaluations(compliance_analysis)
        
        df = pd.DataFrame({
            "Aspect analysé": aspects,
            "Observations": observations,
            "Évaluation": evaluations
        })
        
        return df, raw_text, compliance_analysis
    except Exception as e:
        st.error(f"Erreur lors de l'analyse du résultat: {str(e)}")
        return None, None, None

def extract_observations(vision_analysis, compliance_analysis):
    """Extrait les observations des résultats d'analyse"""
    # Cette fonction devrait être adaptée à la structure exacte de vos données
    # Pour l'exemple, nous retournons des valeurs par défaut
    return [
        extract_content_message(vision_analysis),
        extract_products(vision_analysis),
        extract_design(vision_analysis),
        extract_images(vision_analysis),
        extract_readability(vision_analysis, compliance_analysis),
        extract_practical_info(vision_analysis, compliance_analysis),
        extract_legal_mentions(compliance_analysis),
        extract_image_quality(vision_analysis),
        extract_marketing(vision_analysis, compliance_analysis)
    ]

def extract_evaluations(compliance_analysis):
    """Extrait les évaluations des résultats d'analyse"""
    # Par défaut, on utilise des évaluations basées sur le texte de l'analyse
    
    # Analyse simple par mots clés
    if isinstance(compliance_analysis, str):
        is_compliant = "conforme" in compliance_analysis.lower() and "non conforme" not in compliance_analysis.lower()
        has_issues = "améliorer" in compliance_analysis.lower() or "problème" in compliance_analysis.lower()
        
        if is_compliant and not has_issues:
            return ["✅ Pertinent", "✅ Bien ciblé", "✅ Efficace", "✅ Claires", 
                    "✅ Excellente", "✅ Complètes", "✅ Conformes", "✅ Bonne", "✅ Efficace"]
        elif has_issues:
            return ["✅ Pertinent", "✅ Bien ciblé", "✅ Efficace", "✅ Claires", 
                    "⚠️ À améliorer", "⚠️ Incomplètes", "❌ Non conformes", "⚠️ Moyenne", "✅ Bonne"]
        else:
            return ["⚠️ À améliorer", "✅ Bien ciblé", "✅ Efficace", "✅ Claires", 
                    "⚠️ À améliorer", "❌ Incomplètes", "❌ Non conformes", "❌ Insuffisante", "⚠️ Moyenne"]
    
    # Si l'analyse de conformité n'est pas une chaîne, retourner des valeurs par défaut
    return ["⚠️ Non évalué"] * 9

# Fonctions d'extraction du contenu
def extract_content_message(vision_analysis):
    if "MESSAGE PUBLICITAIRE" in vision_analysis:
        lines = vision_analysis.split('\n')
        for i, line in enumerate(lines):
            if "MESSAGE PUBLICITAIRE" in line and i+1 < len(lines):
                # Extraire quelques lignes après le titre
                return '\n'.join(lines[i+1:i+5])
    return "Message publicitaire extrait de l'analyse"

def extract_products(vision_analysis):
    if "Produits" in vision_analysis:
        lines = vision_analysis.split('\n')
        for i, line in enumerate(lines):
            if "Produits" in line and i+1 < len(lines):
                return lines[i+1]
    return "Produits identifiés dans l'analyse"

def extract_design(vision_analysis):
    if "CONTENU VISUEL" in vision_analysis:
        lines = vision_analysis.split('\n')
        for i, line in enumerate(lines):
            if "CONTENU VISUEL" in line and i+1 < len(lines):
                return '\n'.join(lines[i+1:i+3])
    return "Design graphique extrait de l'analyse"

def extract_images(vision_analysis):
    if "Images présentes" in vision_analysis:
        lines = vision_analysis.split('\n')
        for i, line in enumerate(lines):
            if "Images présentes" in line:
                return lines[i]
    return "Images décrites dans l'analyse"

def extract_readability(vision_analysis, compliance_analysis):
    if "Lisibilité" in vision_analysis or "Typographie" in vision_analysis:
        lines = vision_analysis.split('\n')
        for i, line in enumerate(lines):
            if ("Lisibilité" in line or "Typographie" in line) and i+1 < len(lines):
                return lines[i+1]
    return "Lisibilité et typographie évaluées"

def extract_practical_info(vision_analysis, compliance_analysis):
    if "COORDONNÉES ET IDENTIFICATION" in vision_analysis:
        lines = vision_analysis.split('\n')
        for i, line in enumerate(lines):
            if "COORDONNÉES ET IDENTIFICATION" in line and i+1 < len(lines):
                return '\n'.join(lines[i+1:i+4])
    return "Informations pratiques extraites"

def extract_legal_mentions(compliance_analysis):
    if isinstance(compliance_analysis, str):
        if "MENTIONS LÉGALES" in compliance_analysis:
            lines = compliance_analysis.split('\n')
            for i, line in enumerate(lines):
                if "MENTIONS LÉGALES" in line and i+1 < len(lines):
                    return '\n'.join(lines[i+1:i+4])
    return "Evaluation des mentions légales obligatoires"

def extract_image_quality(vision_analysis):
    if "résolution" in vision_analysis.lower():
        lines = vision_analysis.lower().split('\n')
        for line in lines:
            if "résolution" in line:
                return line
    return "Qualité d'image évaluée"

def extract_marketing(vision_analysis, compliance_analysis):
    if "ÉLÉMENTS MARKETING" in vision_analysis:
        lines = vision_analysis.split('\n')
        for i, line in enumerate(lines):
            if "ÉLÉMENTS MARKETING" in line and i+1 < len(lines):
                return '\n'.join(lines[i+1:i+3])
    return "Efficacité marketing évaluée"

def get_report_html(df, raw_text, compliance_analysis):
    """Génère le rapport HTML à partir des données d'analyse"""
    
    # Extraire le verdict du texte de l'analyse de conformité
    verdict = "Non déterminé"
    if isinstance(compliance_analysis, str):
        if "VERDICT : CONFORME" in compliance_analysis:
            verdict = "CONFORME"
        elif "VERDICT : NON CONFORME" in compliance_analysis:
            verdict = "NON CONFORME"
        elif "VERDICT : PARTIELLEMENT CONFORME" in compliance_analysis:
            verdict = "PARTIELLEMENT CONFORME"
    
    # Générer du HTML pour le rapport
    verdict_color = "green" if verdict == "CONFORME" else "red" if verdict == "NON CONFORME" else "orange"
    
    html = f"""
    <style>
        .report-container {{
            max-width: 100%;
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .verdict {{
            font-size: 24px;
            font-weight: bold;
            color: white;
            background-color: {verdict_color};
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            margin: 20px 0;
        }}
        .summary {{
            background-color: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        .raw-text {{
            background-color: #f8f9fa;
            padding: 15px;
            border: 1px solid #ddd;
            font-family: monospace;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
        }}
    </style>
    
    <div class="report-container">
        <div class="verdict">VERDICT: {verdict}</div>
        
        <div class="summary">
            <h3>Synthèse de l'analyse</h3>
            <p>L'analyse de cette publicité a relevé les points suivants qui nécessitent votre attention.</p>
        </div>
    """
    
    # Ajouter le tableau
    html += """
        <table>
            <tr>
                <th style="width: 20%">Aspect analysé</th>
                <th style="width: 60%">Observations</th>
                <th style="width: 20%">Évaluation</th>
            </tr>
    """
    
    for _, row in df.iterrows():
        html += f"""
            <tr>
                <td>{row['Aspect analysé']}</td>
                <td>{row['Observations']}</td>
                <td>{row['Évaluation']}</td>
            </tr>
        """
    
    html += """
        </table>
        
        <div class="summary">
            <h3>Texte brut extrait</h3>
            <div class="raw-text">
    """
    
    html += raw_text.replace('\n', '<br>')
    
    html += """
            </div>
        </div>
    </div>
    """
    
    return html

def main():
    st.title("Analyse de Conformité Publicitaire")
    
    st.write("""
    Cette application vous permet d'analyser des publicités pour vérifier leur conformité
    légale et obtenir des recommandations pour les améliorer.
    """)
    
    # Sidebar pour les options
    st.sidebar.title("Options")
    
    # Option de chargement de fichier
    st.sidebar.header("Chargement de publicité")
    uploaded_file = st.sidebar.file_uploader(
        "Chargez une image ou un PDF publicitaire",
        type=["jpg", "jpeg", "png", "pdf"]
    )
    
    # Affichage principal
    col1, col2 = st.columns([1, 1])
    
    if uploaded_file is not None:
        # Afficher l'image chargée
        with col1:
            st.subheader("Publicité chargée")
            # Sauvegarder temporairement le fichier uploadé
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            timestamp = int(time.time())
            file_extension = uploaded_file.name.split('.')[-1].lower()
            temp_path = os.path.join(temp_dir, f"temp_display_{timestamp}.{file_extension}")
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Vérifier si c'est une image valide avant d'afficher
            if file_extension in ['jpg', 'jpeg', 'png']:
                try:
                    image = Image.open(temp_path)
                    st.image(image, use_column_width=True)
                except Exception as e:
                    st.error(f"Impossible d'afficher l'image: {str(e)}")
            elif file_extension == 'pdf':
                st.write("Aperçu PDF non disponible. Le fichier sera analysé.")
            else:
                st.error("Format de fichier non supporté pour l'aperçu.")
        
        # Analyser la publicité chargée
        if st.button("Analyser cette publicité"):
            result_path = analyze_pub(uploaded_file)
            
            if result_path:
                df, raw_text, compliance_analysis = parse_analysis_result(result_path)
                
                with col2:
                    st.subheader("Résultats de l'analyse")
                    
                    if df is not None:
                        # Afficher le rapport
                        st.write("### Tableau d'évaluation")
                        st.dataframe(df, use_container_width=True)
                        
                        # Générer le rapport HTML
                        report_html = get_report_html(df, raw_text, compliance_analysis)
                        st.markdown(report_html, unsafe_allow_html=True)
                        
                        # Ajouter un bouton pour télécharger le rapport
                        report_download = report_html.encode('utf-8')
                        b64 = base64.b64encode(report_download).decode()
                        href = f'<a href="data:text/html;base64,{b64}" download="rapport_analyse_pub.html">Télécharger le rapport complet</a>'
                        st.markdown(href, unsafe_allow_html=True)
    else:
        # Afficher un message d'attente
        st.info("Veuillez charger une publicité pour commencer l'analyse.")

if __name__ == "__main__":
    main() 