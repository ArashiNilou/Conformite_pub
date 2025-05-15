import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from config.azure_config import AzureConfig
from models.ai_models import AIModels
from tools.tools import Tools
from raptor.raptor_setup import RaptorSetup
from agent.react_agent import create_react_agent
from llama_index.core.callbacks import CBEventType, CallbackManager
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from utils.token_counter import TokenCounter, create_token_counter
from PIL import Image
import io
import os
import sys
from utils.pdf_converter import convert_pdf_to_image
from utils.output_saver import save_output
import json
from datetime import datetime
from utils.raw_text_extractor import RawTextExtractor
from utils.company_verifier import check_company_in_text
import traceback

class CustomCallbackHandler(BaseCallbackHandler):
    """Handler personnalisé pour logger les événements de l'agent"""

    # Mapping centralisé action → étape
    ACTION_TO_STEP = {
        "analyze_vision": "vision_analysis",
        "verify_consistency": "consistency_check",
        "verify_product_logo_consistency": "product_logo_consistency",
        "verify_dates": "dates_verification",
        "search_legislation": "legislation",
        "get_clarifications": "clarifications",
        "analyze_compliance": "compliance_analysis",
        "extract_raw_text": "raw_text"
    }

    def __init__(self) -> None:
        super().__init__([], [])
        print("\n🔄 Initialisation du CustomCallbackHandler")
        self.steps = {
            "vision_analysis": "",
            "consistency_check": "",
            "product_logo_consistency": "",
            "dates_verification": "",
            "legislation": "",
            "clarifications": "",
            "compliance_analysis": "",
            "raw_text": ""
        }
        print("✅ Steps initialisés :", self.steps.keys())
        self.current_action = None
        self.token_counter = None  # Sera défini plus tard
        # Désactiver la sauvegarde intermédiaire des résultats
        self.save_intermediate_results = False

    def set_token_counter(self, token_counter: TokenCounter) -> None:
        """Définir le compteur de tokens pour ce handler."""
        self.token_counter = token_counter

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Log le début d'un événement"""
        if event_type == CBEventType.FUNCTION_CALL and payload:
            tool_metadata = payload.get("tool", {})
            if hasattr(tool_metadata, "name"):
                self.current_action = tool_metadata.name
                print(f"✅ Action courante mise à jour: {self.current_action}")

                # Définir l'étape courante dans le compteur de tokens via mapping
                if self.token_counter:
                    step = self.ACTION_TO_STEP.get(self.current_action, "other")
                    self.token_counter.set_current_step(step)

        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Log la fin d'un événement"""
        if event_type == CBEventType.FUNCTION_CALL and payload:
            if "function_call_response" in payload:
                print(f"\n{'='*50}")
                print(f"📝 Traitement de la réponse pour l'action: {self.current_action}")
                response = str(payload["function_call_response"])

                # Supprimer le préfixe "assistant:" s'il est présent
                if response.startswith("assistant:"):
                    response = response[len("assistant:"):].strip()

                # Stocker la réponse selon l'action via mapping
                step = self.ACTION_TO_STEP.get(self.current_action)
                if step and step in self.steps:
                    print(f"💾 Sauvegarde de la réponse pour l'étape: {step}...")
                    self.steps[step] = response

                print(f"{'='*50}\n")

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Début d'une trace"""
        print(f"\n📝 Début de la trace: {trace_id}")

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Fin d'une trace"""
        print(f"📝 Fin de la trace: {trace_id}\n")
        
        # Afficher les statistiques de tokens à la fin de l'analyse
        if self.token_counter:
            self.token_counter.print_step_stats()
            self.token_counter.save_stats()

def initialize_system(callback_handler):
    """
    Initialise les composants du système d'analyse
    
    Args:
        callback_handler: Gestionnaire d'événements
        
    Returns:
        tuple: (azure_config, ai_models, tools, raptor_setup)
    """
    print("\n🔄 Initialisation du système...")
    
    # Configuration
    azure_config = AzureConfig()
    
    # Modèles IA
    ai_models = AIModels(azure_config)
    
    # Base de connaissances Raptor
    raptor_setup = RaptorSetup(ai_models=ai_models)
    
    # Outils d'analyse
    tools = Tools(llm=ai_models.llm, raptor=raptor_setup)
    
    print("✅ Système initialisé avec succès\n")
    
    return azure_config, ai_models, tools, raptor_setup

async def analyze_image(image_path: str, agent = None) -> None:
    """
    Analyse une image ou un PDF avec l'agent React
    
    Args:
        image_path: Chemin vers l'image ou le PDF à analyser
        agent: Agent React préconfigurer (optionnel)
    """
    # Valider et préparer le chemin du fichier
    path = validate_image_path(image_path)
    if not path:
        print(f"❌ Fichier invalide : {image_path}")
        return

    # Convertir le PDF en image si nécessaire
    if Path(path).suffix.lower() == '.pdf':
        try:
            print(f"🔄 Conversion du PDF en image...")
            path = convert_pdf_to_image(path)
            print(f"✅ PDF converti en image : {path}")
        except Exception as e:
            print(f"❌ Erreur lors de la conversion du PDF : {e}")
            return
    
    # Si aucun agent n'est fourni, en créer un nouveau
    if agent is None:
        # Initialiser le système
        callback_handler = CustomCallbackHandler()
        azure_config, ai_models, tools, raptor_setup = initialize_system(callback_handler)
        
        # Créer un CallbackManager avec notre handler
        callback_manager = CallbackManager([callback_handler])
        
        # Ajouter le compteur de tokens au gestionnaire de callbacks
        token_counter = create_token_counter(verbose=True, save_dir="stats/tokens")
        callback_manager.add_handler(token_counter)
        
        # Connecter le compteur de tokens au callback_handler
        callback_handler.set_token_counter(token_counter)
        
        # Créer l'agent
        agent = create_react_agent(
            ai_models=ai_models, 
            tools=tools, 
            callback_manager=callback_manager,
            verbose=True
        )
    else:
        callback_handler = next((handler for handler in agent.callback_manager.handlers if isinstance(handler, CustomCallbackHandler)), None)
        tools = agent.tools

    start_time = datetime.now()
    print(f"⏱️  Début de l'analyse : {start_time.strftime('%H:%M:%S')}")
    
    # Variable pour stocker les résultats
    final_output_path = None
    
    # Exécuter l'analyse
    try:
        # Essayer d'abord avec `aquery` qui est souvent utilisé dans les versions récentes
        if hasattr(agent, 'aquery'):
            raw_response = await asyncio.wait_for(agent.aquery(path), timeout=300)  # Timeout de 5 minutes
        # Sinon essayer avec `achat` 
        elif hasattr(agent, 'achat'):
            raw_response = await asyncio.wait_for(agent.achat(path), timeout=300)  # Timeout de 5 minutes
        # Ou essayer avec `run` en mode synchrone si nécessaire
        elif hasattr(agent, 'run'):
            raw_response = agent.run(path)  # Pas de timeout pour run synchrone
        else:
            raise AttributeError("L'agent ne possède aucune méthode appropriée pour l'exécution (aquery, achat, run)")
        
        # Convertir la réponse en chaîne de caractères
        if hasattr(raw_response, 'response'):
            response = raw_response.response
        elif hasattr(raw_response, 'result'):
            response = raw_response.result
        elif hasattr(raw_response, 'output'):
            response = raw_response.output
        elif hasattr(raw_response, 'message'):
            response = raw_response.message
        elif isinstance(raw_response, dict) and 'response' in raw_response:
            response = raw_response['response']
        elif isinstance(raw_response, str):
            response = raw_response
        else:
            # En dernier recours, convertir en chaîne de caractères
            response = str(raw_response)
            print(f"⚠️ Conversion de l'objet Response en chaîne - type original: {type(raw_response)}")
        
        # Vérification automatique du nom d'entreprise via API Sirene
        raw_text = callback_handler.steps.get("raw_text", "")
        company_verification = check_company_in_text(raw_text)
        if company_verification.get("detected"):
            print(f"\n🔎 Vérification du nom d'entreprise : {company_verification}")
        else:
            print("\n🔎 Aucun nom d'entreprise détecté dans le texte brut.")
    
        # Attendre UNIQUEMENT la fin de toutes les étapes pour sauvegarder le résultat final
        final_output_path = save_output(path, {
            "converted_file": path if path != image_path else None,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "steps": {
                "vision_analysis": callback_handler.steps["vision_analysis"],
                "consistency_check": callback_handler.steps["consistency_check"],
                "product_logo_consistency": callback_handler.steps["product_logo_consistency"],
                "dates_verification": callback_handler.steps["dates_verification"],
                "legislation": callback_handler.steps["legislation"],
                "clarifications": callback_handler.steps["clarifications"],
                "compliance_analysis": callback_handler.steps["compliance_analysis"],
                "raw_text": callback_handler.steps["raw_text"]
            },
            "company_verification": company_verification,
            "final_response": response
        })
        
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"⏱️  Fin de l'analyse : {end_time.strftime('%H:%M:%S')} (durée: {duration})")
        print(f"🏁 Analyse terminée")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse de l'image : {str(e)}")
        traceback.print_exc()
    
    return final_output_path

def validate_image_path(path: str) -> str:
    """
    Valide le chemin de l'image ou du PDF
    
    Args:
        path: Chemin à valider
        
    Returns:
        str: Chemin validé
        
    Raises:
        ArgumentTypeError: Si le chemin n'est pas valide
    """
    file_path = Path(path)
    if not file_path.exists():
        raise argparse.ArgumentTypeError(f"Le fichier {path} n'existe pas")
    if not file_path.is_file():
        raise argparse.ArgumentTypeError(f"{path} n'est pas un fichier")
    if file_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.pdf']:
        raise argparse.ArgumentTypeError(f"{path} n'est pas un format supporté (formats acceptés : jpg, jpeg, png, pdf)")
    return str(file_path.absolute())

def get_files_to_analyze(path: str, recursive: bool = False) -> List[str]:
    """
    Récupère la liste des fichiers à analyser
    
    Args:
        path: Chemin vers le fichier ou dossier
        recursive: Explorer récursivement les sous-dossiers
        
    Returns:
        List[str]: Liste des chemins de fichiers à analyser
    """
    path_obj = Path(path)
    supported_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.pdf', '.tiff', '.webp']
    files = []
    
    if path_obj.is_file():
        # C'est un fichier unique
        if path_obj.suffix.lower() in supported_extensions:
            print(f"✅ Fichier à analyser: {path_obj}")
            return [str(path_obj)]
        else:
            print(f"⚠️ Format de fichier non supporté: {path_obj}")
            return []
    
    elif path_obj.is_dir():
        # C'est un dossier
        print(f"📂 Analyse du dossier: {path_obj}")
        
        # Récupérer tous les fichiers du dossier avec les extensions supportées
        if recursive:
            for ext in supported_extensions:
                files.extend(list(path_obj.glob(f"**/*{ext}")))
        else:
            for ext in supported_extensions:
                files.extend(list(path_obj.glob(f"*{ext}")))
                
        files = [str(f) for f in files]
        
        if files:
            print(f"✅ {len(files)} fichiers trouvés dans le dossier")
        else:
            print(f"⚠️ Aucun fichier supporté trouvé dans le dossier")
            
        return files
    
    else:
        print(f"❌ Chemin invalide: {path_obj}")
        return []

async def analyze_files(files: List[str]) -> None:
    """
    Analyse une liste de fichiers
    
    Args:
        files: Liste des chemins de fichiers à analyser
    """
    if not files:
        print("⚠️ Aucun fichier à analyser")
        return
        
    print(f"🔍 Analyse de {len(files)} fichier(s)...")
    
    # Initialiser le système
    callback_handler = CustomCallbackHandler()
    azure_config, ai_models, tools, raptor_setup = initialize_system(callback_handler)
    
    # Créer un CallbackManager avec notre handler
    callback_manager = CallbackManager([callback_handler])
    
    # Ajouter le compteur de tokens au gestionnaire de callbacks
    token_counter = create_token_counter(verbose=True, save_dir="stats/tokens")
    callback_manager.add_handler(token_counter)
    
    # Connecter le compteur de tokens au callback_handler
    callback_handler.set_token_counter(token_counter)
    
    # Créer l'agent
    agent = create_react_agent(
        ai_models=ai_models, 
        tools=tools, 
        callback_manager=callback_manager,
        verbose=True
    )
    
    # Analyser chaque fichier
    for file_path in files:
        try:
            print(f"\n📄 Analyse du fichier: {file_path}")
            await analyze_image(file_path)
        except Exception as e:
            print(f"❌ Erreur lors de l'analyse de {file_path}: {str(e)}")
            
    print("\n✅ Analyse terminée")

def test_text_extraction(files: List[str], tools: Tools, mode: str = "docling", ocr_engine: str = "tesseract") -> None:
    """
    Teste la fonctionnalité d'extraction de texte sur les fichiers spécifiés
    
    Args:
        files: Liste des fichiers à analyser
        tools: Instance des outils avec la fonctionnalité d'extraction de texte
        mode: Mode d'extraction ('docling', 'gpt4v', 'azure_cv')
        ocr_engine: Moteur OCR à utiliser avec Docling ('tesseract', 'easyocr', 'rapidocr', 'tesseract_api')
    """
    print(f"\n🧪 Test de l'extraction de texte sur {len(files)} fichier(s)...")
    print(f"📊 Mode: {mode}, Moteur OCR: {ocr_engine if mode == 'docling' else 'N/A'}")
    
    for file_path in files:
        print(f"\n📄 Fichier: {file_path}")
        try:
            extracted_text = tools.extract_text_from_image(file_path, mode, ocr_engine)
            print("\n📝 Texte extrait:")
            print("-" * 50)
            print(extracted_text)
            print("-" * 50)
        except Exception as e:
            print(f"❌ Erreur lors de l'extraction de texte: {str(e)}")

def extract_raw_text(files: List[str], method: str = "auto") -> None:
    """
    Utilise le nouvel extracteur pour obtenir le texte brut des images sans corrections
    
    Args:
        files: Liste des fichiers à analyser
        method: Méthode d'extraction à utiliser ('tesseract', 'easyocr', 'auto', 'gpt_vision')
    """
    print(f"\n🔍 Extraction de texte BRUT sur {len(files)} fichier(s)...")
    print(f"📊 Méthode: {method}")
    
    # Si nous utilisons GPT Vision, utiliser les outils de l'application principale
    if method == "gpt_vision":
        # Initialiser le système
        callback_handler = CustomCallbackHandler()
        azure_config, ai_models, tools, raptor_setup = initialize_system(callback_handler)
        
        for file_path in files:
            print(f"\n📄 Fichier: {file_path}")
            
            try:
                # Extraire le texte brut avec GPT Vision
                extracted_text = tools.extract_raw_text_with_vision(file_path)
                
                print("\n===== TEXTE EXTRAIT AVEC GPT VISION =====")
                print("-" * 50)
                print(extracted_text if extracted_text else "[Aucun texte extrait]")
                print("-" * 50)
                
            except Exception as e:
                print(f"❌ Erreur lors de l'extraction de texte avec GPT Vision: {str(e)}")
        
        return
    
    # Initialiser l'extracteur de texte brut traditionnel pour les autres méthodes
    extractor = RawTextExtractor()
    
    for file_path in files:
        print(f"\n📄 Fichier: {file_path}")
        
        try:
            # Extraire le texte brut
            results = extractor.extract_raw_text(file_path, method)
            
            if not results:
                print("❌ Aucun texte extrait")
                continue
                
            # Afficher les résultats
            for method_name, text in results.items():
                print(f"\n===== TEXTE EXTRAIT AVEC {method_name.upper()} =====")
                print("-" * 50)
                print(text if text else "[Aucun texte extrait]")
                print("-" * 50)
                
                # Ne pas sauvegarder les résultats intermédiaires en fichiers distincts
                print(f"💾 Texte extrait avec {method_name} (affiché mais non sauvegardé)")
            
        except Exception as e:
            print(f"❌ Erreur lors de l'extraction de texte: {str(e)}")

def parse_args():
    """Parse les arguments de la ligne de commande"""
    parser = argparse.ArgumentParser(description="Analyse de publicités pour conformité légale")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--files", nargs="+", help="Chemins vers les fichiers à analyser")
    group.add_argument("--dir", help="Chemin vers le répertoire contenant les fichiers à analyser")
    parser.add_argument("--test_text_extraction", action="store_true", 
                        help="Active uniquement le test d'extraction de texte sans analyse complète")
    parser.add_argument("--extract_raw_text", action="store_true",
                        help="Extrait le texte brut des images sans corrections")
    parser.add_argument("--mode", choices=["docling", "gpt4v", "azure_cv"], default="docling",
                        help="Mode d'extraction de texte (docling, gpt4v, azure_cv)")
    parser.add_argument("--ocr", choices=["tesseract", "easyocr", "rapidocr", "tesseract_api"], 
                        default="tesseract", help="Moteur OCR à utiliser avec Docling")
    parser.add_argument("--method", choices=["tesseract", "easyocr", "auto", "gpt_vision"], default="auto",
                        help="Méthode d'extraction de texte brut")
    
    return parser.parse_args()

def main():
    """Point d'entrée principal de l'application"""
    args = parse_args()
    
    if args.files or args.dir:
        callback_handler = CustomCallbackHandler()
        azure_config, ai_models, tools, raptor_setup = initialize_system(callback_handler)
        
        files_to_analyze = get_files_to_analyze(args.dir if args.dir else args.files[0])
        
        if args.test_text_extraction:
            test_text_extraction(files_to_analyze, tools, args.mode, args.ocr)
        elif args.extract_raw_text:
            extract_raw_text(files_to_analyze, args.method)
        else:
            asyncio.run(analyze_files(files_to_analyze))
    else:
        print("❌ Aucun fichier ou répertoire spécifié. Utilisez --file ou --dir.")

if __name__ == "__main__":
    main() 