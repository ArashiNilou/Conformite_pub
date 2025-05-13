import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import os
import shutil

class OutputSaver:
    """Gère la sauvegarde des résultats d'analyse en JSON"""
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialise le gestionnaire de sauvegarde
        
        Args:
            output_dir: Répertoire de sauvegarde des résultats
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_analysis: Dict[str, Any] = {
            "timestamp": "",
            "image_path": "",
            "raw_text": "",
            "vision_result": "",
            "consistency_check": "",
            "dates_verification": "",
            "legislation": "",
            "clarifications": "",
            "compliance_analysis": "",
            "extracted_text": "",
            "product_logo_consistency": ""  # Nouveau champ pour les incohérences produit/logo
        }
    
    def _generate_filename(self, image_path: str) -> str:
        """
        Génère un nom de fichier unique pour les résultats
        
        Args:
            image_path: Chemin de l'image analysée
            
        Returns:
            str: Nom du fichier de sauvegarde
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = Path(image_path).stem
        return f"analyse_{image_name}_{timestamp}.json"
    
    def start_new_analysis(self, image_path: str) -> None:
        """
        Démarre une nouvelle analyse
        
        Args:
            image_path: Chemin de l'image à analyser
        """
        self.current_analysis = {
            "timestamp": datetime.now().isoformat(),
            "image_path": str(image_path),
            "raw_text": "",
            "vision_result": "",
            "consistency_check": "",
            "dates_verification": "",
            "legislation": "",
            "clarifications": "",
            "compliance_analysis": "",
            "extracted_text": "",
            "product_logo_consistency": ""
        }
    
    def save_vision_result(self, result: str) -> None:
        """Sauvegarde le résultat de l'analyse visuelle"""
        self.current_analysis["vision_result"] = result
        self._save_current_analysis()
    
    def save_consistency_check(self, result: str) -> None:
        """Sauvegarde le résultat de la vérification de cohérence"""
        self.current_analysis["consistency_check"] = result
        self._save_current_analysis()
    
    def save_legislation(self, result: str) -> None:
        """Sauvegarde la législation trouvée"""
        self.current_analysis["legislation"] = result
        self._save_current_analysis()
    
    def save_clarifications(self, result: str) -> None:
        """Sauvegarde les clarifications"""
        self.current_analysis["clarifications"] = result
        self._save_current_analysis()
    
    def save_compliance_analysis(self, result: str) -> None:
        """Sauvegarde l'analyse de conformité"""
        self.current_analysis["compliance_analysis"] = result
        self._save_current_analysis()
    
    def save_text_extraction(self, result: str, mode: str = "docling") -> None:
        """
        Sauvegarde le texte extrait de l'image
        
        Args:
            result: Texte extrait
            mode: Mode d'extraction utilisé (docling, gpt4v, azure_cv)
        """
        self.current_analysis["extracted_text"] = {
            "text": result,
            "extraction_mode": mode,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self._save_current_analysis()
    
    def save_raw_text(self, result: str) -> None:
        """
        Sauvegarde le texte brut extrait de l'image
        
        Args:
            result: Texte brut extrait
        """
        self.current_analysis["raw_text"] = result
        self._save_current_analysis()
        
        # Créer un dossier spécifique pour les textes bruts
        raw_text_dir = self.output_dir / "raw_text"
        raw_text_dir.mkdir(exist_ok=True)
        
        # Sauvegarder dans un fichier texte séparé
        if self.current_analysis["image_path"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{Path(self.current_analysis['image_path']).stem}_raw_{timestamp}.txt"
            output_path = raw_text_dir / filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
                
            print(f"\n💾 Texte brut sauvegardé séparément dans : {output_path}")
        
        return
    
    def save_dates_verification(self, result: str) -> None:
        """
        Sauvegarde le résultat de la vérification des dates
        
        Args:
            result: Résultat de la vérification des dates
        """
        self.current_analysis["dates_verification"] = result
        self._save_current_analysis()
        
        # Créer un dossier spécifique pour les vérifications de dates
        dates_dir = self.output_dir / "dates_verification"
        dates_dir.mkdir(exist_ok=True)
        
        # Sauvegarder dans un fichier texte séparé
        if self.current_analysis["image_path"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{Path(self.current_analysis['image_path']).stem}_dates_{timestamp}.txt"
            output_path = dates_dir / filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
                
            print(f"\n💾 Vérification des dates sauvegardée séparément dans : {output_path}")
    
    def save_product_logo_consistency(self, result: str) -> None:
        """Sauvegarde le résultat de la vérification de cohérence produit/logo"""
        self.current_analysis["product_logo_consistency"] = result
        self._save_current_analysis()
    
    def is_analysis_in_progress(self) -> bool:
        """
        Vérifie si une analyse est actuellement en cours
        
        Returns:
            bool: True si une analyse est en cours, False sinon
        """
        return bool(self.current_analysis.get("image_path", ""))
    
    def _save_current_analysis(self) -> None:
        """Sauvegarde l'analyse en cours dans un fichier JSON"""
        if not self.current_analysis["image_path"]:
            raise ValueError("Aucune analyse en cours")
            
        filename = self._generate_filename(self.current_analysis["image_path"])
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.current_analysis, f, ensure_ascii=False, indent=2)
            
        print(f"\n💾 Résultats sauvegardés dans : {output_path}")

def make_json_serializable(obj):
    """
    Convertit les objets non-sérialisables en format compatible avec JSON
    
    Args:
        obj: Objet à convertir
        
    Returns:
        Un objet compatible JSON
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    else:
        try:
            # Essayer des attributs communs pour obtenir une représentation textuelle
            if hasattr(obj, 'response'):
                return str(obj.response)
            elif hasattr(obj, 'text'):
                return str(obj.text)
            elif hasattr(obj, 'content'):
                return str(obj.content)
            elif hasattr(obj, '__str__'):
                return str(obj)
            else:
                return repr(obj)
        except Exception:
            return repr(obj)

def save_output(input_path: str, analysis_data: Dict[str, Any]) -> str:
    """
    Sauvegarde les résultats d'analyse en JSON
    
    Args:
        input_path: Chemin de l'image ou du PDF analysé
        analysis_data: Données d'analyse
        
    Returns:
        str: Chemin du fichier de sortie
    """
    input_file = Path(input_path)
    
    # Créer le répertoire de sortie
    base_output_dir = Path("outputs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_dir = base_output_dir / input_file.stem
    source_dir.mkdir(parents=True, exist_ok=True)
    
    # Créer le nom du fichier de sortie
    output_path = source_dir / f"analyse_{timestamp}.json"
    
    # Fonction pour rendre les données sérialisables en JSON
    def make_json_serializable(data):
        if data is None:
            return ""
        if isinstance(data, (str, int, float, bool)):
            return data
        if isinstance(data, (list, tuple)):
            return [make_json_serializable(item) for item in data]
        if isinstance(data, dict):
            return {k: make_json_serializable(v) for k, v in data.items()}
        return str(data)
    
    # Nettoyer les données pour JSON
    clean_data = {
        "input_file": str(input_file.absolute()),
        "converted_file": analysis_data.get("converted_file"),
        "timestamp": analysis_data.get("timestamp"),
        "steps": {
            "vision_analysis": make_json_serializable(analysis_data.get("steps", {}).get("vision_analysis", "")),
            "consistency_check": make_json_serializable(analysis_data.get("steps", {}).get("consistency_check", "")),
            "product_logo_consistency": make_json_serializable(analysis_data.get("steps", {}).get("product_logo_consistency", "")),
            "dates_verification": make_json_serializable(analysis_data.get("steps", {}).get("dates_verification", "")),
            "legislation": make_json_serializable(analysis_data.get("steps", {}).get("legislation", "")),
            "clarifications": make_json_serializable(analysis_data.get("steps", {}).get("clarifications", "")),
            "compliance_analysis": make_json_serializable(analysis_data.get("steps", {}).get("compliance_analysis", "")),
            "text_extraction": make_json_serializable(analysis_data.get("steps", {}).get("text_extraction", "")),
            "raw_text": make_json_serializable(analysis_data.get("steps", {}).get("raw_text", ""))
        },
        "final_response": make_json_serializable(analysis_data.get("final_response", "")),
        "extracted_text": make_json_serializable(analysis_data.get("extracted_text", "")),
        "company_verification": make_json_serializable(analysis_data.get("company_verification", {}))
    }
    
    # Sauvegarder en JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)
        
    # Copier l'image convertie si elle existe
    if clean_data.get("converted_file"):
        image_path = Path(clean_data["converted_file"])
        if image_path.exists():
            image_output = source_dir / f"image_{timestamp}{image_path.suffix}"
            shutil.copy2(image_path, image_output)
            
    return str(output_path.relative_to(base_output_dir.parent)) 