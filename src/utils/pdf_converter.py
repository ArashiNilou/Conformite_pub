import fitz
from pathlib import Path
import os
from datetime import datetime
import argparse

def convert_pdf_to_image(pdf_path: str, dpi: int = 300, format: str = "png") -> str:
    """
    Convertit un PDF en image
    
    Args:
        pdf_path: Chemin vers le PDF
        dpi: Résolution de l'image (dots per inch)
        format: Format de l'image (png, jpg)
        
    Returns:
        str: Chemin vers l'image convertie
    """
    try:
        # Créer le dossier converted_images s'il n'existe pas
        output_dir = Path("converted_images")
        output_dir.mkdir(exist_ok=True)
        
        # Générer le nom du fichier de sortie
        pdf_name = Path(pdf_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{pdf_name}_{timestamp}.{format}"
        
        # Ouvrir le PDF
        pdf = fitz.open(pdf_path)
        
        # Convertir la première page en image avec la résolution spécifiée
        page = pdf[0]
        zoom = dpi / 72  # zoom factor (72 dpi est la résolution par défaut)
        mat = fitz.Matrix(zoom, zoom)  # matrice de transformation pour le zoom
        pix = page.get_pixmap(matrix=mat)
        
        # Sauvegarder l'image
        pix.save(str(output_path))
        print(f"✅ PDF converti en image : {output_path}")
        print(f"📊 Résolution: {dpi} dpi, Format: {format}, Dimensions: {pix.width}x{pix.height}")
        
        return str(output_path)
        
    except Exception as e:
        print(f"❌ Erreur lors de la conversion du PDF : {str(e)}")
        raise
    finally:
        if 'pdf' in locals():
            pdf.close()

if __name__ == "__main__":
    # Permet d'exécuter le script directement pour tester
    parser = argparse.ArgumentParser(description="Convertit un PDF en image avec contrôle de qualité")
    parser.add_argument("pdf_path", help="Chemin vers le fichier PDF à convertir")
    parser.add_argument("--dpi", type=int, default=300, help="Résolution de l'image (dpi)")
    parser.add_argument("--format", choices=["png", "jpg", "jpeg"], default="png", 
                        help="Format de sortie de l'image")
    
    args = parser.parse_args()
    
    # Convertir le PDF en image avec les paramètres spécifiés
    output_path = convert_pdf_to_image(args.pdf_path, args.dpi, args.format.lower())
    print(f"\n🖼️ Image générée : {output_path}") 