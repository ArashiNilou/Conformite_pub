from ..utils.text_extractor import TextExtractor as BaseTextExtractor

class TextExtractor:
    def __init__(self):
        self.base_extractor = BaseTextExtractor()

    def extract(self, image_path: str) -> str:
        """
        Extrait le texte brut d'une image publicitaire (OCR multi-moteur, fallback inclus).
        Args:
            image_path: Chemin vers l'image à analyser
        Returns:
            str: Texte extrait
        """
        return self.base_extractor.extract_text(image_path, fallback=True) 