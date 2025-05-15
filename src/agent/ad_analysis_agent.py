from extractors.text_extractor import TextExtractor
from analyzers.compliance_checker import ComplianceChecker
from utils.logo_product_matcher import LogoProductMatcher
from models.ai_models import AIModels
from raptor.raptor_setup import RaptorSetup
from config.azure_config import AzureConfig

class AdAnalysisAgent:
    def __init__(self, text_extractor=None, compliance_checker=None, logo_product_matcher=None, ai_models=None, raptor_setup=None):
        self.text_extractor = text_extractor or TextExtractor()
        self.compliance_checker = compliance_checker or ComplianceChecker()
        self.logo_product_matcher = logo_product_matcher or LogoProductMatcher()
        self.ai_models = ai_models or AIModels(AzureConfig())
        self.raptor_setup = raptor_setup or RaptorSetup(self.ai_models)

    def analyze(self, image_path: str) -> dict:
        raw_text = self.text_extractor.extract(image_path)
        products = self.logo_product_matcher.extract_products_from_text(raw_text)
        logos = self.logo_product_matcher.extract_logos_from_text(raw_text)
        # Recherche automatique de la législation via RAG
        legislation_result = self.raptor_setup.search(raw_text)
        compliance = self.compliance_checker.check(raw_text, products=products, logos=logos, legislation_result=legislation_result)
        return {
            "raw_text": raw_text,
            "products": products,
            "logos": logos,
            "legislation_result": legislation_result,
            "compliance": compliance
        } 