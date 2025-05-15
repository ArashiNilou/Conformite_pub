import re
from utils.logo_product_matcher import LogoProductMatcher

class ComplianceChecker:
    def __init__(self):
        self.logo_product_matcher = LogoProductMatcher()

    def check_rag_legislation(self, legislation_text, ad_text, products):
        """
        Analyse la législation extraite du RAG et détecte les non-conformités spécifiques enrichies.
        Args:
            legislation_text (str): Texte de la législation extraite (RAG)
            ad_text (str): Texte de la publicité à analyser
            products (list): Liste des produits détectés
        Returns:
            list: Liste de non-conformités détectées
        """
        non_conformities = []
        if not legislation_text or not ad_text:
            return non_conformities
        # 1. Mentions obligatoires globales
        mentions_obligatoires = re.findall(r"mention obligatoire ?: ([^\n\r]+)", legislation_text, re.IGNORECASE)
        for mention in mentions_obligatoires:
            if mention.lower() not in ad_text.lower():
                non_conformities.append(f"Mention légale obligatoire absente : '{mention}'")
        # 2. Mentions interdites globales
        mentions_interdites = re.findall(r"mention interdite ?: ([^\n\r]+)", legislation_text, re.IGNORECASE)
        for mention in mentions_interdites:
            if mention.lower() in ad_text.lower():
                non_conformities.append(f"Mention légale interdite présente : '{mention}'")
        # 3. Mentions obligatoires conditionnelles par produit
        cond_obligatoires = re.findall(r"mention obligatoire pour ([^:]+) ?: ([^\n\r]+)", legislation_text, re.IGNORECASE)
        for condition, mention in cond_obligatoires:
            for prod in products:
                if condition.lower() in prod.lower() and mention.lower() not in ad_text.lower():
                    non_conformities.append(f"Mention obligatoire pour '{condition}' absente alors que le produit '{prod}' est présent : '{mention}'")
        # 4. Mentions interdites conditionnelles par produit
        cond_interdites = re.findall(r"mention interdite pour ([^:]+) ?: ([^\n\r]+)", legislation_text, re.IGNORECASE)
        for condition, mention in cond_interdites:
            for prod in products:
                if condition.lower() in prod.lower() and mention.lower() in ad_text.lower():
                    non_conformities.append(f"Mention interdite pour '{condition}' présente alors que le produit '{prod}' est présent : '{mention}'")
        # 5. Mentions à formuler exactement
        exact_mentions = re.findall(r"mention exacte ?: ([^\n\r]+)", legislation_text, re.IGNORECASE)
        for mention in exact_mentions:
            if mention.lower() not in ad_text.lower():
                non_conformities.append(f"La mention obligatoire doit être formulée exactement ainsi : '{mention}'")
        return non_conformities

    def check(self, raw_text: str, products: list = None, logos: list = None, vision_result: str = None, legislation_result: str = None, **kwargs) -> dict:
        """
        Applique les règles de conformité sur le texte extrait et les produits/logos détectés.
        Args:
            raw_text: Texte brut extrait de l'image
            products: Liste des produits détectés (optionnel)
            logos: Liste des logos détectés (optionnel)
            vision_result: Résultat de l'analyse visuelle (optionnel)
            legislation_result: Résultat de la recherche de législation (optionnel)
        Returns:
            dict: Résultat de conformité (non-conformités, détails, etc.)
        """
        # Liste complète des mentions légales nutritionnelles PNNS à surveiller
        pnns_mentions = [
            "mangerbouger.fr",
            "pour votre santé, mangez au moins cinq fruits et légumes par jour",
            "pour votre santé, évitez de manger trop gras, trop sucré, trop salé",
            "pour votre santé, pratiquez une activité physique régulière",
            "pour votre santé, évitez de grignoter entre les repas",
            "pour votre santé, évitez de consommer trop de sel",
            "pour votre santé, limitez les produits sucrés",
            "pour votre santé, limitez les produits gras",
            "pour votre santé, limitez les produits salés",
            "pour votre santé, limitez la consommation d'alcool"
        ]
        # Texte à analyser pour la mention et les produits : raw_text si possible, sinon vision_result
        texte_a_analyser = raw_text if raw_text and not raw_text.startswith('ERREUR') else vision_result if vision_result else ""
        # Vérifier la présence d'au moins une mention PNNS dans le texte analysé
        mentions_pnns_trouvees = [m for m in pnns_mentions if m in texte_a_analyser.lower()]
        mention_pnns_presente = len(mentions_pnns_trouvees) > 0
        # Détection des produits transformés et non transformés
        produits_transformes = False
        tous_non_transformes = False
        produits_detectes = []
        if texte_a_analyser:
            produits_detectes = self.logo_product_matcher.extract_products_from_text(texte_a_analyser)
            if produits_detectes:
                tous_non_transformes = self.logo_product_matcher.is_non_transformed_product(produits_detectes)
                produits_transformes = not tous_non_transformes
        # Liste des non-conformités spécifiques à ajouter
        non_conformites = []
        if produits_transformes and not mention_pnns_presente:
            non_conformites.append(
                "Absence de mention légale nutritionnelle obligatoire (PNNS) alors que des produits transformés sont présents."
            )
        if tous_non_transformes and mention_pnns_presente:
            produits_str = ", ".join(produits_detectes) if produits_detectes else "(non détectés)"
            non_conformites.append(
                f"Non-conformité : la mention légale nutritionnelle (PNNS) est présente alors qu'aucun produit transformé n'est détecté (ex : {produits_str}). Cette mention ne doit pas figurer pour des produits non transformés."
            )
        # Vérification de la présence du numéro RCS et du site internet dans le texte analysé
        rcs_present = bool(re.search(r"\bRCS\b", texte_a_analyser, re.IGNORECASE))
        site_present = bool(re.search(r"\bhttps?://|www\.[a-z0-9\-]+\.[a-z]{2,}\b", texte_a_analyser, re.IGNORECASE))
        site_present = site_present and not re.search(r"www\.mangerbouger\.fr", texte_a_analyser, re.IGNORECASE)
        if not rcs_present:
            non_conformites.append("Absence de numéro RCS : la publicité doit comporter le numéro RCS de l'entreprise.")
        if not site_present:
            non_conformites.append("Absence de site internet de l'entreprise : aucun site internet spécifique à l'annonceur n'est mentionné.")
        # Intégration de la vérification avancée via RAG
        if legislation_result:
            rag_non_conformities = self.check_rag_legislation(legislation_result, texte_a_analyser, produits_detectes)
            non_conformites.extend(rag_non_conformities)
        return {
            "non_conformities": non_conformites,
            "products": produits_detectes,
            "mentions_pnns": mentions_pnns_trouvees
        } 