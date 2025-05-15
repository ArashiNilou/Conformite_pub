from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from config.azure_config import AzureConfig

class AIModels:
    """Initialisation des modèles AI"""
    def __init__(self, config: AzureConfig):
        self.embedding_model = AzureOpenAIEmbedding(
            engine="text-embedding-3-large",
            model="text-embedding-3-large",
            api_key=config.API_KEY,
            azure_endpoint=config.ENDPOINT,
            api_version=config.API_VERSION,
        )
        
        self.llm = AzureOpenAI(
            azure_endpoint=config.ENDPOINT,
            engine="gpt4o",
            api_version=config.API_VERSION,
            model="gpt-4o",
            api_key=config.API_KEY,
            supports_content_blocks=True
        ) 

    def extract_raw_text_with_vision(self, image_path: str) -> str:
        """
        Extrait le texte brut d'une image en utilisant GPT Vision
        
        Args:
            image_path: Chemin de l'image
            
        Returns:
            str: Texte brut extrait
        """
        print(f"\n🔍 Extraction de texte brut avec GPT Vision: {image_path}")
        
        # Convertir l'image
        image_data = self._prepare_image(image_path)
        
        try:
            # Construire le prompt
            system_prompt = """Extrait TOUT le texte visible dans l'image avec une extrême précision.
N'ajoutez aucune interprétation, correction ou explication. Structurez clairement avec ces sections:

**TEXTE PRINCIPAL :**
[Tout le texte principal, grands titres, descriptions de produits, prix]

**PETITS CARACTÈRES :**
[Tout texte en petits caractères comme mentions légales, conditions]

**RENVOIS D'ASTÉRISQUES :**
[Pour chaque astérisque, indiquez "[Astérisque #X sur le mot 'Y'] correspond à [texte associé]"]

**MOTS POTENTIELLEMENT MAL ORTHOGRAPHIÉS :**
[Listez les mots qui semblent mal orthographiés]

**TEXTE DIFFICILEMENT LISIBLE :**
[Mentionnez les zones où le texte est difficile à lire]

Respectez EXACTEMENT la typographie originale (majuscules, minuscules, mise en forme).
Préservez fidèlement la structure et les sauts de ligne."""
            
            # Modèle: gpt-4-vision-preview
            model = self.gpt4_vision
            
            # Token counter
            with self.token_context.with_current_step("raw_text"):
                # Construire le message avec l'image
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "text", 
                                "text": "Extrais tout le texte visible dans cette image avec précision, sans aucune interprétation."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ]
                
                # Appel à l'API
                resp = self.gpt4o.chat.completions.create(
                    messages=messages,
                    temperature=0.0,
                    max_tokens=2048
                )
                
                # Tracer l'utilisation des tokens
                self.token_context.add_prompt_tokens(resp.usage.prompt_tokens)
                self.token_context.add_completion_tokens(resp.usage.completion_tokens)
                
                # Extraire le texte
                raw_text = resp.choices[0].message.content.strip()
                
                print("✅ Extraction de texte brut réussie")
                
                return raw_text
                
        except Exception as e:
            print(f"❌ Erreur lors de l'extraction de texte brut: {str(e)}")
            return f"ERREUR: {str(e)}" 