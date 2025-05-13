import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.packs.raptor import RaptorRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from models.ai_models import AIModels
from prompts.prompts import search_query
from time import sleep
from tenacity import retry, stop_after_attempt, wait_exponential
import fitz
from typing import Dict, Any

class RaptorSetup:
    """Configuration et initialisation de Raptor"""
    def __init__(self, ai_models: AIModels):
        print("\n🔧 Initialisation de ChromaDB...")
        try:
            # Initialisation de la base de données
            self.client = chromadb.PersistentClient(path="./RAPTOR_db")
            print("✅ Client ChromaDB créé avec succès")
            
            # Vérifier si la collection existe déjà
            collection_names = self.client.list_collections()
            print(f"📚 Collections existantes : {collection_names}")
            
            # Vérifier si notre collection existe
            if "legislation_PUB" in [c.name for c in collection_names]:
                print("📚 Collection 'legislation_PUB' trouvée")
                self.collection = self.client.get_collection("legislation_PUB")
            else:
                print("📚 Création de la collection 'legislation_PUB'")
                self.collection = self.client.create_collection("legislation_PUB")
            
            print(f"✅ Collection 'legislation_PUB' initialisée - Nombre d'éléments : {self.collection.count()}")
            
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
            print("✅ Vector store initialisé")
            
            # Initialisation du retriever avec un cache
            print("\n🔄 Configuration du retriever Raptor...")
            self.retriever = RaptorRetriever(
                [],
                embed_model=ai_models.embedding_model,
                llm=ai_models.llm,
                vector_store=self.vector_store,
                similarity_top_k=5,
                mode="collapsed",
                verbose=True
            )
            print("✅ Retriever configuré")
            
            # Cache pour les résultats de recherche
            self._search_cache = {}
            
            # Initialisation du query engine
            print("\n🔄 Configuration du query engine...")
            self.query_engine = RetrieverQueryEngine.from_args(
                self.retriever,
                llm=ai_models.llm
            )
            print("✅ Query engine configuré")
            
        except Exception as e:
            print(f"\n❌ Erreur lors de l'initialisation de ChromaDB : {str(e)}")
            print("📝 Détails de l'erreur :")
            import traceback
            print(traceback.format_exc())
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def search(self, query: str) -> str:
        """
        Recherche la législation applicable dans la base de connaissances
        
        Args:
            query: Contexte de la recherche
            
        Returns:
            str: Textes de loi trouvés
        """
        # Vérifier le cache
        if query in self._search_cache:
            print("\n📚 Utilisation du cache pour la recherche...")
            return self._search_cache[query]
        
        print(f"\n📚 Recherche de législation pour: {query[:200]}...")
        
        # Construire la requête de recherche
        formatted_query = search_query.format(query=query)
        print(f"\nRequête formatée: {formatted_query[:200]}...")
        
        try:
            print("\n🔍 Début de la recherche dans ChromaDB...")
            print(f"📊 Nombre d'éléments dans la collection : {self.collection.count()}")
            
            # Récupérer les documents pertinents avec retry
            print("🔄 Exécution de la requête via le retriever...")
            results = self.retriever.retrieve(formatted_query)
            print(f"✅ Requête exécutée - Nombre de résultats : {len(results)}")
            
            # Extraire le texte des résultats
            text_results = []
            for i, node in enumerate(results, 1):
                print(f"\n📄 Traitement du résultat {i}/{len(results)}")
                if hasattr(node, 'text'):
                    text_results.append(node.text)
                    print(f"✅ Texte extrait (longueur: {len(node.text)} caractères)")
                elif hasattr(node, 'content'):
                    text_results.append(node.content)
                    print(f"✅ Contenu extrait (longueur: {len(node.content)} caractères)")
            
            result_text = "\n".join(text_results) if text_results else "Aucune législation trouvée."
            print(f"\n📝 Résultat final - Longueur totale : {len(result_text)} caractères")
            
            # Mettre en cache le résultat
            self._search_cache[query] = result_text
            print("✅ Résultat mis en cache")
            
            # Attendre entre les requêtes
            sleep(2)
            
            return result_text
            
        except Exception as e:
            print(f"\n❌ Erreur lors de la recherche : {str(e)}")
            print("📝 Détails de l'erreur :")
            import traceback
            print(traceback.format_exc())
            raise

    def query(self, query_text: str) -> str:
        """
        Exécute une requête via le query engine
        
        Args:
            query_text: La question à poser
            
        Returns:
            str: La réponse générée
        """
        print(f"\n📚 Exécution de la requête Raptor: {query_text[:200]}...")
        try:
            # Utiliser directement les résultats de la recherche précédente si disponible
            if hasattr(self, '_last_search_results'):
                print("💡 Utilisation des résultats de recherche précédents")
                response = self.llm.complete(query_text + "\n\nContexte:\n" + self._last_search_results)
                return str(response)
            
            # Sinon, utiliser le query engine
            print("🔄 Utilisation du query engine...")
            response = self.query_engine.query(query_text)
            print("✅ Réponse générée")
            return str(response)
            
        except Exception as e:
            print(f"\n❌ Erreur lors de la requête Raptor: {str(e)}")
            print("📝 Détails de l'erreur :")
            import traceback
            print(traceback.format_exc())
            raise

    def extract_pdf_content(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extrait le contenu d'un PDF pour l'agent React
        
        Args:
            pdf_path: Chemin vers le PDF
            
        Returns:
            Dict[str, Any]: Contenu structuré du PDF
        """
        try:
            # Ouvrir le PDF
            pdf = fitz.open(pdf_path)
            
            # Extraire le contenu
            content = {
                "type": "pdf",
                "pages": [],
                "metadata": {
                    "total_pages": len(pdf),
                    "filename": pdf_path
                }
            }
            
            # Extraire chaque page
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                
                # Extraire texte et images
                text = page.get_text()
                
                # Ajouter la page au contenu
                content["pages"].append({
                    "page_number": page_num + 1,
                    "content": text
                })
            
            print(f"📄 PDF extrait : {len(content['pages'])} pages")
            return content
            
        except Exception as e:
            print(f"\n❌ Erreur lors de l'extraction du PDF : {str(e)}")
            raise
        finally:
            if 'pdf' in locals():
                pdf.close() 