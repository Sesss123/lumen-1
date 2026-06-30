import os
import chromadb
from chromadb.utils import embedding_functions

# =====================================================================
# 1. RAG Knowledge Base (ශ්‍රී ලාංකික ආහාර පිළිබඳ දත්ත ගොනුව)
# =====================================================================
FOOD_KNOWLEDGE_BASE = [
    {
        "id": "food_001",
        "name": "Kottu Roti",
        "content": "Kottu Roti is a popular Sri Lankan street food made of chopped flatbread (godamba roti), vegetables, egg, meat, and rich aromatic spices. It originated in the 1970s in Batticaloa and Trincomalee.",
        "metadata": {"calories": 450, "type": "Street Food", "spiciness": "High"}
    },
    {
        "id": "food_002",
        "name": "Lamprais",
        "content": "Lamprais is a Dutch-Burgher delicacy consisting of boiled eggs, eggplant (brinjal) moju, frikkadel meatballs, mixed meat curry, and sambal, all baked slowly in a banana leaf.",
        "metadata": {"calories": 520, "type": "Traditional/Burgher", "spiciness": "Medium"}
    },
    {
        "id": "food_003",
        "name": "Jaffna Crab Curry",
        "content": "An iconic fiery dish from Northern Sri Lanka (Jaffna), known for its thick roasted curry powder, tamarind tanginess, and heavy use of fresh chilies and coconut milk.",
        "metadata": {"calories": 310, "type": "Seafood", "spiciness": "Very High"}
    },
    {
        "id": "food_004",
        "name": "Hoppers (Appa)",
        "content": "Hoppers are bowl-shaped pancakes made from fermented rice flour and coconut milk. Often served with a fiery Lunu Miris (onion sambol) and seeni sambol.",
        "metadata": {"calories": 120, "type": "Breakfast/Dinner", "spiciness": "Mild (Base)"}
    },
    {
        "id": "food_005",
        "name": "Ambul Thiyal (Sour Fish Curry)",
        "content": "A unique Southern Sri Lankan fish preservation dish originating from Ambalangoda. Tuna is coated in a thick, sour paste made of goraka (Malabar tamarind) and black pepper.",
        "metadata": {"calories": 250, "type": "Seafood/Preserved", "spiciness": "Medium-High"}
    },
    {
        "id": "food_006",
        "name": "String Hoppers (Idiyappam)",
        "content": "A traditional dish consisting of rice flour dough pressed into thin noodles and steamed. Typically paired with Kiri Hodi (coconut milk gravy) and Pol Sambol.",
        "metadata": {"calories": 200, "type": "Breakfast/Dinner", "spiciness": "Mild"}
    }
]

class FoodRAGSystem:
    def __init__(self, db_path="./chroma_db"):
        print("🚀 Initializing ChromaDB RAG Vector Store...")
        
        # Ensure the directory exists relative to the current working directory
        os.makedirs(db_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=db_path)
        
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name="srilankan_food_knowledge",
            embedding_function=self.embedding_fn
        )
        
        if self.collection.count() == 0:
            self._populate_database()

    def _populate_database(self):
        print("📥 Populating Vector Database with Food Knowledge...")
        ids = [item["id"] for item in FOOD_KNOWLEDGE_BASE]
        documents = [item["content"] for item in FOOD_KNOWLEDGE_BASE]
        metadatas = [item["metadata"] for item in FOOD_KNOWLEDGE_BASE]
        
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"✅ {len(ids)} food items successfully added to RAG Database!")

    def retrieve_knowledge(self, query: str, top_k: int = 1) -> str:
        """
        AI එකට අවශ්‍ය කෑම පිළිබඳ දත්ත සෙවීම.
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            if not results or not results.get('documents') or not results['documents'][0]:
                return "✨ An authentic Sri Lankan delicacy bursting with exotic island spices and fresh tropical ingredients."
                
            best_match = results['documents'][0][0]
            metadata = results['metadatas'][0][0] if results.get('metadatas') and results['metadatas'][0] else {}
            
            # Create a rich cultural string
            spiciness = metadata.get("spiciness", "Unknown")
            type_str = metadata.get("type", "Traditional")
            
            return f"📚 Cultural History: {best_match} | Type: {type_str} | Spice Level: {spiciness}"
            
        except Exception as e:
            print(f"RAG Error: {e}")
            return "✨ An authentic Sri Lankan delicacy bursting with exotic island spices and fresh tropical ingredients."
