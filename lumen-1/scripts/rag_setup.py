import json
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1. දත්ත load කිරීම
def load_data(json_path):
    print(f"Reading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    for place in data:
        name = place.get("name", "Unknown").strip()
        category = place.get("category_id", place.get("category", "Place")).strip()
        district = place.get("district_id", place.get("district", "Unknown")).strip()
        id_val = place.get("id", "Unknown").strip()
        desc = place.get("description", "No description available.").strip()
        activities = place.get("activities", "Sightseeing").strip()
        best_time = place.get("best_time_to_visit", "All year round").strip()
        ticket = place.get("ticket_price", place.get("ticket", "Free")).strip()
        safety = place.get("safety_level", "Safe").strip()
        parking = place.get("parking_avail", place.get("parking", "Unknown")).strip()
        
        text_content = f"{name} is a {category} located in {district}. " \
                       f"Description: {desc} " \
                       f"Activities: {activities}. " \
                       f"Best time to visit: {best_time}. " \
                       f"Ticket: {ticket}, " \
                       f"Parking: {parking}, " \
                       f"Safety: {safety}."
        
        doc = Document(
            page_content=text_content,
            metadata={"id": id_val, "name": name, "category": category}
        )
        documents.append(doc)
    return documents

# 2. Vector Database එක හැදීම
def setup_rag():
    # Resolve directory paths relative to script file (ගොනු පිහිටීම් නිවැරදිව හඳුනාගැනීමට)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(project_root, "data")
    
    # Try multiple database paths in order of preference (දත්ත ගොනුව සොයාගැනීම සඳහා විකල්ප මාර්ග කිහිපයක් පරීක්ෂා කරයි)
    candidate_paths = [
        os.path.join(data_dir, "tripme_database_complete_NEW.json"),
        os.path.join(data_dir, "tripme_database_complete.json"),
        os.path.join(data_dir, "tripme_database_augmented.json"),
        os.path.join(project_root, "tripme_database_complete_NEW.json"),
        os.path.join(project_root, "tripme_database_complete.json"),
    ]
    
    json_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            json_path = path
            break
            
    if not json_path:
        print(f"Error: tripme දත්ත ගොනුව සොයාගැනීමට නොහැකි විය. කරුණාකර එය 'data/' ෆෝල්ඩරයේ තිබේදැයි පරීක්ෂා කරන්න.")
        return

    docs = load_data(json_path)
    
    print("Creating Embeddings (This takes some time)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Storing in ChromaDB...")
    # Unify ChromaDB directory relative to project root (Dashboard එක සමඟ එකඟ වන පරිදි)
    chroma_dir = os.path.join(project_root, "chroma_db")
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings,
        persist_directory=chroma_dir
    )
    vectorstore.persist()
    print(f"✅ RAG Setup Complete! Vector Database එක සාර්ථකව '{chroma_dir}' හි save වුණා.")

if __name__ == "__main__":
    setup_rag()
