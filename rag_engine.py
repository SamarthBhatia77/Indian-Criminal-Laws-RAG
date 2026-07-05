import os
import chromadb
from google import genai
from google.genai import types

def get_chroma_collection():
    """
    Connects to the local ChromaDB persistent instance and returns the law collection.
    """
    db_path = os.path.join(os.getcwd(), "database")
    if not os.path.exists(db_path):
        return None
        
    try:
        chroma_client = chromadb.PersistentClient(path=db_path)
        # Returns the collection if it exists
        return chroma_client.get_collection(name="indian_laws_comparison")
    except Exception as e:
        print(f"Error accessing ChromaDB: {e}")
        return None

def query_vector_store(query_text, api_key, law_filter=None, top_k=5):
    """
    Generates embedding for the query and retrieves the top_k relevant sections from ChromaDB.
    """
    collection = get_chroma_collection()
    if collection is None or collection.count() == 0:
        return []
        
    # Initialize the new Google Gen AI Client
    client = genai.Client(api_key=api_key)
    
    try:
        # Generate embedding for search query
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=query_text
        )
        query_embedding = response.embeddings[0].values
        
        # Prepare filter if selected
        where_filter = None
        if law_filter and law_filter != "All":
            where_filter = {"law_type": law_filter}
            
        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
        )
        
        # Format results
        formatted_results = []
        if results and 'documents' in results and results['documents']:
            docs = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            ids = results['ids'][0]
            
            for idx in range(len(docs)):
                formatted_results.append({
                    "id": ids[idx],
                    "content": docs[idx],
                    "metadata": metadatas[idx],
                    "similarity": 1.0 - distances[idx]  # Convert distance to similarity score
                })
        return formatted_results
    except Exception as e:
        print(f"Error querying vector store: {e}")
        return []

def direct_lookup(law_type, section_number):
    """
    Performs a direct section lookup in the vector store metadata without calling the LLM.
    Enables quick side-by-side comparison tables.
    """
    collection = get_chroma_collection()
    if collection is None:
        return None
        
    section_number = str(section_number).strip()
    
    try:
        # Search for exact match in metadata for old or new section number
        # We query by old_section
        results_old = collection.get(
            where={
                "$and": [
                    {"law_type": law_filter_map(law_type)},
                    {"old_section": section_number}
                ]
            }
        )
        
        if results_old and results_old['documents']:
            return parse_chroma_get_result(results_old, 0)
            
        # Search for exact match in new_section
        results_new = collection.get(
            where={
                "$and": [
                    {"law_type": law_filter_map(law_type)},
                    {"new_section": section_number}
                ]
            }
        )
        
        if results_new and results_new['documents']:
            return parse_chroma_get_result(results_new, 0)
            
        # Try soft match using query text search in metadata (if exact failed)
        # We look up sections that contain the section number
        all_docs = collection.get(where={"law_type": law_filter_map(law_type)})
        if all_docs and all_docs['metadatas']:
            for idx, meta in enumerate(all_docs['metadatas']):
                if section_number in str(meta.get("old_section", "")) or section_number in str(meta.get("new_section", "")):
                    return parse_chroma_get_result(all_docs, idx)
                    
        return None
    except Exception as e:
        print(f"Error in direct section lookup: {e}")
        return None

def law_filter_map(law_name):
    """Maps display names to metadata filters."""
    mapping = {
        "IPC to BNS (Substantive)": "IPC-BNS",
        "CrPC to BNSS (Procedural)": "CrPC-BNSS",
        "IEA to BSA (Evidence)": "IEA-BSA"
    }
    return mapping.get(law_name, law_name)

def parse_chroma_get_result(results, index):
    """Helper to parse chroma .get() result lists."""
    content = results['documents'][index]
    metadata = results['metadatas'][index]
    
    # Extract details by parsing content lines
    lines = content.split('\n')
    old_desc = ""
    new_desc = ""
    
    for line in lines:
        if line.startswith("Old Description:"):
            old_desc = line.replace("Old Description:", "").strip()
        elif line.startswith("New Description:"):
            new_desc = line.replace("New Description:", "").strip()
            
    return {
        "law_type": metadata.get("law_type"),
        "old_section": metadata.get("old_section"),
        "old_heading": metadata.get("old_heading"),
        "old_description": old_desc or "Refer to text details.",
        "new_section": metadata.get("new_section"),
        "new_heading": metadata.get("new_heading"),
        "new_description": new_desc or "Refer to text details.",
        "full_content": content
    }

def generate_rag_response(query_text, context_docs, api_key, chat_history=[]):
    """
    Assembles contextual documents and history, feeds them into Gemini API,
    and returns a structured comparative response.
    """
    # Initialize the new Google Gen AI Client
    client = genai.Client(api_key=api_key)
    
    # Prepare retrieved sections context
    context_str = ""
    for idx, doc in enumerate(context_docs):
        context_str += f"--- SOURCE DOCUMENT {idx+1} (Law: {doc['metadata']['law_type']}, Old: Sec {doc['metadata']['old_section']}, New: Sec {doc['metadata']['new_section']}) ---\n"
        context_str += doc['content'] + "\n\n"
        
    system_prompt = (
        "You are an expert Indian Legal Advisor specializing in the transition from the legacy colonial-era criminal laws "
        "(Indian Penal Code - IPC, Code of Criminal Procedure - CrPC, Indian Evidence Act - IEA) to the newly enacted laws of 2023 "
        "(Bharatiya Nyaya Sanhita - BNS, Bharatiya Nagarik Suraksha Sanhita - BNSS, Bharatiya Sakshya Adhiniyam - BSA) which came into effect on July 1, 2024.\n\n"
        "Your task is to analyze the user's question, utilize the provided SOURCE DOCUMENTS containing corresponding sections, "
        "and explain the differences clearly, professionally, and accurately.\n\n"
        "Guidelines:\n"
        "1. Highlight exactly which section of the old law has been replaced by which section of the new law.\n"
        "2. Detail the modifications (e.g. increase/decrease in punishment, changes in definitions, new procedural mandates like electronic recording, virtual trials, or timelines).\n"
        "3. If a section has been repealed without replacement or is entirely new, point that out explicitly.\n"
        "4. Base your answer strictly on the provided context. If the source documents do not contain the answer, "
        "use your general legal knowledge of these acts to answer, but add a warning note that the specific sections were not found in the local database.\n"
        "5. Keep the tone professional, structural, and objective. Use Markdown tables, bullet points, and bold text for high readability."
    )
    
    # Format chat history for Gemini API
    prompt = "Below is the context retrieved from the database:\n\n"
    prompt += context_str
    
    if chat_history:
        prompt += "Here is the conversation history:\n"
        for msg in chat_history[-6:]:  # include last 3 exchanges (6 messages)
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt += f"{role}: {msg['content']}\n"
            
    prompt += f"\nUser Question: {query_text}\n"
    prompt += "Provide a detailed comparison and response:"
    
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2
            )
        )
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini LLM: {str(e)}"
