import os
# Set environment variables to prevent multi-threaded deadlocks and offline hangs in PyTorch/OpenMP/MKL/HuggingFace
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_num_threads(1)

import chromadb
from groq import Groq
import threading
import time

# Caching local model instance to prevent reloading in Streamlit
_transformer_model = None

def get_transformer_model():
    global _transformer_model
    if _transformer_model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading local SentenceTransformer model 'all-MiniLM-L6-v2' in RAG engine...")
        _transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _transformer_model

# Caching ChromaDB client to prevent multiple clients locking the same sqlite file
_chroma_client = None

def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        db_path = os.path.join(os.getcwd(), "database")
        _chroma_client = chromadb.PersistentClient(path=db_path)
    return _chroma_client

def get_chroma_collection():
    """
    Connects to the local ChromaDB persistent instance and returns the law collection.
    """
    db_path = os.path.join(os.getcwd(), "database")
    if not os.path.exists(db_path):
        return None
        
    try:
        chroma_client = get_chroma_client()
        # Returns the collection if it exists
        return chroma_client.get_collection(name="indian_laws_comparison")
    except Exception as e:
        print(f"Error accessing ChromaDB: {e}")
        return None

def query_vector_store(query_text, api_key, law_filter=None, top_k=5):
    """
    Generates embedding locally for the query and retrieves the top_k relevant sections from ChromaDB.
    """
    print(f"[RAG Engine] query_vector_store called for query: {query_text[:60]}...")
    collection = get_chroma_collection()
    if collection is None or collection.count() == 0:
        print("[RAG Engine] Warning: Law collection in ChromaDB is empty or not found.")
        return []
        
    try:
        # Load local model and encode query locally
        print("[RAG Engine] Getting SentenceTransformer model instance...")
        model = get_transformer_model()
        print("[RAG Engine] Generating local query embedding...")
        query_embedding = model.encode(query_text).tolist()
        print("[RAG Engine] Query embedding generated successfully. Searching ChromaDB...")
        
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
        print(f"[RAG Engine] ChromaDB search completed. Retrieved {len(formatted_results)} sections.")
        return formatted_results
    except Exception as e:
        print(f"[RAG Engine] Error querying vector store: {e}")
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

# Thread lock to serialize all API requests to Groq across all sessions
_api_request_lock = threading.Lock()
# Timestamp of the end of the last API request
_last_api_request_time = 0.0

def generate_rag_response(query_text, context_docs, api_key, chat_history=[]):
    """
    Assembles contextual documents and history, feeds them into Groq API,
    and returns a structured comparative response.
    """
    global _last_api_request_time
    print("[RAG Engine] generate_rag_response called. Preparing request payload...")
    
    # Initialize the Groq Client with a 60-second timeout to prevent hangs
    client = Groq(
        api_key=api_key,
        timeout=60.0
    )
    
    # Prepare retrieved sections context
    context_str = ""
    for idx, doc in enumerate(context_docs):
        context_str += f"--- SOURCE DOCUMENT {idx+1} (Law: {doc['metadata']['law_type']}, Old: Sec {doc['metadata']['old_section']}, New: Sec {doc['metadata']['new_section']}) ---\n"
        context_str += doc['content'] + "\n\n"
        
    system_prompt = (
        "You are an expert Indian Legal Advisor specializing in the transition from the legacy colonial-era criminal laws "
        "(Indian Penal Code - IPC, Code of Criminal Procedure - CrPC, Indian Evidence Act - IEA) to the newly enacted laws of 2023 "
        "(Bharatiya Nyaya Sanhita - BNS, Bharatiya Nagarik Suraksha Sanhita - BNSS, Bharatiya Sakshya Adhiniyam - BSA) which came into effect on July 1, 2024.\n\n"
        "Your task is to analyze the user's question, utilize the provided SOURCE DOCUMENTS (which can contain comparative statutory mappings or direct Question-Answer pairs about specific sections of the new acts), "
        "and explain the differences, concepts, or procedures clearly, professionally, and accurately.\n\n"
        "Guidelines:\n"
        "1. Highlight exactly which section of the old law has been replaced by which section of the new law (if comparative data is present in the context).\n"
        "2. Detail the modifications, definitions, punishments, or procedural requirements as detailed in the source context (such as Zero-FIR, electronic recording, timelines, or custody changes).\n"
        "3. If a section has been repealed without replacement or is entirely new, point that out explicitly.\n"
        "4. Base your answer strictly on the provided context. If the source documents do not contain the answer, "
        "use your general legal knowledge of these acts to answer, but add a warning note that the specific sections were not found in the local database.\n"
        "5. Formatting & Layout Instructions:\n"
        "   - Organize the response using clear, structural Markdown headings (e.g. '### Introduction', '### Key Modifications', '### Comparison Details'). Do not write a single block of text.\n"
        "   - Ensure there are double newlines (blank lines) between paragraphs, headings, list items, and sections to ensure a clean layout.\n"
        "   - Use bold text for key terms, section numbers (e.g. **IPC Section 302**, **BNS Section 103**), and important parameters.\n"
        "   - Use HTML underlines (e.g. '<u>Transition Details</u>') or markdown horizontal rules ('---') to visually separate distinct parts of the comparison.\n"
        "   - Use lists, bullet points, or markdown tables to format multi-item details instead of long text blocks.\n"
        "6. Suggested Follow-up Questions:\n"
        "   - At the very end of your response, add a divider ('---') followed by a section titled '### Suggested Follow-up Questions'.\n"
        "   - Provide exactly 4-5 relevant, specific, and engaging follow-up questions that the user can ask next to explore the topics discussed in the response further."
    )
    
    # Format chat history for Groq API
    prompt = "Below is the context retrieved from the database:\n\n"
    prompt += context_str
    
    if chat_history:
        prompt += "Here is the conversation history:\n"
        for msg in chat_history[-6:]:  # include last 3 exchanges (6 messages)
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt += f"{role}: {msg['content']}\n"
            
    prompt += f"\nUser Question: {query_text}\n"
    prompt += "Provide a detailed comparison and response:"
    
    # Define fallback Groq models to try if the default model fails or times out
    models_to_try = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768"
    ]
    response_text = None
    last_exception = None
    
    # Use the lock to ensure only one API request runs at a time
    with _api_request_lock:
        for model_name in models_to_try:
            # Retry transient errors up to 2 times for each model
            max_retries = 2
            for attempt in range(max_retries + 1):
                # Maintain the 1.5s rate-limit interval
                now = time.time()
                elapsed = now - _last_api_request_time
                if elapsed < 1.5:
                    wait_time = 1.5 - elapsed
                    print(f"[RAG Engine] Waiting {wait_time:.2f} seconds before sending request to maintain 1.5s interval...")
                    time.sleep(wait_time)
                
                try:
                    print(f"[RAG Engine] Sending request to Groq LLM ({model_name}) [Attempt {attempt + 1}/{max_retries + 1}]...")
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        model=model_name,
                        temperature=0.2,
                    )
                    print(f"[RAG Engine] Groq API response received successfully from {model_name}.")
                    response_text = chat_completion.choices[0].message.content
                    # Update request completion time
                    _last_api_request_time = time.time()
                    break
                except Exception as e:
                    # Update request completion time
                    _last_api_request_time = time.time()
                    
                    err_str = str(e)
                    is_transient = "504" in err_str or "503" in err_str or "500" in err_str or "429" in err_str or "RATE_LIMIT" in err_str or "TIMEOUT" in err_str or "DEADLINE_EXCEEDED" in err_str or "TEMPORARY" in err_str
                    
                    print(f"[RAG Engine] Error during Groq API ({model_name}) execution: {e}")
                    last_exception = e
                    
                    if is_transient and attempt < max_retries:
                        backoff_delay = 2.0 ** (attempt + 1)
                        print(f"[RAG Engine] Transient error detected. Retrying {model_name} in {backoff_delay:.2f} seconds...")
                        time.sleep(backoff_delay)
                    else:
                        print(f"[RAG Engine] Non-transient error or maximum retries reached for {model_name}.")
                        break  # Move to the next model in models_to_try
            
            if response_text is not None:
                break
        
    if response_text is not None:
        return response_text
    else:
        return f"Error communicating with Groq LLM (tried {', '.join(models_to_try)}): {str(last_exception)}"
