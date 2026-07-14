import os
# Set environment variables to prevent multi-threaded deadlocks and offline hangs in PyTorch/OpenMP/MKL/HuggingFace
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_num_threads(1)

import pandas as pd
import json
import chromadb
from datasets import load_dataset

# Fallback data for IPC to BNS (in case HF datasets load fails)
IPC_BNS_FALLBACK = [
    {
        "IPC Section": "302",
        "IPC Heading": "Punishment for murder",
        "IPC Descriptions": "Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.",
        "BNS Section": "103(1)",
        "BNS Heading": "Punishment for murder",
        "BNS description": "Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine. Note: Organized crime and mob lynching provisions are added in BNS."
    },
    {
        "IPC Section": "307",
        "IPC Heading": "Attempt to murder",
        "IPC Descriptions": "Whoever does any act with such intention or knowledge, and under such circumstances that, if he by that act caused murder, he would be guilty of murder, shall be punished with imprisonment of either description for a term which may extend to ten years, and shall also be liable to fine.",
        "BNS Section": "109",
        "BNS Heading": "Attempt to murder",
        "BNS description": "Whoever does any act with such intention or knowledge, and under such circumstances that, if he by that act caused murder, he would be guilty of murder, shall be punished with imprisonment of either description for a term which may extend to ten years, and shall also be liable to fine."
    },
    {
        "IPC Section": "124A",
        "IPC Heading": "Sedition",
        "IPC Descriptions": "Whoever by words, either spoken or written, or by signs, or by visible representation, or otherwise, brings or attempts to bring into hatred or contempt, or excites or attempts to excite disaffection towards the Government established by law in India, shall be punished with imprisonment for life, to which fine may be added, or with imprisonment which may extend to three years, to which fine may be added, or with fine.",
        "BNS Section": "152",
        "BNS Heading": "Act endangering sovereignty, unity and integrity of India",
        "BNS description": "Whoever, purposely or knowingly, by words, either spoken or written, or by signs, or by visible representation, or by electronic communication or by use of financial means, or otherwise, excites or attempts to excite, secession or armed rebellion or subversive activities, or encourages feelings of separatist activities or endangers sovereignty or unity and integrity of India; or commits or detests any such act shall be punished with imprisonment for life or with imprisonment which may extend to seven years, and shall also be liable to fine."
    },
    {
        "IPC Section": "375 & 376",
        "IPC Heading": "Rape and punishment for rape",
        "IPC Descriptions": "Defines rape and provides punishment of rigorous imprisonment for a term which shall not be less than ten years, but which may extend to imprisonment for life, and shall also be liable to fine.",
        "BNS Section": "63 & 64",
        "BNS Heading": "Rape and punishment for rape",
        "BNS description": "The provisions are updated. Re-categorized under Chapter V 'Of Offences Against Women and Children'. Minimum punishment for rape remains 10 years extending to life. New offences like sexual intercourse by deceitful means (promising marriage under false pretenses) are added under Section 69."
    },
    {
        "IPC Section": "378 & 379",
        "IPC Heading": "Theft and punishment for theft",
        "IPC Descriptions": "Whoever, intending to take dishonestly any moveable property out of the possession of any person without that person's consent, moves that property in order to such taking, is said to commit theft. Punished with imprisonment for a term which may extend to three years, or with fine, or with both.",
        "BNS Section": "303(1) & 303(2)",
        "BNS Heading": "Theft and punishment for theft",
        "BNS description": "Definition remains similar. The punishment is imprisonment up to three years, or with fine, or with both. For subsequent convictions, severe punishments apply. Snatching is now explicitly defined as a separate offence under Section 304."
    },
    {
        "IPC Section": "420",
        "IPC Heading": "Cheating and dishonestly inducing delivery of property",
        "IPC Descriptions": "Whoever cheats and thereby dishonestly induces the person deceived to deliver any property to any person, or to make, alter or destroy the whole or any part of a valuable security, shall be punished with imprisonment of either description for a term which may extend to seven years, and shall also be liable to fine.",
        "BNS Section": "318(4)",
        "BNS Heading": "Cheating",
        "BNS description": "Cheating and dishonestly inducing delivery of property is renumbered to Section 318(4). The core elements remain the same, but it is grouped under offences against property."
    },
    {
        "IPC Section": "498A",
        "IPC Heading": "Husband or relative of husband of a woman subjecting her to cruelty",
        "IPC Descriptions": "Whoever, being the husband or the relative of the husband of a woman, subjects such woman to cruelty shall be punished with imprisonment for a term which may extend to three years and shall also be liable to fine.",
        "BNS Section": "85 & 86",
        "BNS Heading": "Cruelty by husband or relatives",
        "BNS description": "Renumbered to Section 85. Section 86 defines 'cruelty' explicitly, aligning with the judicial definitions of mental and physical harm."
    },
    {
        "IPC Section": "34",
        "IPC Heading": "Acts done by several persons in furtherance of common intention",
        "IPC Descriptions": "When a criminal act is done by several persons in furtherance of the common intention of all, each of such persons is liable for that act in the same manner as if it were done by him alone.",
        "BNS Section": "3(5)",
        "BNS Heading": "Common intention",
        "BNS description": "When a criminal act is done by several persons in furtherance of the common intention of all, each of such persons is liable for that act in the same manner as if it were done by him alone. Renumbered from Section 34 to Section 3(5)."
    },
    {
        "IPC Section": "120A & 120B",
        "IPC Heading": "Definition and punishment of criminal conspiracy",
        "IPC Descriptions": "Defines criminal conspiracy and outlines punishments depending on the severity of the offence conspired.",
        "BNS Section": "61(1) & 61(2)",
        "BNS Heading": "Criminal conspiracy",
        "BNS description": "Renumbered to Section 61. The definition and punishment structure remain identical to the legacy provisions."
    },
    {
        "IPC Section": "304B",
        "IPC Heading": "Dowry death",
        "IPC Descriptions": "Where the death of a woman is caused by any burns or bodily injury or occurs otherwise than under normal circumstances within seven years of her marriage and it is shown that soon before her death she was subjected to cruelty or harassment by her husband or any relative of her husband for, or in connection with, any demand for dowry, such death shall be called 'dowry death', and such husband or relative shall be deemed to have caused her death. Punished with imprisonment for a term which shall not be less than seven years but which may extend to imprisonment for life.",
        "BNS Section": "80",
        "BNS Heading": "Dowry death",
        "BNS description": "Renumbered to Section 80. The essential conditions, timelines (seven years of marriage), and punishment parameters remain unchanged."
    }
]

# Curated dataset for CrPC to BNSS
CRPC_BNSS_DATA = [
    {
        "Old Section": "438",
        "Old Heading": "Anticipatory Bail",
        "Old Description": "Direction for grant of bail to person apprehending arrest. High Court or Court of Session may direct that in the event of such arrest, the person shall be released on bail.",
        "New Section": "482",
        "New Heading": "Anticipatory Bail",
        "New Description": "Anticipatory bail provision is preserved. However, the new law removes certain subjective clauses and clarifies that the court may impose conditions like restricting overseas travel or requiring cooperation with police investigations.",
        "Law Type": "CrPC-BNSS"
    },
    {
        "Old Section": "154",
        "Old Heading": "Information in cognizable cases (FIR)",
        "Old Description": "Every information relating to the commission of a cognizable offence, if given orally to an officer in charge of a police station, shall be reduced to writing by him or under his direction.",
        "New Section": "173",
        "New Heading": "Information in cognizable cases (FIR, Zero FIR, e-FIR)",
        "New Description": "Introduces the concept of 'Zero FIR' enabling registration of an FIR at any police station irrespective of jurisdiction. It also formalizes 'e-FIR', allowing information to be sent electronically, which must be signed by the informant within 3 days for it to be registered.",
        "Law Type": "CrPC-BNSS"
    },
    {
        "Old Section": "167",
        "Old Heading": "Procedure when investigation cannot be completed in twenty-four hours (Custody)",
        "Old Description": "Authorizes detention of the accused in police custody for a maximum period of 15 days. Judicial custody can extend to 60 or 90 days total.",
        "New Section": "187",
        "New Heading": "Detention during investigation",
        "New Description": "Allows the 15-day police custody to be sought in parts/installments throughout the first 40 days (for offences carrying up to 10 years imprisonment) or 60 days (for offences carrying more than 10 years or life/death). This is a major departure from the old rule where police custody was only allowed in the first 15 days of arrest.",
        "Law Type": "CrPC-BNSS"
    },
    {
        "Old Section": "260",
        "Old Heading": "Power to try summarily",
        "Old Description": "Magistrates have discretionary power to try certain petty offences summarily (imprisonment up to 2 years).",
        "New Section": "283",
        "New Heading": "Summary trials",
        "New Description": "Makes summary trials mandatory for petty offences, including theft, receiving stolen property up to Rs. 20,000, trespass, and house trespass, to reduce backlog in courts.",
        "Law Type": "CrPC-BNSS"
    },
    {
        "Old Section": "New Provision",
        "Old Heading": "No equivalent (Physical presence required)",
        "Old Description": "Generally required physical presence of the accused, witnesses, and evidence in court. Trials and recording of statements had to be physically done in person.",
        "New Section": "530",
        "New Heading": "Use of electronic communication and audio-video electronic means",
        "New Description": "Allows all trials, inquiries, and proceedings under the Code to be held in electronic mode. Statements of witnesses, testimonies of experts, and even trials of the accused can be conducted via video conferencing.",
        "Law Type": "CrPC-BNSS"
    },
    {
        "Old Section": "173(8)",
        "Old Heading": "Further Investigation after filing charge sheet",
        "Old Description": "Allows police to conduct further investigation and file supplementary charge sheets, without a specific time frame.",
        "New Section": "193(9)",
        "New Heading": "Time limit for further investigation",
        "New Description": "Retains the power of further investigation, but mandates that such further investigation must be completed within a period of 90 days, which can be extended only with the permission of the Court.",
        "Law Type": "CrPC-BNSS"
    }
]

# Curated dataset for IEA to BSA
IEA_BSA_DATA = [
    {
        "Old Section": "65B",
        "Old Heading": "Admissibility of electronic records",
        "Old Description": "Requires a specific signed paper certificate (under 65B(4)) to admit electronic records as secondary evidence in court.",
        "New Section": "63",
        "New Heading": "Admissibility of electronic records (Updated Certificate)",
        "New Description": "Updates the admissibility criteria for electronic records. It provides a structured format in the Schedule (Schedule 1 and 2) for the certificate. It also expands the definition of electronic records to include server logs, cloud data, smartphones, and messages.",
        "Law Type": "IEA-BSA"
    },
    {
        "Old Section": "62",
        "Old Heading": "Primary Evidence",
        "Old Description": "Primary evidence means the document itself produced for the inspection of the Court.",
        "New Section": "57",
        "New Heading": "Primary Evidence (Expanded Electronic Records)",
        "New Description": "Explicitly includes electronic records under primary evidence. If an electronic record is created, stored, or recorded simultaneously in multiple devices, each such device is primary evidence. It also includes cloud backups and system records.",
        "Law Type": "IEA-BSA"
    },
    {
        "Old Section": "63",
        "Old Heading": "Secondary Evidence",
        "Old Description": "Defines secondary evidence as certified copies, copies made from the original, counterparts, or oral accounts.",
        "New Section": "58",
        "New Heading": "Secondary Evidence (Expanded scope)",
        "New Description": "Expands the list of secondary evidence. Includes oral admissions, written admissions, and evidence of a person who has examined a document which cannot easily be examined in court.",
        "Law Type": "IEA-BSA"
    },
    {
        "Old Section": "133",
        "Old Heading": "Accomplice",
        "Old Description": "An accomplice shall be a competent witness against an accused person; and a conviction is not illegal merely because it proceeds upon the uncorroborated testimony of an accomplice.",
        "New Section": "138",
        "New Heading": "Accomplice testimony",
        "New Description": "Retains the competency of accomplice testimony but is renumbered to Section 138. BSA emphasizes corroboration for accomplices in severe offences.",
        "Law Type": "IEA-BSA"
    }
]

# Caching local model instance to prevent reloading in Streamlit
_transformer_model = None

def get_transformer_model():
    global _transformer_model
    if _transformer_model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading local SentenceTransformer model 'all-MiniLM-L6-v2'...")
        _transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _transformer_model

def load_ipc_bns_dataset():
    """
    Attempts to download IPC to BNS dataset from Hugging Face.
    Falls back to cached local dataset if loading fails.
    """
    print("Attempting to load IPC-BNS mapping from Hugging Face...")
    try:
        # Load dataset from Hugging Face
        dataset = load_dataset("nandhakumarg/IPC_and_BNS_transformation", split="train")
        df = dataset.to_pandas()
        print(f"Successfully loaded {len(df)} rows from Hugging Face.")
        return df
    except Exception as e:
        print(f"Failed to load dataset from HF: {e}")
        print("Falling back to local curated dataset.")
        # Map list of dicts to DataFrame with correct column names
        df_fallback = pd.DataFrame(IPC_BNS_FALLBACK)
        return df_fallback

def load_legal_qa_dataset():
    """
    Loads the bns_bnss_bsa_combined_legal_qa.jsonl file from the database folder.
    Returns a list of structured document chunks.
    """
    qa_path = os.path.join(os.getcwd(), "database", "bns_bnss_bsa_combined_legal_qa.jsonl")
    if not os.path.exists(qa_path):
        print(f"QA dataset not found at {qa_path}. Skipping.")
        return []
        
    print(f"Loading QA dataset from {qa_path}...")
    chunks = []
    try:
        with open(qa_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                item = json.loads(line)
                
                act = item.get("act", "")
                sec_num = item.get("section_number", "")
                sec_title = item.get("section_title", "")
                question = item.get("question", "")
                answer = item.get("answer", "")
                q_type = item.get("question_type", "")
                
                # Map act names to existing categories with distinct QA tag
                if "BNS" in act:
                    law_type = "IPC-BNS-QA"
                elif "BNSS" in act:
                    law_type = "CrPC-BNSS-QA"
                elif "BSA" in act:
                    law_type = "IEA-BSA-QA"
                else:
                    law_type = "General-QA"
                    
                content = (
                    f"Act: {act}\n"
                    f"Section: {sec_num} - {sec_title}\n"
                    f"Question: {question}\n"
                    f"Answer: {answer}\n"
                )
                
                metadata = {
                    "law_type": law_type,
                    "old_section": "",
                    "old_heading": "",
                    "new_section": sec_num,
                    "new_heading": sec_title,
                    "question_type": q_type,
                    "is_qa": True
                }
                
                chunks.append({
                    "id": f"qa_{act.replace(' ', '_').lower()}_{sec_num}_{idx}",
                    "content": content,
                    "metadata": metadata
                })
        print(f"Successfully loaded {len(chunks)} Q&A pairs from QA dataset.")
        return chunks
    except Exception as e:
        print(f"Error reading QA dataset: {e}")
        return []

def create_unified_chunks(ipc_bns_df):
    """
    Combines IPC-BNS, CrPC-BNSS, and IEA-BSA mapping datasets into a single list
    of text chunks with metadata for indexing.
    """
    chunks = []
    
    # Strip spaces from columns and map them case-insensitively for flexible loading
    ipc_bns_df.columns = [str(c).strip() for c in ipc_bns_df.columns]
    columns_lower = {c.lower(): c for c in ipc_bns_df.columns}
    
    # Helper to get column value case-insensitively and handle partial column mismatches
    def get_val(row, key_name):
        key_lower = key_name.lower()
        if key_lower in columns_lower:
            real_key = columns_lower[key_lower]
            val = row[real_key]
            return "" if pd.isna(val) else str(val).strip()
        # Try finding key as substring (e.g. "IPC Descriptions" vs "IPC Description")
        for k in columns_lower:
            if key_lower in k or k in key_lower:
                real_key = columns_lower[k]
                val = row[real_key]
                return "" if pd.isna(val) else str(val).strip()
        return ""
    
    # 1. Process IPC-BNS Data
    for idx, row in ipc_bns_df.iterrows():
        ipc_sec = get_val(row, "IPC Section")
        ipc_head = get_val(row, "IPC Heading")
        ipc_desc = get_val(row, "IPC Descriptions")
        bns_sec = get_val(row, "BNS Section")
        bns_head = get_val(row, "BNS Heading")
        bns_desc = get_val(row, "BNS description")
        
        # Clean formatting
        if not ipc_sec:
            continue
            
        content = (
            f"Law Type: IPC to BNS (Substantive Criminal Law)\n"
            f"Old Provision: Section {ipc_sec} - {ipc_head}\n"
            f"Old Description: {ipc_desc}\n"
            f"New Provision: Section {bns_sec} - {bns_head}\n"
            f"New Description: {bns_desc}\n"
        )
        
        metadata = {
            "law_type": "IPC-BNS",
            "old_section": ipc_sec,
            "old_heading": ipc_head,
            "new_section": bns_sec,
            "new_heading": bns_head,
            "is_qa": False
        }
        
        chunks.append({
            "id": f"ipc_bns_{ipc_sec}_{idx}",
            "content": content,
            "metadata": metadata
        })

    # 2. Process CrPC-BNSS Data
    for idx, item in enumerate(CRPC_BNSS_DATA):
        content = (
            f"Law Type: CrPC to BNSS (Procedural Criminal Law)\n"
            f"Old Provision: Section {item['Old Section']} - {item['Old Heading']}\n"
            f"Old Description: {item['Old Description']}\n"
            f"New Provision: Section {item['New Section']} - {item['New Heading']}\n"
            f"New Description: {item['New Description']}\n"
        )
        metadata = {
            "law_type": "CrPC-BNSS",
            "old_section": item["Old Section"],
            "old_heading": item["Old Heading"],
            "new_section": item["New Section"],
            "new_heading": item["New Heading"],
            "is_qa": False
        }
        chunks.append({
            "id": f"crpc_bnss_{item['Old Section']}_{idx}",
            "content": content,
            "metadata": metadata
        })

    # 3. Process IEA-BSA Data
    for idx, item in enumerate(IEA_BSA_DATA):
        content = (
            f"Law Type: IEA to BSA (Evidence Criminal Law)\n"
            f"Old Provision: Section {item['Old Section']} - {item['Old Heading']}\n"
            f"Old Description: {item['Old Description']}\n"
            f"New Provision: Section {item['New Section']} - {item['New Heading']}\n"
            f"New Description: {item['New Description']}\n"
        )
        metadata = {
            "law_type": "IEA-BSA",
            "old_section": item["Old Section"],
            "old_heading": item["Old Heading"],
            "new_section": item["New Section"],
            "new_heading": item["New Heading"],
            "is_qa": False
        }
        chunks.append({
            "id": f"iea_bsa_{item['Old Section']}_{idx}",
            "content": content,
            "metadata": metadata
        })
        
    return chunks

def build_vector_store(api_key, progress_callback=None):
    """
    Initializes ChromaDB, fetches data, creates embeddings locally using
    sentence-transformers, and inserts documents.
    """
    # Load and process mapping data
    ipc_bns_df = load_ipc_bns_dataset()
    chunks = create_unified_chunks(ipc_bns_df)
    
    # Load QA dataset
    qa_chunks = load_legal_qa_dataset()
    all_chunks = chunks + qa_chunks
    
    print(f"Total structured sections loaded for indexing: {len(all_chunks)}")
    if progress_callback:
        progress_callback(0.1, f"Loaded {len(all_chunks)} sections. Initializing local SentenceTransformer...")
        
    # Lazy load sentence-transformers model
    model = get_transformer_model()
    
    # Setup ChromaDB client
    db_path = os.path.join(os.getcwd(), "database")
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    # Get or create collection
    collection = chroma_client.get_or_create_collection(
        name="indian_laws_comparison",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Check existing documents count
    existing_count = collection.count()
    print(f"Existing document count in collection: {existing_count}")
    
    # We clear and re-index to ensure correct schema, dimensions and complete mapping
    if existing_count > 0:
        print("Clearing existing vector storage to rebuild index...")
        chroma_client.delete_collection("indian_laws_comparison")
        collection = chroma_client.get_or_create_collection(
            name="indian_laws_comparison",
            metadata={"hnsw:space": "cosine"}
        )
    
    total_docs = len(all_chunks)
    batch_size = 100  # Indexing in batches
    
    print("Generating local embeddings and indexing sections. This may take a moment...")
    
    for i in range(0, total_docs, batch_size):
        batch = all_chunks[i:i+batch_size]
        ids = [doc["id"] for doc in batch]
        documents = [doc["content"] for doc in batch]
        metadatas = [doc["metadata"] for doc in batch]
        
        # Calculate embeddings using local sentence-transformers model
        try:
            embeddings = model.encode(documents, show_progress_bar=False).tolist()
        except Exception as ex:
            print(f"Error generating local embeddings for batch: {ex}")
            # Fallback to zero embeddings of dimension 384 (all-MiniLM-L6-v2 size)
            embeddings = [[0.0] * 384 for _ in documents]
        
        # Add to ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        progress = 0.15 + (0.85 * (min(i + batch_size, total_docs) / total_docs))
        status_msg = f"Indexed {min(i + batch_size, total_docs)} / {total_docs} sections locally..."
        print(status_msg)
        if progress_callback:
            progress_callback(progress, status_msg)
            
    print("Vector database indexing complete!")
    if progress_callback:
        progress_callback(1.0, "Vector DB successfully indexed locally!")
        
    return len(all_chunks)

if __name__ == "__main__":
    # Test script run from CLI
    build_vector_store(api_key=None)
