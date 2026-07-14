import os
# Load local .env variables if present
if os.path.exists(".env"):
    try:
        with open(".env", "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#") and "=" in line:
                    k, v = line.strip().split("=", 1)
                    os.environ[k.strip()] = v.strip()
    except Exception:
        pass

# Set environment variables to prevent multi-threaded deadlocks and offline hangs in PyTorch/OpenMP/MKL/HuggingFace
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_num_threads(1)

import sys
import subprocess
import streamlit as st

# Self-healing upgrade for the groq SDK
upgraded_sdk = False
try:
    import groq
except ImportError:
    import sys
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "groq"])
        upgraded_sdk = True
    except Exception as e:
        pass

import pandas as pd
import time
import json
import data_processor
import rag_engine

# --- Chat History Helpers ---
def save_chat_session(chat_history, chat_id=None):
    if not chat_history:
        return chat_id
        
    os.makedirs(os.path.join(os.getcwd(), "database", "chats"), exist_ok=True)
    
    if chat_id is None:
        chat_id = f"chat_{int(time.time())}.json"
        
    # Get a title based on the first user query
    first_query = ""
    for msg in chat_history:
        if msg["role"] == "user":
            first_query = msg["content"]
            break
            
    title = first_query[:35] + "..." if len(first_query) > 35 else (first_query or "Untitled Chat")
    
    chat_file_path = os.path.join(os.getcwd(), "database", "chats", chat_id)
    chat_data = {
        "title": title,
        "timestamp": time.time(),
        "history": chat_history
    }
    
    try:
        with open(chat_file_path, "w", encoding="utf-8") as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving chat session: {e}")
        
    return chat_id

def get_chat_sessions():
    chats_dir = os.path.join(os.getcwd(), "database", "chats")
    if not os.path.exists(chats_dir):
        return []
        
    sessions = []
    for fname in os.listdir(chats_dir):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(chats_dir, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sessions.append({
                        "id": fname,
                        "title": data.get("title", "Untitled Chat"),
                        "timestamp": data.get("timestamp", 0)
                    })
            except Exception:
                pass
                
    # Sort by timestamp descending (newest first)
    sessions.sort(key=lambda x: x["timestamp"], reverse=True)
    return sessions

def load_chat_session(chat_id):
    chat_file_path = os.path.join(os.getcwd(), "database", "chats", chat_id)
    if not os.path.exists(chat_file_path):
        return []
    try:
        with open(chat_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("history", [])
    except Exception as e:
        print(f"Error loading chat session: {e}")
        return []

def delete_chat_session(chat_id):
    chat_file_path = os.path.join(os.getcwd(), "database", "chats", chat_id)
    if os.path.exists(chat_file_path):
        try:
            os.remove(chat_file_path)
            return True
        except Exception as e:
            print(f"Error deleting chat session: {e}")
    return False
# --- Custom Styling & CSS (Aesthetics and Premium UX) ---
def apply_custom_css():
    st.markdown("""
    <style>
    /* Import modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main header styling with gradient */
    .app-title {
        background: linear-gradient(90deg, #6366F1 0%, #3B82F6 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.05rem;
    }
    
    .app-subtitle {
        font-weight: 400;
        font-size: 1.1rem;
        color: #888896;
        margin-bottom: 2rem;
    }
    
    /* Premium Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    .law-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    
    .badge-ipc { background-color: rgba(99, 102, 241, 0.15); color: #818CF8; border: 1px dashed #6366F1; }
    .badge-crpc { background-color: rgba(59, 130, 246, 0.15); color: #60A5FA; border: 1px dashed #3B82F6; }
    .badge-iea { background-color: rgba(236, 72, 153, 0.15); color: #F472B6; border: 1px dashed #EC4899; }
    
    /* Side-by-side comparison tables & cards */
    .comparison-container {
        display: flex;
        gap: 20px;
        width: 100%;
        margin-top: 15px;
    }
    
    .compare-half {
        flex: 1;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 18px;
    }
    
    .compare-title {
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 8px;
    }
    
    .old-law-color { color: #E5E7EB; }
    .new-law-color { color: #60A5FA; font-weight: bold; }
    
    /* Chat message styles */
    .user-bubble {
        background-color: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.2);
        padding: 15px;
        border-radius: 12px 12px 0px 12px;
        margin-bottom: 15px;
        text-align: right;
    }
    
    .assistant-bubble {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 18px;
        border-radius: 12px 12px 12px 0px;
        margin-bottom: 20px;
        line-height: 1.6;
    }
    
    /* Citation/Source Box */
    .citation-box {
        background: rgba(59, 130, 246, 0.05);
        border-left: 3px solid #3B82F6;
        border-radius: 0 8px 8px 0;
        padding: 10px 15px;
        margin-top: 10px;
        font-size: 0.85rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Set Page Config
st.set_page_config(
    page_title="Indian Criminal Laws comparative RAG Tool",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Styling
apply_custom_css()

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_key" not in st.session_state:
    st.session_state.api_key = os.environ.get("GROQ_API_KEY", "")
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None
if "prev_selected_chat_id" not in st.session_state:
    st.session_state.prev_selected_chat_id = None

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown('<div style="text-align: center; color: #6366F1; margin-bottom: 10px;"><i class="bi bi-balance" style="font-size: 3.5rem;"></i></div>', unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; margin-top: 0;'>Criminal Law Transition</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.85rem; color:#888;'>Compare legacy codes (IPC, CrPC, IEA) with active reforms (BNS, BNSS, BSA) using AI</p>", unsafe_allow_html=True)
    st.write("---")
    
    # API Key Input
    api_key_input = st.text_input(
        "Groq API Key",
        value=st.session_state.api_key,
        type="password",
        help="Required to run chatbot. Set the GROQ_API_KEY environment variable or paste it here."
    )
    
    if api_key_input:
        st.session_state.api_key = api_key_input
        st.success("API Key loaded", icon=":material/vpn_key:")
    else:
        st.warning("Enter Groq API Key to run searches", icon=":material/key:")
        
    st.write("---")
    
    # Conversation History (Past Chats)
    st.markdown("##### :material/history: Conversation History")
    sessions = get_chat_sessions()
    options = ["New Conversation"] + [s["title"] for s in sessions]
    option_ids = [None] + [s["id"] for s in sessions]
    
    current_index = 0
    if st.session_state.active_chat_id in option_ids:
        current_index = option_ids.index(st.session_state.active_chat_id)
        
    selected_option_index = st.selectbox(
        "Select conversation:",
        range(len(options)),
        format_func=lambda x: options[x],
        index=current_index,
        key="chat_history_selector"
    )
    
    selected_chat_id = option_ids[selected_option_index]
    
    # Handle conversation change
    if st.session_state.prev_selected_chat_id != selected_chat_id:
        st.session_state.prev_selected_chat_id = selected_chat_id
        st.session_state.active_chat_id = selected_chat_id
        if selected_chat_id:
            st.session_state.chat_history = load_chat_session(selected_chat_id)
        else:
            st.session_state.chat_history = []
        st.rerun()
        
    # Delete Selected Chat
    if st.session_state.active_chat_id:
        if st.button("Delete Selected Conversation", icon=":material/delete:", use_container_width=True):
            delete_chat_session(st.session_state.active_chat_id)
            st.session_state.active_chat_id = None
            st.session_state.chat_history = []
            st.session_state.prev_selected_chat_id = None
            st.success("Conversation deleted!")
            time.sleep(1)
            st.rerun()
            
    st.write("---")
    
    # Database Status and Management
    st.markdown("##### :material/database: Vector Database Status")
    collection = rag_engine.get_chroma_collection()
    
    if collection is None or collection.count() == 0:
        st.error("Database not indexed.", icon=":material/database_off:")
        st.write("The database is currently empty. Run the indexing routine below using your API key to download and compile statutory mappings.")
        
        # Ingestion Trigger Button
        if st.button("Ingest & Build Index", icon=":material/cloud_download:", use_container_width=True):
            if not st.session_state.api_key:
                st.error("Please enter a valid Groq API Key first.")
            else:
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                def progress_cb(prog_val, msg_val):
                    progress_bar.progress(prog_val)
                    status_text.text(msg_val)
                
                try:
                    total_indexed = data_processor.build_vector_store(
                        api_key=st.session_state.api_key,
                        progress_callback=progress_cb
                    )
                    st.success(f"Ingested & indexed {total_indexed} sections successfully!", icon=":material/check_circle:")
                    time.sleep(2)
                    st.rerun()
                except Exception as ex:
                    st.error(f"Ingestion failed: {ex}")
    else:
        doc_count = collection.count()
        st.success(f"Database Ready ({doc_count} provisions)", icon=":material/database:")
        
        # Optional Re-index button
        if st.button("Rebuild Law Index", icon=":material/refresh:", use_container_width=True):
            if not st.session_state.api_key:
                st.error("Please enter a Groq API Key first.")
            else:
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                def progress_cb(prog_val, msg_val):
                    progress_bar.progress(prog_val)
                    status_text.text(msg_val)
                    
                try:
                    total_indexed = data_processor.build_vector_store(
                        api_key=st.session_state.api_key,
                        progress_callback=progress_cb
                    )
                    st.success(f"Re-indexed {total_indexed} sections successfully!", icon=":material/check_circle:")
                    time.sleep(1)
                    st.rerun()
                except Exception as ex:
                    st.error(f"Re-indexing failed: {ex}")

    st.write("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#666;'>"
        "<b>Law Scope:</b><br>"
        "• IPC (1860) → BNS (2023)<br>"
        "• CrPC (1973) → BNSS (2023)<br>"
        "• IEA (1872) → BSA (2023)<br><br>"
        "Active Date: July 1, 2024"
        "</div>",
        unsafe_allow_html=True
    )

# --- Main Screen Layout ---
st.markdown("<h1 class='app-title'><i class='bi bi-balance' style='font-size: 2.8rem; margin-right: 15px;'></i>Indian Criminal Law Transition Portal</h1>", unsafe_allow_html=True)
st.markdown("<p class='app-subtitle'>Study and query differences between legacy codes and the new criminal laws easily</p>", unsafe_allow_html=True)

if upgraded_sdk:
    st.warning("The Groq SDK has been installed. Please restart your Streamlit server in your terminal window (press `Ctrl+C` to stop it, then run `streamlit run app.py` again) to apply the update.", icon=":material/warning:")


# Tabs Navigation
tab_chat, tab_translate, tab_dashboard = st.tabs([
    ":material/forum: Comparative Chatbot (RAG)", 
    ":material/translate: Section Translator", 
    ":material/analytics: Law Reform Dashboard"
])

# ==============================================================================
# TAB 1: Comparative Chatbot (RAG)
# ==============================================================================
with tab_chat:
    st.markdown("### :material/forum: Ask the AI Legal Assistant")
    st.write("Ask questions comparing legacy and new codes. The chatbot uses semantic search over statutory mappings to supply correct sections and detail reforms.")
    
    # Check if API Key and DB are present
    db_ready = collection is not None and collection.count() > 0
    
    if not st.session_state.api_key:
        st.info("Please input your Groq API Key in the sidebar to talk with the chatbot.", icon=":material/key:")
    elif not db_ready:
        st.info("The database has no documents. Please trigger 'Ingest & Build Index' in the sidebar first.", icon=":material/database:")
    else:
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="user-bubble"><b>You:</b><br>{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-bubble"><b>Assistant:</b><br>{message["content"]}</div>', unsafe_allow_html=True)
                
                # Check for citations to display
                if "citations" in message and message["citations"]:
                    with st.expander("📚 View Retrieved Section Contexts"):
                        for cite in message["citations"]:
                            law_type = cite["metadata"].get("law_type", "General")
                            badge_cls = "badge-ipc" if "IPC" in law_type else ("badge-crpc" if "CrPC" in law_type else "badge-iea")
                            
                            old_sec = cite["metadata"].get("old_section", "")
                            old_heading = cite["metadata"].get("old_heading", "")
                            new_sec = cite["metadata"].get("new_section", "")
                            new_heading = cite["metadata"].get("new_heading", "")
                            
                            if old_sec:
                                citation_header = f'<b>Old:</b> Sec {old_sec} ({old_heading}) | <b>New:</b> Sec {new_sec} ({new_heading})'
                            else:
                                citation_header = f'<b>New Law Detail:</b> Sec {new_sec} ({new_heading})'
                                
                            st.markdown(
                                f'<span class="law-badge {badge_cls}">{law_type}</span> '
                                f'{citation_header}<br>'
                                f'<div class="citation-box">{cite["content"].replace(chr(10), "<br>")}</div><br>',
                                unsafe_allow_html=True
                            )
        
        # User input query
        query = st.chat_input("Ask a question (e.g. 'What are the main changes in the punishment of Cheating?' or 'How is Sedition replaced?')")
        
        if query:
            # Display user query
            st.markdown(f'<div class="user-bubble"><b>You:</b><br>{query}</div>', unsafe_allow_html=True)
            
            with st.spinner("Searching statutory records and formulating answer..."):
                # 1. Query local database to get context
                retrieved_sections = rag_engine.query_vector_store(
                    query_text=query,
                    api_key=st.session_state.api_key,
                    top_k=5
                )
                
                # 2. Call LLM to generate comparative explanation
                response_text = rag_engine.generate_rag_response(
                    query_text=query,
                    context_docs=retrieved_sections,
                    api_key=st.session_state.api_key,
                    chat_history=st.session_state.chat_history
                )
                
                # Update history
                message_entry = {
                    "role": "assistant",
                    "content": response_text,
                    "citations": retrieved_sections
                }
                
                st.session_state.chat_history.append({"role": "user", "content": query})
                st.session_state.chat_history.append(message_entry)
                
                # Save chat session to disk
                st.session_state.active_chat_id = save_chat_session(
                    st.session_state.chat_history,
                    st.session_state.active_chat_id
                )
                
                # Rerun to render chat bubble
                st.rerun()

# ==============================================================================
# TAB 2: Section Translator
# ==============================================================================
with tab_translate:
    st.markdown("### :material/compare: Side-by-Side Section Lookup")
    st.write("Input a legacy section number (like IPC 302, CrPC 438, IEA 65B) or a new section number to translate and view the exact text modifications side-by-side.")
    
    col_sel_type, col_sel_sec, col_btn = st.columns([3, 3, 2])
    
    with col_sel_type:
        selected_law = st.selectbox(
            "Select Act Category",
            ["IPC to BNS (Substantive)", "CrPC to BNSS (Procedural)", "IEA to BSA (Evidence)"]
        )
    with col_sel_sec:
        section_number = st.text_input("Enter Section Number (e.g. 302, 154, 62, 65B, 103)")
        
    with col_btn:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        lookup_clicked = st.button("Translate & Compare", icon=":material/compare_arrows:", use_container_width=True)
        
    if lookup_clicked or section_number:
        if not db_ready:
            st.error("Database is empty. Please run indexing from the sidebar first.", icon=":material/database_off:")
        elif not section_number:
            st.warning("Please enter a section number to translate.", icon=":material/warning:")
        else:
            with st.spinner("Locating statutory references..."):
                translation_result = rag_engine.direct_lookup(selected_law, section_number)
                
                if translation_result:
                    badge_cls = "badge-ipc" if "IPC" in selected_law else ("badge-crpc" if "CrPC" in selected_law else "badge-iea")
                    
                    st.markdown(
                        f'<div class="glass-card">'
                        f'<span class="law-badge {badge_cls}">{translation_result["law_type"]} Transition</span>'
                        f'<h4>Old Section {translation_result["old_section"]} ➔ New Section {translation_result["new_section"]}</h4>'
                        f'</div>', 
                        unsafe_allow_html=True
                    )
                    
                    # Side-by-Side Columns
                    st.markdown(f"""
                    <div class="comparison-container">
                        <!-- Old Law Card -->
                        <div class="compare-half">
                            <div class="compare-title old-law-color">
                                <i class="bi bi-journal-text" style="margin-right: 8px;"></i>Old: {selected_law.split(" to ")[0]} - Sec {translation_result["old_section"]}
                            </div>
                            <p><b>Heading:</b> {translation_result["old_heading"]}</p>
                            <p style="font-size:0.95rem; line-height:1.5; color:#D1D5DB;">{translation_result["old_description"]}</p>
                        </div>
                        
                        <!-- New Law Card -->
                        <div class="compare-half" style="border-color: rgba(96, 165, 250, 0.2); background: rgba(59, 130, 246, 0.02);">
                            <div class="compare-title new-law-color">
                                <i class="bi bi-gavel" style="margin-right: 8px;"></i>New: {selected_law.split(" to ")[1].split(" ")[0]} - Sec {translation_result["new_section"]}
                            </div>
                            <p><b>Heading:</b> {translation_result["new_heading"]}</p>
                            <p style="font-size:0.95rem; line-height:1.5; color:#D1D5DB;">{translation_result["new_description"]}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(f"No direct mapping found for Section '{section_number}' under category '{selected_law}'. "
                               f"Please note that the new laws have completely restructured, merged, or deleted some provisions. "
                               f"Try asking the Comparative Chatbot tab a general question about this offence.")

# ==============================================================================
# TAB 3: Law Reform Dashboard
# ==============================================================================
with tab_dashboard:
    st.markdown("### :material/assignment: Core Legal Reform Summaries")
    st.write("Understand the key systemic shifts brought by the new laws starting July 1, 2024.")
    
    # 3 Column Stat cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="glass-card" style="border-top: 4px solid #6366F1;">
            <h4>Bharatiya Nyaya Sanhita (BNS)</h4>
            <p style="font-size: 0.85rem; color:#888;">Replaces Indian Penal Code (IPC) 1860</p>
            <hr style="opacity: 0.1;">
            <ul style="font-size:0.9rem; padding-left:15px; margin-bottom: 0;">
                <li><b>Total Sections:</b> Reduced from 511 to 358.</li>
                <li>Sedition is repealed; replaced by treason laws (Section 152).</li>
                <li>Community service introduced as a new form of punishment.</li>
                <li>Explicit definition of mob lynching and organized crime.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="glass-card" style="border-top: 4px solid #3B82F6;">
            <h4>Bharatiya Nagarik Suraksha Sanhita (BNSS)</h4>
            <p style="font-size: 0.85rem; color:#888;">Replaces Code of Criminal Procedure (CrPC) 1973</p>
            <hr style="opacity: 0.1;">
            <ul style="font-size:0.9rem; padding-left:15px; margin-bottom: 0;">
                <li><b>Total Sections:</b> Expanded from 484 to 531.</li>
                <li><b>Zero FIR:</b> Mandatory registration anywhere in India.</li>
                <li>Mandatory audio-video recording of search and seizure.</li>
                <li>Stricter timelines for charge sheet filing and judgements.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="glass-card" style="border-top: 4px solid #EC4899;">
            <h4>Bharatiya Sakshya Adhiniyam (BSA)</h4>
            <p style="font-size: 0.85rem; color:#888;">Replaces Indian Evidence Act (IEA) 1872</p>
            <hr style="opacity: 0.1;">
            <ul style="font-size:0.9rem; padding-left:15px; margin-bottom: 0;">
                <li><b>Total Sections:</b> Expanded from 167 to 170.</li>
                <li><b>Primary Evidence:</b> Device logs, cloud files, messages.</li>
                <li>Standardized certificate format for electronic evidence.</li>
                <li>Allows testimony of experts/witnesses virtually.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### :material/swap_horiz: Crucial Transitions & Major Systemic Changes")
    
    # Comparison table for dashboard
    comparison_data = {
        "Theme": [
            "FIR Filing", 
            "Sedition (124A)", 
            "Organized Crime", 
            "Terrorist Acts", 
            "Electronic Evidence", 
            "Trial Presence", 
            "Police Custody"
        ],
        "Legacy Framework (IPC / CrPC / IEA)": [
            "Must register FIR in local jurisdiction (except informal Zero FIRs which were slow).",
            "Broadly defined as inciting disaffection toward government (highly controversial).",
            "No specific central definition under IPC; relied on state acts (like MCOCA).",
            "Handled exclusively under special acts like UAPA, not in general penal code.",
            "Required signed physical paper certificates (under 65B(4)) for secondary evidence.",
            "Accused presence required; virtual testimony not systematically integrated.",
            "Police custody allowed only within the first 15 days of arrest."
        ],
        "Reformed Framework (BNS / BNSS / BSA)": [
            "Formal Zero FIR mandated. Informant can file e-FIR, signed within 3 days.",
            "Repealed. Replaced by Sec 152 BNS targeting acts endangering unity & sovereignty.",
            "Explicitly defined and penalized (Section 111 BNS). Covers syndicates.",
            "Brought into the general penal code (Section 113 BNS) with defined penalties.",
            "Stored digital records, servers, cloud devices recognized as Primary Evidence.",
            "Section 530 BNSS permits virtual trials, statements, and witness expert panels.",
            "Police custody allowed in parts/installments during first 40 or 60 days (Sec 187 BNSS)."
        ]
    }
    
    df_compare = pd.DataFrame(comparison_data)
    st.table(df_compare)
