import os
import sys
import subprocess
import streamlit as st

# Self-healing upgrade for the new google-genai SDK to resolve deprecation issues
upgraded_sdk = False
try:
    from google import genai
except ImportError:
    import sys
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "google-genai"])
        upgraded_sdk = True
    except Exception as e:
        pass

import pandas as pd
import time
import data_processor
import rag_engine


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
    st.session_state.api_key = os.environ.get("GEMINI_API_KEY", "")

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown('<div style="text-align: center;"><span style="font-size: 3.5rem;">⚖️</span></div>', unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; margin-top: 0;'>Criminal Law Transition</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.85rem; color:#888;'>Compare legacy codes (IPC, CrPC, IEA) with active reforms (BNS, BNSS, BSA) using AI</p>", unsafe_allow_html=True)
    st.write("---")
    
    # API Key Input
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=st.session_state.api_key,
        type="password",
        help="Required to run embedding search and chatbot. Set the GEMINI_API_KEY environment variable or paste it here."
    )
    
    if api_key_input:
        st.session_state.api_key = api_key_input
        st.success("✓ API Key loaded")
    else:
        st.warning("⚠️ Enter Gemini API Key to run searches")
        
    st.write("---")
    
    # Database Status and Management
    st.markdown("##### Vector Database Status")
    collection = rag_engine.get_chroma_collection()
    
    if collection is None or collection.count() == 0:
        st.error("❌ Database not indexed.")
        st.write("The database is currently empty. Run the indexing routine below using your API key to download and compile statutory mappings.")
        
        # Ingestion Trigger Button
        if st.button("📥 Ingest & Build Index", use_container_width=True):
            if not st.session_state.api_key:
                st.error("Please enter a valid Gemini API Key first.")
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
                    st.success(f"✓ Ingested & indexed {total_indexed} sections successfully!")
                    time.sleep(2)
                    st.rerun()
                except Exception as ex:
                    st.error(f"Ingestion failed: {ex}")
    else:
        doc_count = collection.count()
        st.success(f"✓ Database Ready ({doc_count} provisions indexed)")
        
        # Optional Re-index button
        if st.button("🔄 Rebuild Law Index", use_container_width=True):
            if not st.session_state.api_key:
                st.error("Please enter a Gemini API Key to run embeddings.")
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
                    st.success(f"✓ Re-indexed {total_indexed} sections successfully!")
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
st.markdown("<h1 class='app-title'>Indian Criminal Law Transition Portal</h1>", unsafe_allow_html=True)
st.markdown("<p class='app-subtitle'>Study and query differences between legacy codes and the new criminal laws easily</p>", unsafe_allow_html=True)

if upgraded_sdk:
    st.warning("🔄 **The new Google Gen AI SDK (google-genai) has been installed** to resolve compatibility issues and deprecation warnings. Please **restart your Streamlit server** in your terminal window (press `Ctrl+C` to stop it, then run `streamlit run app.py` again) to apply the update.")


# Tabs Navigation
tab_chat, tab_translate, tab_dashboard = st.tabs([
    "💬 Comparative Chatbot (RAG)", 
    "🔍 Section Translator", 
    "📊 Law Reform Dashboard"
])

# ==============================================================================
# TAB 1: Comparative Chatbot (RAG)
# ==============================================================================
with tab_chat:
    st.markdown("### Ask the AI Legal Assistant")
    st.write("Ask questions comparing legacy and new codes. The chatbot uses semantic search over statutory mappings to supply correct sections and detail reforms.")
    
    # Check if API Key and DB are present
    db_ready = collection is not None and collection.count() > 0
    
    if not st.session_state.api_key:
        st.info("💡 Please input your Google Gemini API Key in the sidebar to talk with the chatbot.")
    elif not db_ready:
        st.info("💡 The database has no documents. Please trigger 'Ingest & Build Index' in the sidebar first.")
    else:
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="user-bubble"><b>You:</b><br>{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-bubble"><b>Assistant:</b><br>{message["content"]}</div>', unsafe_allow_html=True)
                
                # Check for citations to display
                if "citations" in message and message["citations"]:
                    with st.expander("📚 View Retreived Section Contexts"):
                        for cite in message["citations"]:
                            law_type = cite["metadata"].get("law_type", "General")
                            badge_cls = "badge-ipc" if "IPC" in law_type else ("badge-crpc" if "CrPC" in law_type else "badge-iea")
                            st.markdown(
                                f'<span class="law-badge {badge_cls}">{law_type}</span> '
                                f'<b>Old:</b> Sec {cite["metadata"].get("old_section")} ({cite["metadata"].get("old_heading")}) | '
                                f'<b>New:</b> Sec {cite["metadata"].get("new_section")} ({cite["metadata"].get("new_heading")})<br>'
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
                
                # Rerun to render chat bubble
                st.rerun()

# ==============================================================================
# TAB 2: Section Translator
# ==============================================================================
with tab_translate:
    st.markdown("### Side-by-Side Section Lookup")
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
        lookup_clicked = st.button("Translate & Compare", use_container_width=True)
        
    if lookup_clicked or section_number:
        if not db_ready:
            st.error("Database is empty. Please run indexing from the sidebar first.")
        elif not section_number:
            st.warning("Please enter a section number to translate.")
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
                                📜 Old: {selected_law.split(" to ")[0]} - Sec {translation_result["old_section"]}
                            </div>
                            <p><b>Heading:</b> {translation_result["old_heading"]}</p>
                            <p style="font-size:0.95rem; line-height:1.5; color:#D1D5DB;">{translation_result["old_description"]}</p>
                        </div>
                        
                        <!-- New Law Card -->
                        <div class="compare-half" style="border-color: rgba(96, 165, 250, 0.2); background: rgba(59, 130, 246, 0.02);">
                            <div class="compare-title new-law-color">
                                ⚖️ New: {selected_law.split(" to ")[1].split(" ")[0]} - Sec {translation_result["new_section"]}
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
    st.markdown("### Core Legal Reform Summaries")
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
        
    st.markdown("### Crucial Transitions & Major Systemic Changes")
    
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
