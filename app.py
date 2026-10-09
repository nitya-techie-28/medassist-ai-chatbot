"""
MedAssist AI: Modern Clinical Conversational Interface
Author: Nitya (Intern ID: 11005)
Task: Task 3 - Conversational NLP AI Chatbot
Organization: Navodita Infotech
Frontend: Streamlit Custom Clinical Suite
"""

import streamlit as st
from rag_engine import MedRAGEngine

# Page Configuration
st.set_page_config(
    page_title="MedAssist AI - Healthcare Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Tech Medical UI Styling
st.markdown("""
<style>
    /* Dark clinical aesthetic */
    .stApp {
        background-color: #080d1a;
        color: #e2e8f0;
    }
    
    /* Header Card */
    .header-box {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.6) 0%, rgba(15, 23, 42, 0.9) 100%);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(16, 185, 129, 0.3);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        margin-bottom: 24px;
    }
    
    /* Quick Pill Buttons */
    .stButton>button {
        background-color: #0f172a;
        color: #38bdf8;
        border: 1px solid #1e293b;
        border-radius: 10px;
        font-size: 0.85rem;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #064e3b;
        color: #34d399;
        border-color: #10b981;
    }
    
    /* Disclaimer Bar */
    .disclaimer-card {
        background-color: rgba(245, 158, 11, 0.08);
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #fde68a;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<div class="header-box">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <h1 style="color: #34d399; margin: 0; font-size: 1.85rem; font-weight: 700;">
                🏥 MedAssist AI <span style="font-size: 1rem; color: #38bdf8; font-weight: 400; margin-left: 8px;">v2.4 Enterprise</span>
            </h1>
            <p style="color: #94a3b8; margin-top: 6px; margin-bottom: 0; font-size: 0.95rem;">
                Context-Aware Clinical Decision Support &amp; Patient Triage System
            </p>
        </div>
        <div style="text-align: right;">
            <span style="background-color: #065f46; color: #a7f3d0; padding: 6px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">
                ● NEURAL RAG ENGINE ACTIVE
            </span>
            <p style="color: #64748b; font-size: 0.8rem; margin-top: 6px; margin-bottom: 0;">
                Intern: <strong>Nitya</strong> (ID: 11005) | Navodita Infotech
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Safety Disclaimer
st.markdown("""
<div class="disclaimer-card">
    ⚠️ <strong>Clinical Notice:</strong> MedAssist AI provides informational guidance referenced against evidence-based medical documentation. In the event of an acute emergency, dial 112 / 911 or visit the nearest emergency department immediately.
</div>
""", unsafe_allow_html=True)

# Initialize RAG Engine
@st.cache_resource
def load_rag():
    return MedRAGEngine("knowledge_base.txt")

with st.spinner("Connecting neural vector weights & indexing clinical protocols..."):
    engine = load_rag()

# Sidebar: System Health & Inquiries
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=500&auto=format&fit=crop&q=60", caption="Clinical Telemetry Console")
    
    st.markdown("### 📊 Architecture Stack")
    st.markdown("- **Embedding Model:** `all-MiniLM-L6-v2`")
    st.markdown("- **Vector Index:** FAISS (Normalized Cosine)")
    st.markdown("- **Retrieval:** Semantic Top-$k$ Similarity")
    st.markdown("- **Synthesizer:** Universal Clinical Triage")
    
    st.markdown("---")
    st.markdown("### 💡 Quick Clinical Protocols")
    
    quick_queries = [
        "What are the warning signs of cancer?",
        "What is the first aid protocol for hypoglycemia?",
        "When does high blood pressure become a crisis?",
        "What are the signs of an acute stroke?",
        "How to treat first and second-degree burns?",
        "Emergency steps for anaphylactic shock?"
    ]
    
    for q in quick_queries:
        if st.button(q, use_container_width=True):
            st.session_state["user_query_input"] = q

    st.markdown("---")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state["chat_messages"] = []
        st.rerun()

# Chat History State
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {
            "role": "assistant",
            "content": "👋 Greetings! I am **MedAssist AI**, your clinical triage and medical query assistant. You can ask me any health-related question, from symptom evaluations (like cancer warning signs, chest pain, or fever) to emergency first-aid protocols.",
            "sources": []
        }
    ]

# Display Existing Chat Messages
for msg in st.session_state["chat_messages"]:
    with st.chat_message(msg["role"], avatar="👨‍⚕️" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("🔍 Verified Knowledge Attribution (FAISS Vector Store)", expanded=False):
                for idx, src in enumerate(msg["sources"], 1):
                    score_pct = round(src["score"] * 100, 1)
                    st.markdown(f"**Document Chunk [{idx}]:** `{src['title']}` | **Match Score:** `{score_pct}%`")
                    st.code(src["chunk"], language="markdown")

# Process Input
user_input = st.chat_input("Ask any medical question (e.g., What are the signs of cancer? How to treat high fever?)...")

if "user_query_input" in st.session_state:
    user_input = st.session_state.pop("user_query_input")

if user_input:
    # Append User Message
    st.session_state["chat_messages"].append({"role": "user", "content": user_input, "sources": []})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Generate Assistant Response
    with st.chat_message("assistant", avatar="👨‍⚕️"):
        with st.spinner("Analyzing query against clinical knowledge base & vector store..."):
            reply, sources = engine.generate_response(user_input)
            st.markdown(reply)
            
            if sources:
                with st.expander("🔍 Verified Knowledge Attribution (FAISS Vector Store)", expanded=True):
                    for idx, src in enumerate(sources, 1):
                        score_pct = round(src["score"] * 100, 1)
                        st.markdown(f"**Document Chunk [{idx}]:** `{src['title']}` | **Match Score:** `{score_pct}%`")
                        st.code(src["chunk"], language="markdown")

    # Persist in state
    st.session_state["chat_messages"].append({
        "role": "assistant",
        "content": reply,
        "sources": sources
    })