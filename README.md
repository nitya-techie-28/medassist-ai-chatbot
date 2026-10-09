# 🏥 MedAssist AI: Conversational Clinical & Healthcare RAG Assistant

**Author:** Nitya  
**Intern ID:** 11005  
**Task:** Task 3 — Conversational NLP AI Chatbot  
**Domain:** Artificial Intelligence  
**Organization:** Navodita Infotech  

---

## 📌 Project Overview
MedAssist AI is a clinical decision-support and patient triage chatbot built using a Retrieval-Augmented Generation (RAG) architecture. It synthesizes dense semantic vector search with local generative modeling to answer clinical inquiries with zero-hallucination source attribution.

## 🚀 Key Architectural Features
- **Dense Vector Embeddings:** Hugging Face `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense semantic vectors).
- **In-Memory Vector Store:** `FAISS` (Facebook AI Similarity Search) with cosine similarity normalization for fast retrieval.
- **Local Conversational LLM:** `google/flan-t5-base` for contextual response generation without external API dependencies.
- **Evidence-Based Knowledge Repository:** Verified protocols across Oncology, Cardiovascular Emergencies, Neurology (BE-FAST), Pulmonology, Endocrinology, and Emergency First-Aid.
- **Interactive UI:** Streamlit interface featuring source attribution cards, vector confidence metrics, and clinical safety disclaimers.

## 🛠️ Tech Stack
- **Language:** Python 3.12+
- **Deep Learning & NLP:** PyTorch, Hugging Face Transformers, Sentence-Transformers
- **Vector Index:** FAISS-CPU
- **Interface:** Streamlit

## ⚙️ Setup and Installation

```bash
# 1. Clone repository
git clone [https://github.com/nitya-techie-28/medassist-ai-chatbot.git](https://github.com/nitya-techie-28/medassist-ai-chatbot.git)
cd medassist-ai-chatbot

# 2. Set up virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run application
streamlit run app.py