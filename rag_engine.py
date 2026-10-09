"""
MedAssist AI: Conversational Retrieval-Augmented Generation (RAG)
Engine: Direct AutoModel Seq2Seq Inference + FAISS Cosine Index
Author: Nitya (Intern ID: 11005)
Task: Task 3 - Conversational NLP AI Chatbot (Navodita Infotech)
"""

import os
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class MedRAGEngine:
    def __init__(self, knowledge_file="knowledge_base.txt"):
        print("[INFO] Initializing Dense Embedding Retriever...")
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        print("[INFO] Loading Conversational Clinical LLM (google/flan-t5-base)...")
        model_name = "google/flan-t5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Place model on GPU if available, else CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        
        self.chunks = []
        self.titles = []
        self.index = None
        self.build_index(knowledge_file)

    def build_index(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Knowledge file '{filepath}' not found.")

        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        raw_sections = raw_text.split("[SECTION:")
        for sec in raw_sections:
            sec_clean = sec.strip()
            if not sec_clean:
                continue
            lines = sec_clean.split("\n", 1)
            raw_title = lines[0].replace("]", "").strip()
            body = lines[1].strip() if len(lines) > 1 else ""
            
            self.titles.append(raw_title)
            self.chunks.append(f"{raw_title}: {body}")

        embeddings = self.embedding_model.encode(self.chunks, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f"[INFO] Successfully indexed {len(self.chunks)} verified clinical protocols.")

    def retrieve(self, query: str, top_k: int = 2):
        query_vec = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)

        sims, indices = self.index.search(query_vec, top_k)
        results = []
        for sim, idx in zip(sims[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "score": float(sim),
                    "title": self.titles[idx],
                    "chunk": self.chunks[idx]
                })
        return results

    def _generate_text(self, prompt: str, max_tokens: int = 300) -> str:
        """Direct robust Seq2Seq generation without deprecated pipeline tasks."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,
                do_sample=False,
                repetition_penalty=1.2
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_response(self, user_query: str):
        docs = self.retrieve(user_query, top_k=2)
        top_doc = docs[0] if docs else None
        
        # If relevant knowledge is matched in the vector store
        if top_doc and top_doc["score"] >= 0.25:
            context = top_doc["chunk"]
            similarity_pct = round(top_doc["score"] * 100, 1)
            
            prompt = (
                f"You are MedAssist, a professional medical assistant. Provide a detailed, clear, and comprehensive explanation for the patient's question based strictly on the provided medical reference.\n\n"
                f"Reference:\n{context}\n\n"
                f"Patient Question: {user_query}\n\n"
                f"Detailed Explanation:"
            )
            
            generated_answer = self._generate_text(prompt, max_tokens=300)
            
            # Combine the focused generative answer with clean reference points
            formatted_response = (
                f"{generated_answer}\n\n"
                f"### 📋 Key Clinical Takeaways ({top_doc['title']})\n"
                f"{context}\n\n"
                f"---\n"
                f"📊 **Telemetry:** Grounded via FAISS Vector Index (Confidence: **{similarity_pct}%**)\n"
                f"⚠️ **Clinical Advisory:** *Informational guidance based on clinical guidelines. Always consult a licensed medical provider for diagnostic evaluation.*"
            )
            return formatted_response, docs

        # Fallback for questions outside indexed protocols
        fallback_prompt = (
            f"You are MedAssist, an expert medical AI assistant. Explain the following medical condition, symptom, or concept clearly, accurately, and thoroughly for a patient.\n\n"
            f"Question: {user_query}\n\n"
            f"Detailed Medical Explanation:"
        )
        generated_answer = self._generate_text(fallback_prompt, max_tokens=300)
        
        formatted_response = (
            f"{generated_answer}\n\n"
            f"---\n"
            f"💡 **General Clinical Guidance:** For persistent, acute, or changing symptoms, prioritize an in-person clinical consultation with a healthcare professional."
        )
        return formatted_response, []