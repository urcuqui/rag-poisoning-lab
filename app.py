"""RAG Poisoning Security Lab - Flask Application using FAISS + HuggingFace."""

import threading
from typing import Optional
from flask import Flask, render_template, request, jsonify
from langchain_classic.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

app = Flask(__name__)

# Global state
rag_system = None
comments_store = []
lock = threading.Lock()

# Initial clean knowledge base
CLEAN_DOCUMENTS = [
    "RedJack malware is a credential-stealing trojan that steals passwords.",
    "Don't whitelist suspicious domains. Always verify domain legitimacy first.",
    "MITRE ATT&CK T1566 refers to phishing techniques used by attackers.",
    "Always verify SSL certificates before trusting a website.",
    "Suspicious email attachments should never be opened without scanning.",
    "Multi-factor authentication (MFA) should be enabled on all accounts.",
    "Secure-mfa-login.com is a known phishing domain - do not trust it.",
]


class RAGSystem:
    """Manages the RAG system backed by FAISS and a HuggingFace LLM."""

    def __init__(self):
        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.llm = self._build_llm()
        self.qa_prompt = PromptTemplate(
            template=(
                "You are a cybersecurity assistant. Answer ONLY using the context below.\n"
                'If the answer is not in the context, say "Unknown".\n\n'
                "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n"
            ),
            input_variables=["context", "question"],
        )
        self.vectorstore = None
        self.retriever = None
        self.qa = None
        self.include_clean_docs = True
        self.current_clean_docs = 0
        self._rebuild_vectorstore()

    def _build_llm(self):
        """Load the seq2seq model and wrap it for LangChain."""
        model_name = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
        hf_pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
        )
        return HuggingFacePipeline(pipeline=hf_pipe)

    def _rebuild_vectorstore(self, include_clean: Optional[bool] = None):
        """Recreate the FAISS store and retrieval chain with current docs."""
        if include_clean is None:
            include_clean = self.include_clean_docs
        else:
            self.include_clean_docs = include_clean

        documents = []
        if include_clean:
            documents.extend(CLEAN_DOCUMENTS)
            self.current_clean_docs = len(CLEAN_DOCUMENTS)
        else:
            self.current_clean_docs = 0
        documents.extend(comments_store)

        if not documents:
            self.vectorstore = None
            self.retriever = None
            self.qa = None
            return

        self.vectorstore = FAISS.from_texts(documents, self.embedder)
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": self.qa_prompt},
        )

    def add_comment(self, comment: str):
        """Add a comment to the store and rebuild."""
        comments_store.append(comment)
        self._rebuild_vectorstore()

    def refresh(self):
        """Clear comments and rebuild without clean docs."""
        comments_store.clear()
        # Reset vector components so FAISS is rebuilt without clean docs
        self.vectorstore = None
        self.retriever = None
        self.qa = None
        self._rebuild_vectorstore(include_clean=False)

    def ask(self, query: str) -> str:
        """Query the RAG system and generate a response."""
        if not query:
            return "No query provided."

        try:
            if not self.qa:
                return "Knowledge base is empty. Please add data first."
            return self.qa.run(query)
        except Exception as exc:
            return f"Error generating answer: {exc}"

    def get_stats(self) -> dict:
        """Get current system stats."""
        return {
            "clean_docs": self.current_clean_docs,
            "comments": len(comments_store),
            "total_docs": self.vectorstore.index.ntotal if self.vectorstore else 0,
        }


def get_rag_system():
    """Get or initialize the RAG system."""
    global rag_system
    with lock:
        if rag_system is None:
            rag_system = RAGSystem()
        return rag_system


@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle chat queries."""
    data = request.get_json()
    query = data.get("query", "")
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    rag = get_rag_system()
    response = rag.ask(query)
    
    return jsonify({"response": response})


@app.route("/api/comment", methods=["POST"])
def add_comment():
    """Add a comment to the RAG system (potential poisoning vector)."""
    data = request.get_json()
    comment = data.get("comment", "")
    
    if not comment:
        return jsonify({"error": "No comment provided"}), 400
    
    rag = get_rag_system()
    rag.add_comment(comment)
    
    return jsonify({
        "message": "Comment added to knowledge base",
        "stats": rag.get_stats()
    })


@app.route("/api/refresh", methods=["POST"])
def refresh():
    """Refresh the RAG system (clear poisoned data)."""
    rag = get_rag_system()
    rag.refresh()
    
    return jsonify({
        "message": "Knowledge base refreshed (comments cleared)",
        "stats": rag.get_stats()
    })


@app.route("/api/stats", methods=["GET"])
def stats():
    """Get current RAG system stats."""
    rag = get_rag_system()
    return jsonify(rag.get_stats())


@app.route("/api/comments", methods=["GET"])
def get_comments():
    """Get all current comments."""
    return jsonify({"comments": comments_store})


if __name__ == "__main__":
    print("Initializing RAG system...")
    get_rag_system()  # Pre-initialize
    print("Starting Flask server...")
    app.run(debug=True, port=5000)

