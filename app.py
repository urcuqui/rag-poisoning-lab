"""
RAG Poisoning Security Lab - Flask Application
Demonstrates how user comments can poison a RAG vector store.

This version uses a simple TF-IDF based similarity search for offline use.
"""

import re
import math
from collections import Counter
from flask import Flask, render_template, request, jsonify
import threading

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


class SimpleVectorStore:
    """A simple TF-IDF based document store for offline use."""

    def __init__(self):
        self.documents = []
        self.tfidf_vectors = []
        self.idf = {}

    def _tokenize(self, text):
        """Simple tokenization."""
        return re.findall(r'\b\w+\b', text.lower())

    def _compute_tf(self, tokens):
        """Compute term frequency."""
        counter = Counter(tokens)
        total = len(tokens)
        return {term: count / total for term, count in counter.items()}

    def _compute_idf(self, docs):
        """Compute inverse document frequency."""
        n_docs = len(docs)
        all_terms = set()
        for doc in docs:
            all_terms.update(self._tokenize(doc))

        idf = {}
        for term in all_terms:
            doc_count = sum(1 for doc in docs if term in self._tokenize(doc))
            idf[term] = math.log(n_docs / (1 + doc_count)) + 1
        return idf

    def _compute_tfidf(self, text):
        """Compute TF-IDF vector for text."""
        tokens = self._tokenize(text)
        tf = self._compute_tf(tokens)
        return {term: tf_val * self.idf.get(term, 0) for term, tf_val in tf.items()}

    def _cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors."""
        common_terms = set(vec1.keys()) & set(vec2.keys())
        if not common_terms:
            return 0.0

        dot_product = sum(vec1[t] * vec2[t] for t in common_terms)
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot_product / (mag1 * mag2)

    def add_documents(self, documents):
        """Add documents to the store."""
        self.documents = documents
        self.idf = self._compute_idf(documents)
        self.tfidf_vectors = [self._compute_tfidf(doc) for doc in documents]

    def search(self, query, k=3):
        """Search for most similar documents."""
        query_vec = self._compute_tfidf(query)
        similarities = [
            (i, self._cosine_similarity(query_vec, doc_vec))
            for i, doc_vec in enumerate(self.tfidf_vectors)
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [(self.documents[i], score) for i, score in similarities[:k]]


class RAGSystem:
    """Manages the RAG system with vector store."""

    def __init__(self):
        self.vectorstore = SimpleVectorStore()
        self._rebuild_vectorstore()

    def _rebuild_vectorstore(self):
        """Rebuild the vector store with clean docs + comments."""
        all_docs = CLEAN_DOCUMENTS.copy()
        all_docs.extend(comments_store)
        self.vectorstore.add_documents(all_docs)

    def add_comment(self, comment: str):
        """Add a comment to the store and rebuild."""
        comments_store.append(comment)
        self._rebuild_vectorstore()

    def refresh(self):
        """Clear comments and rebuild with clean docs only."""
        comments_store.clear()
        self._rebuild_vectorstore()

    def ask(self, query: str) -> str:
        """Query the RAG system and generate a response."""
        results = self.vectorstore.search(query, k=3)

        if not results or results[0][1] < 0.1:
            return "I don't have enough information to answer that question."

        # Build context from retrieved documents
        context = "\n".join([doc for doc, score in results])

        # Simple response generation based on context
        return self._generate_response(query, context, results)

    def _generate_response(self, query, context, results):
        """Generate a response based on retrieved context."""
        query_lower = query.lower()

        # Check for specific question patterns
        if "whitelist" in query_lower or "trust" in query_lower:
            for doc, score in results:
                if "whitelist" in doc.lower() or "trust" in doc.lower():
                    if "don't" in doc.lower() or "do not" in doc.lower() or "phishing" in doc.lower():
                        return f"No, you should not. {doc}"
                    elif "whitelist" in doc.lower() and "secure" in doc.lower():
                        return f"Based on available information: {doc}"
            return f"Based on security best practices: {results[0][0]}"

        if "what is" in query_lower or "what's" in query_lower:
            return f"Based on available information: {results[0][0]}"

        if "should" in query_lower:
            return f"According to security guidelines: {results[0][0]}"

        # Default: return most relevant context
        return f"Based on the knowledge base: {results[0][0]}"

    def get_stats(self) -> dict:
        """Get current system stats."""
        return {
            "clean_docs": len(CLEAN_DOCUMENTS),
            "comments": len(comments_store),
            "total_docs": len(CLEAN_DOCUMENTS) + len(comments_store)
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

