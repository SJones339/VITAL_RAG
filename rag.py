"""Student exercise: implement retrieval over the fictional clinic policies."""
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS = Path(__file__).parent / "knowledge_base"


def load_documents(directory=DOCS):
    """Return list of (filename, text) for *.txt files."""
    # TODO 1: Load documents in deterministic filename order.
    docs = []
    for path in sorted(Path(directory).glob("*.txt")):
        docs.append((path.name, path.read_text(encoding="utf-8")))
    return docs


def retrieve(question: str, k: int = 2, documents=None):
    """Return top-k (filename, text, score) matches using TF-IDF."""
    # TODO 2: Vectorize documents and query, rank cosine similarities.
    # Exclude zero-score matches; break ties by filename.
    if documents is None:
        documents = load_documents()
    if not documents or k <= 0:
        return []

    vectorizer = TfidfVectorizer()
    doc_matrix = vectorizer.fit_transform([text for _, text in documents])
    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, doc_matrix).flatten()
    ranked = [
        (name, text, float(score))
        for (name, text), score in zip(documents, scores)
        if score > 0
    ]
    ranked.sort(key=lambda x: (-x[2], x[0]))
    return ranked[:k]


def build_context(results):
    """Format hits with [Source: filename] headers."""
    # TODO 3: Build the prompt context.
    if not results:
        return ""
    return "\n\n".join(
        f"[Source: {filename}]\n{text}" for filename, text, _ in results
    )
