import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_pdf_pages(data, max_pages=300):
    import fitz  # PyMuPDF
    doc = fitz.open(stream=data, filetype="pdf")
    pages = [{"page": i + 1, "text": doc[i].get_text().strip()} for i in range(min(len(doc), max_pages))]
    if not any(p["text"] for p in pages):
        raise ValueError("No extractable text: scanned/image-only PDF. OCR is not available in this deployment.")
    return pages

def extract_docx(data):
    import docx
    return [{"page": 1, "text": "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)}]

class Index:
    def __init__(self, pages):
        self.pages = [p for p in pages if p["text"].strip()]
        if not self.pages: raise ValueError("no text to index")
        self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self.mat = self.vec.fit_transform([p["text"] for p in self.pages])

    def search(self, q, k=3, min_score=0.05):
        s = cosine_similarity(self.vec.transform([q]), self.mat).ravel()
        return [{"page": self.pages[i]["page"], "score": float(s[i]), "text": self.pages[i]["text"][:1200]}
                for i in s.argsort()[::-1][:k] if s[i] >= min_score]  # [] means the document lacks the answer

def key_terms(pages, k=10):
    v = TfidfVectorizer(stop_words="english", max_features=2000, ngram_range=(1, 2))
    m = v.fit_transform([p["text"] for p in pages if p["text"].strip()])
    sc = m.sum(axis=0).A1; t = v.get_feature_names_out()
    return [t[i] for i in sc.argsort()[::-1][:k]]

def extractive_summary(pages, n=5):
    """Local summary: top-scoring real sentences, each tagged with its page. No model involved."""
    import re
    sents = [(p["page"], s.strip()) for p in pages for s in re.split(r"(?<=[.!?])\s+", p["text"]) if len(s.split()) >= 6]
    if not sents: raise ValueError("not enough text to summarise")
    v = TfidfVectorizer(stop_words="english"); m = v.fit_transform([s for _, s in sents])
    sc = m.sum(axis=1).A1 / [len(s.split()) ** .5 for _, s in sents]
    return [sents[i] for i in sorted(sc.argsort()[::-1][:n])]

def similarity_matrix(texts):
    if len(texts) < 2: raise ValueError("need at least 2 texts")
    return cosine_similarity(TfidfVectorizer(stop_words="english").fit_transform(texts))

def validate_upload(data, name, max_mb=15, max_pages=300):
    if len(data) > max_mb * 1048576: raise ValueError(f"file exceeds {max_mb} MB")
    if name.lower().endswith(".pdf") and data[:4] != b"%PDF": raise ValueError("not a valid PDF")
