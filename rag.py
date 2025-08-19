import os
import tempfile
import hashlib
from typing import List, Dict, Any, Optional
import pandas as pd
from pypdf import PdfReader
from pptx import Presentation
from docx import Document
from PIL import Image
import pytesseract
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

class RAGCore:
    def __init__(self, chroma_dir: str, collection_name: str, embed_model: str):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embed_model = embed_model
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        os.makedirs(chroma_dir, exist_ok=True)
        self.chroma = chromadb.Client(Settings(is_persistent=True, persist_directory=self.chroma_dir))
        self.collection = self.chroma.get_or_create_collection(self.collection_name)

    def reset(self):
        self.chroma.delete_collection(self.collection_name)
        self.collection = self.chroma.get_or_create_collection(self.collection_name)

    def _parse_pdf(self, file_bytes: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            reader = PdfReader(tmp.name)
        return "\n\n".join([page.extract_text() or "" for page in reader.pages])

    def _parse_pptx(self, file_bytes: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            pres = Presentation(tmp.name)
        texts = []
        for i, slide in enumerate(pres.slides, start=1):
            slide_text = [shape.text for shape in slide.shapes if hasattr(shape, "text")]
            if slide.notes_slide and slide.notes_slide.notes_text_frame:
                slide_text.append("NOTES: " + slide.notes_slide.notes_text_frame.text)
            if slide_text:
                texts.append(f"[Slide {i}] " + "\n".join(slide_text))
        return "\n\n".join(texts)

    def _parse_csv(self, file_bytes: bytes, filename: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            df = pd.read_csv(tmp.name)
        return f"[Table: {filename}]\n" + df.to_csv(index=False)

    def _parse_docx(self, file_bytes: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            doc = Document(tmp.name)
        texts = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                texts.append(paragraph.text)
        return "\n\n".join(texts)

    def _parse_image(self, file_bytes: bytes, filename: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            image = Image.open(tmp.name)
            # Use OCR to extract text from image
            try:
                text = pytesseract.image_to_string(image)
                return f"[Image: {filename}]\n{text}"
            except Exception as e:
                return f"[Image: {filename}] OCR failed: {str(e)}"

    def _embed(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.embed_model, input=texts)
        return [d.embedding for d in resp.data]

    def index_file(self, file, filename: Optional[str] = None):
        filename = filename or getattr(file, "name", "upload")
        b = file.read()
        digest = sha1_bytes(b)
        ext = filename.split(".")[-1].lower()
        if ext == "pdf":
            text = self._parse_pdf(b)
        elif ext == "pptx":
            text = self._parse_pptx(b)
        elif ext == "csv":
            text = self._parse_csv(b, filename)
        elif ext == "docx":
            text = self._parse_docx(b)
        elif ext in ["png", "jpg", "jpeg", "tiff", "tif"]:
            text = self._parse_image(b, filename)
        else:
            text = b.decode("utf-8", errors="ignore")
        chunks = chunk_text(text)
        metadata = [{"source": filename, "digest": digest, "chunk_index": i} for i, _ in enumerate(chunks)]
        ids = [f"{digest}_{i}" for i in range(len(chunks))]
        embeddings = self._embed(chunks)
        self.collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadata)

    def _bm25_candidates(self, query: str, docs: List[str], k: int = 10) -> List[int]:
        tokenized_corpus = [d.split() for d in docs]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query.split())
        return sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:k]

    def _retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        q_emb = self._embed([query])[0]
        vec_res = self.collection.query(query_embeddings=[q_emb], n_results=top_k*3, include=["metadatas", "documents"])
        docs, metas = vec_res["documents"][0], vec_res["metadatas"][0]
        idx = self._bm25_candidates(query, docs, k=top_k)
        return [{"text": docs[i], "metadata": metas[i]} for i in idx]

    def _llm_answer(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        sys = """You are an expert startup pitch analyst and business consultant with deep expertise in:
        - Business model analysis and validation
        - Market opportunity assessment (TAM/SAM/SOM)
        - Competitive landscape analysis
        - Financial modeling and projections
        - Investment readiness evaluation
        - Go-to-market strategy development
        - Risk assessment and mitigation

        YOUR ROLE: Analyze startup materials and provide strategic business insights that would be valuable to investors, founders, and business stakeholders.

        RESPONSE STRUCTURE:
        1. **Direct Answer**: Address the question using the provided context snippets and web search results when available
        2. **Business Insights**: Extract startup-relevant insights (market size, competitive advantages, business model, etc.)
        3. **Strategic Analysis**: Provide actionable business recommendations when possible
        4. **Citations**: Always cite sources like (filename#chunk) for uploaded documents and web URLs for online sources

        FOCUS AREAS:
        - Market opportunity and size
        - Competitive positioning and differentiation
        - Business model viability and scalability
        - Financial projections and funding needs
        - Go-to-market strategy effectiveness
        - Risk factors and mitigation strategies

        CONTEXT SOURCES:
        - **Uploaded Documents**: Use these as primary sources for startup-specific analysis
        - **Web Search Results**: Supplement with current market data, industry trends, and competitive intelligence when available
        - **Combined Analysis**: Integrate document insights with web research for comprehensive startup analysis

        IMPORTANT: 
        - Prioritize uploaded document context for startup-specific information
        - Use web search results to enhance analysis with current market data and industry insights
        - Always cite your sources clearly (documents vs. web sources)
        - Maintain a professional, investor-ready tone"""
        
        formatted_ctx = []
        for i, c in enumerate(contexts, start=1):
            meta = c.get("metadata", {})
            formatted_ctx.append(f"[{i}] ({meta.get('source','unknown')}#{meta.get('chunk_index',0)})\n{c.get('text','')}")
        ctx_text = "\n\n".join(formatted_ctx) if formatted_ctx else "(no context)"
        prompt = f"Context:\n{ctx_text}\n\nQuestion: {question}\n\nProvide a startup-focused analysis with citations:"
        model = os.getenv("MODEL_NAME", "gpt-4o-mini")
        resp = self.client.chat.completions.create(model=model, messages=[{"role":"system","content":sys},{"role":"user","content":prompt}], temperature=0.2)
        return resp.choices[0].message.content

    def _estimate_coverage(self, answer_text: str, contexts_count: int) -> float:
        used = answer_text.count("(") + answer_text.count("[")
        base = min(1.0, contexts_count / 6.0)
        bonus = min(0.4, used * 0.02)
        return min(1.0, base + bonus)

    def answer(self, question: str, top_k: int = 4, override_contexts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        contexts = override_contexts if override_contexts is not None else self._retrieve(question, top_k=top_k)
        answer = self._llm_answer(question, contexts)
        coverage = self._estimate_coverage(answer, len(contexts))
        sources = list({c.get("metadata",{}).get("source","unknown") for c in contexts})
        return {"answer": answer, "contexts": contexts, "coverage": coverage, "sources": sources}
