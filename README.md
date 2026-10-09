# DocuQuery

A single-document RAG application. Upload a PDF, ask questions, and receive answers grounded in retrieved passages with page-level source references.

---

## Architecture

```
PDF upload → extract pages (PyMuPDF)
          → chunk text (sliding window with overlap)
          → embed chunks (Sentence Transformers: all-MiniLM-L6-v2)
          → retrieve top-k (cosine similarity, NumPy)
          → generate answer (Gemini 1.5 Flash)
          → display answer + source passages with page numbers
```

All vectors are stored in application memory — no external database required.

---

## Setup

### 1. Prerequisites

- Python 3.10 or later
- A [Google AI Studio](https://aistudio.google.com/) API key (free tier works)

### 2. Clone the repository

```bash
git clone <repo-url>
cd DocuQuery
```

### 3. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB) and caches it locally.

### 5. Set your API key

```bash
export GEMINI_API_KEY="your-api-key-here"
```

On Windows (PowerShell):

```powershell
$env:GEMINI_API_KEY = "your-api-key-here"
```

### 6. Run the application

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

1. **Upload a PDF** using the sidebar file picker (text-based PDFs only; scanned PDFs are not supported).
2. Wait for the index to build (extraction + chunking + embedding).
3. **Type a question** in the text box.
4. Read the generated answer and expand the **Source Passages** to see which pages it is grounded in.
5. Upload a different PDF at any time — the previous index is replaced automatically.

---

## Screenshots

> *(Add screenshots here after first run)*

---

## Running Tests

```bash
pytest
```

The embedder tests load the real `all-MiniLM-L6-v2` model and take ~15–30 seconds on first run.

---

## Known Limitations

- **Scanned PDFs**: Text extraction requires a machine-readable PDF. OCR is out of scope for this version.
- **Large PDFs**: Very large documents (hundreds of pages) may take noticeable time to embed. No streaming or progress bar is shown for the embedding step beyond the spinner.
- **Single document**: Only one PDF can be active at a time.
- **No conversation memory**: Each question is answered independently; follow-up questions do not reference prior answers.
- **Gemini rate limits**: Free-tier API keys have per-minute token limits. Reduce `top_k` in `embedder.retrieve()` if you hit limits on large documents.

---

## Project Structure

```
DocuQuery/
├── app.py              # Streamlit application
├── rag/
│   ├── extractor.py    # PyMuPDF page extraction
│   ├── chunker.py      # Overlapping word-based chunker
│   ├── embedder.py     # SentenceTransformers + cosine retrieval
│   └── generator.py    # Gemini answer generation
├── tests/
│   ├── test_extractor.py
│   ├── test_chunker.py
│   └── test_embedder.py
├── requirements.txt
├── .gitignore
└── README.md
```
