# Retail Sales (LangGraph + Streamlit)

**Linear Extract → Transform → Load pipeline + Retrieval-Augmented Generation**

A practical retail sales data-cleaning application that:

1. **Extracts** a messy CSV with pandas  
2. **Transforms** it (deduplication, missing-value handling, date/product standardization, validation, total-sales calculation)  
3. **Loads** a clean, analytics-ready CSV  
4. **Builds a RAG knowledge base** from the cleaned rows and answers natural-language questions using only retrieved sales records  

Built with **LangGraph** (ETL), **LangChain + Chroma + HuggingFace embeddings + Groq** (RAG), **Streamlit** (UI), and **LangSmith** (optional tracing).

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Extract   │────▶│  Transform   │────▶│    Load     │
│  (pandas)   │     │ (clean/val)  │     │  (to CSV)   │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                                ▼
                                     ┌─────────────────────┐
                                     │  RAG Knowledge Base │
                                     │  (Chroma + embeds)  │
                                     └──────────┬──────────┘
                                                │
                                                ▼
                                     Retail Sales Assistant
                                     (retrieve → Groq LLM)
```

- **State** (`src/state.py`): TypedDict carrying `raw_df`, `cleaned_df`, `stats`, `errors`.
- **Nodes** live in `src/nodes/` (unchanged).
- **Graph** (`src/graph.py`): linear three-node `StateGraph` (unchanged).
- **RAG** (`rag.py` + `prompts.py`): documents → embeddings → Chroma → similarity search → Groq answer.
- **UI** (`app.py`): upload → preview → run ETL → metrics → download → ask questions.

---

## Folder Structure

```
retail-sales-etl-lab/
├── app.py                  # Streamlit entry point (ETL + RAG UI)
├── rag.py                  # NEW – document builder, Chroma, retriever, QA chain
├── prompts.py              # NEW – RAG system & human prompt templates
├── requirements.txt
├── .env.example            # LangSmith + GROQ_API_KEY
├── .gitignore
├── README.md
├── sample_data/
│   └── messy_sales.csv
├── chroma_db/              # created at runtime (persistent vector store)
├── src/
│   ├── __init__.py
│   ├── state.py
│   ├── graph.py
│   └── nodes/
│       ├── __init__.py
│       ├── extract.py
│       ├── transform.py
│       └── load.py
└── tests/
    ├── __init__.py
    ├── test_extract.py
    ├── test_transform.py
    └── test_graph.py
```

---

## Local Setup (Windows)

### 1. Create the project folder

In **File Explorer** or **VS Code**:

```text
mkdir retail-sales-etl-lab
cd retail-sales-etl-lab
```

(Or simply clone / copy the whole folder you received.)

### 2. Create a virtual environment

**Command Prompt / PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` in the prompt.

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure environment variables

```powershell
copy .env.example .env
```

Open `.env` and fill in:

```env
# LangSmith (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_your_real_key_here
LANGCHAIN_PROJECT=retail-sales-etl-lab

# Groq (required for the RAG assistant)
GROQ_API_KEY=gsk_your_real_key_here
```

> **Security note**: Any key that ever appears in chat, screenshots or logs must be rotated before you deploy.

### 5. Run the app

```powershell
streamlit run app.py
```

Your browser opens at `http://localhost:8501`.  
Upload a CSV (or tick “Use built-in sample data”) → **Run ETL Pipeline** → download the cleaned file → ask questions in the **Retail Sales Assistant**.

---

## Testing

With the virtual environment activated:

```powershell
pytest tests/ -v
```

Expected output (approximate):

```
tests/test_extract.py::test_extract_success PASSED
tests/test_extract.py::test_extract_missing_file PASSED
tests/test_extract.py::test_extract_no_path PASSED
tests/test_transform.py::test_transform_basic PASSED
tests/test_transform.py::test_transform_empty PASSED
tests/test_transform.py::test_transform_negative_qty_and_price PASSED
tests/test_graph.py::test_full_pipeline PASSED
```

To also see coverage:

```powershell
pytest tests/ --cov=src --cov-report=term-missing
```

---

## GitHub Deployment

### 1. Initialize Git (inside the project root)

```powershell
git init
git add .
git commit -m "Initial commit: Lab 1 – Retail Sales ETL with LangGraph + Streamlit"
```

### 2. Create a new repository on GitHub

- Go to https://github.com/new  
- Name it e.g. `retail-sales-etl-lab`  
- **Do not** initialize with README (you already have one)  
- Click **Create repository**

### 3. Push

```powershell
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/retail-sales-etl-lab.git
git push -u origin main
```

(Replace `YOUR_USERNAME` with your GitHub handle.)

---

## Streamlit Community Cloud Deployment (recommended)

Streamlit Community Cloud is the simplest and most reliable place to host a persistent Streamlit app.

1. Go to https://share.streamlit.io and sign in with GitHub.  
2. Click **New app**.  
3. Select the repository, branch (`main`), and main file path: `app.py`.  
4. Under **Advanced settings → Secrets** paste the same variables you use in `.env`:

   ```toml
   LANGCHAIN_TRACING_V2 = "true"
   LANGCHAIN_API_KEY = "lsv2_pt_your_real_key_here"
   LANGCHAIN_PROJECT = "retail-sales-etl-lab"
   GROQ_API_KEY = "gsk_your_real_key_here"
   ```

5. Click **Deploy**.  

The app will be available at a public URL like  
`https://YOUR_APP_NAME.streamlit.app`.

---

## Vercel Notes (optional only)

**Vercel does not natively host long-running Streamlit servers.**  
You can still use Vercel for:

- A static landing / marketing page that links to your Streamlit Cloud app.  
- An alternative architecture (e.g. FastAPI + React) – **not** covered in this lab.

If you later want a pure serverless approach you would need to rewrite the UI as a Next.js/React front-end talking to a separate API; that is outside the scope of Lab 1.

---

## RAG Architecture

After a successful ETL run the cleaned DataFrame is converted into LangChain `Document`s and indexed.

### 1. Documents
Every cleaned row becomes one document. Page content looks like:

```
Order ID: ORD-0003. Date: 2024-01-16. Product: MacBook Pro. Category: Laptops.
Quantity: 1. Unit Price: 2499.0. Total Sales: 2499.0. Region (Store): S02. Customer: C003.
```

Metadata stores the same fields for filtering / display.

### 2. Embeddings
`HuggingFaceEmbeddings` with the local model `sentence-transformers/all-MiniLM-L6-v2` (no external embedding API required).

### 3. Vector store – ChromaDB
Persistent directory `./chroma_db`. Collection name: `retail_sales`.  
On each new ETL run the previous collection is cleared and rebuilt so the index always matches the latest cleaned data.

### 4. Similarity search
Top-k (default 6) documents are retrieved via cosine similarity for every user question.

### 5. Question answering
A Groq `llama-3.1-8b-instant` chat model receives the retrieved context plus the question.  
The system prompt forces the model to answer **only** from that context; otherwise it must reply exactly:

```
The uploaded sales dataset does not contain enough information to answer this question.
```

LangSmith continues to trace both the LangGraph ETL graph and the LangChain RAG chain when the corresponding env vars are set.

### New files explained

| File | Role |
|------|------|
| `rag.py` | Document builder, embedding model, Chroma build/load, retriever, QA chain, clear helper |
| `prompts.py` | System + human prompt templates that enforce grounded answers |
| `chroma_db/` | Runtime folder for the persistent vector store (git-ignored) |

Existing ETL modules under `src/` were **not** modified.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: No module named 'src'` | Wrong working directory | Always run `streamlit run app.py` from the project root. |
| `FileNotFoundError` for sample data | Path resolution | Ensure `sample_data/messy_sales.csv` sits next to `app.py`. |
| LangSmith traces missing | Env vars wrong | Check `.env` / Streamlit secrets; restart the app. |
| `GROQ_API_KEY is not set` | Missing key | Add it to `.env` or Streamlit secrets (see `.env.example`). |
| RAG answer is the fallback sentence | Question outside the data | Expected behaviour – the model is forced to stay grounded. |
| Slow first RAG build | Downloading the embedding model | One-time download of `all-MiniLM-L6-v2`; subsequent runs are fast. |
| `UnicodeDecodeError` on upload | Non-UTF-8 CSV | Extract node falls back to latin-1; re-save as UTF-8 if needed. |
| Download button does nothing | Browser blocker | Allow downloads for localhost / the Streamlit domain. |

---

## What the complete project demonstrates

| Technology | Where it appears |
|------------|------------------|
| **LangGraph** | Linear three-node ETL graph (`src/graph.py`) |
| **ETL** | Extract → Transform → Load nodes |
| **Pandas** | CSV I/O, cleaning, validation, aggregations |
| **Streamlit** | Upload, metrics, download, chat-style Q&A UI |
| **LangChain** | Documents, retriever, prompt template, LCEL chain |
| **ChromaDB** | Persistent vector store for sales embeddings |
| **Embeddings** | Local Sentence-Transformer via HuggingFace |
| **Similarity Search** | Top-k cosine retrieval over cleaned rows |
| **RAG** | Retrieve → ground LLM → answer only from data |
| **LangSmith** | Optional tracing of both ETL graph and RAG chain |
| **Groq** | Fast open-source LLM for grounded answers |

---

## Final Checklist

- [ ] All files present (including new `rag.py` and `prompts.py`)  
- [ ] `python -m venv .venv` + `pip install -r requirements.txt` succeeds  
- [ ] `pytest tests/ -v` – existing ETL tests still green  
- [ ] `streamlit run app.py` – sample data runs end-to-end, produces downloadable CSV, and RAG assistant answers questions  
- [ ] `.env` created from `.env.example` (real keys never committed)  
- [ ] `GROQ_API_KEY` set for the assistant  
- [ ] Git repo initialized, committed, pushed to GitHub  
- [ ] Streamlit Community Cloud app deployed with secrets configured  
- [ ] Any API key that ever appeared in chat has been rotated  

---

**Made for learning.**  
Rotate keys, never commit secrets, and happy building!
