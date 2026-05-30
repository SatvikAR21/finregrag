# FinReg-RAG 🏦

> A production-grade Retrieval Augmented Generation system for financial regulatory documents — built with hybrid search, cross-encoder reranking, citation enforcement, and automated evaluation.

![CI Pipeline](https://github.com/SatvikAR21/finreg-rag/actions/workflows/eval.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)
![LangChain](https://img.shields.io/badge/LangChain-0.1-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## What This System Does

FinReg-RAG lets compliance officers, analysts, and engineers ask plain-English questions over dense financial regulatory documents and receive **cited, hallucination-free answers** grounded strictly in the source material.

**Example:**
> *"What is the minimum CET1 capital ratio required under Basel III?"*
> → *"Banks must maintain a minimum Common Equity Tier 1 (CET1) ratio of 4.5% of risk-weighted assets. [1]"*
> → Source: Basel III Framework, Page 12

If the documents don't support an answer, the system **refuses to respond** rather than hallucinate.

---

## Evaluation Results

| Metric | Score |
|---|---|
| Overall Evaluation Score | **97.5%** |
| Citation Compliance | **100%** |
| Hallucination Rate | **0%** |
| Refusal Accuracy | **100%** |
| Average End-to-End Latency | **2,512ms** |

*Evaluated against a 10-case RAGAS harness, gated into CI/CD on every push.*

---

## System Architecture
User Question
│
▼
┌─────────────────────────────────────────────┐
│              React Frontend                  │
│  (Query Panel · Source Panel · Analytics)   │
└──────────────────┬──────────────────────────┘
│ HTTP (FastAPI)
▼
┌─────────────────────────────────────────────┐
│           Hybrid Retrieval Layer             │
│   BM25 Keyword Search                        │
│ + ChromaDB Vector Search (MiniLM-L6-v2)     │
│ + Reciprocal Rank Fusion                     │
└──────────────────┬──────────────────────────┘
│
▼
┌─────────────────────────────────────────────┐
│         Cohere Cross-Encoder Reranker        │
└──────────────────┬──────────────────────────┘
│
▼
┌─────────────────────────────────────────────┐
│   Citation-Enforced Generation               │
│   Groq API · Llama 3.3 70B                  │
│   YAML Prompt Management                     │
└──────────────────┬──────────────────────────┘
│
▼
┌─────────────────────────────────────────────┐
│   Observability Layer                        │
│   SQLite + JSONL · Per-Stage Latency Logs   │
└─────────────────────────────────────────────┘

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, Tailwind CSS |
| API Server | FastAPI, Uvicorn |
| Retrieval | ChromaDB, rank-bm25, sentence-transformers |
| Reranking | Cohere Rerank API |
| Generation | Groq API, Llama 3.3 70B |
| Orchestration | LangChain |
| PDF Parsing | PyMuPDF |
| Evaluation | RAGAS |
| Logging | SQLite, JSONL |
| CI/CD | GitHub Actions |

---

## Project Structure
finreg-rag/
├── phase1/                  # PDF ingestion, chunking, embedding, ChromaDB storage
├── phase2/                  # Hybrid retrieval, Cohere reranking, citation enforcement
├── phase3/                  # SQLite/JSONL logging, analytics dashboard
├── phase4/                  # RAGAS evaluation harness, eval dataset, CI pipeline
├── api/                     # FastAPI server wrapping the full pipeline
├── frontend/                # React + Tailwind dashboard
├── data/
│   ├── raw/                 # Source PDFs (Basel III, MiFID II, IFRS 9)
│   ├── processed/           # Parsed JSON + chunk files
│   └── chromadb/            # Vector store
├── logs/                    # Query logs (SQLite + JSONL)
├── .github/workflows/       # GitHub Actions CI pipeline
└── requirements.txt

---

## Regulatory Documents Covered

| Document | Domain | Why It Matters |
|---|---|---|
| Basel III Framework | Capital adequacy | Core banking regulation post-2008 crisis |
| MiFID II Guidelines | Markets in financial instruments | EU investment services regulation |
| IFRS 9 | Financial instruments accounting | Global accounting standard for loan loss provisioning |
| SEC 10-K / 10-Q Filings | Public company disclosures | US securities compliance |
| FINRA Rulebook | Broker-dealer conduct | US financial industry self-regulation |
| FCA Handbook | UK financial conduct | Post-Brexit UK regulatory framework |

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Groq API key (free at [console.groq.com](https://console.groq.com))
- Cohere API key (free at [cohere.com](https://cohere.com))

### 1. Clone and set up environment
```bash
git clone https://github.com/SatvikAR21/finreg-rag.git
cd finreg-rag
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure API keys
```bash
# Create a .env file in the root directory
GROQ_API_KEY=your_groq_key_here
COHERE_API_KEY=your_cohere_key_here
```

### 3. Ingest documents
```bash
python phase1/ingest.py
python phase1/chunk.py
python phase1/embed_and_store.py
```

### 4. Start the API server
```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Start the frontend
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

---

## How Citation Enforcement Works

The system refuses to answer if retrieved chunks do not contain sufficient evidence. This is enforced at the generation stage — the prompt explicitly instructs the model to cite sources using `[1]`, `[2]` markers and to respond with a structured refusal if evidence is absent. The `citation_enforcer.py` module validates every response before it is returned to the user.

---

## CI/CD Pipeline

Every push to `main` triggers a GitHub Actions workflow that:
1. Installs all dependencies
2. Runs the 10-case evaluation harness
3. Fails the build if the overall score drops below 90% or citation compliance drops below 100%

This ensures no regression is ever silently merged.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
