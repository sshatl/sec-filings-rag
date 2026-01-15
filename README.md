# SEC Filings RAG Assistant

This project implements a **private Retrieval-Augmented Generation (RAG) system**
over U.S. SEC filings (10-K, 10-Q) to answer questions using **source-grounded LLM responses**.

The system ingests real-world financial disclosures, indexes them in a vector database,
and generates answers **strictly based on retrieved source documents**, reducing hallucinations.

---

## 🔍 What This Project Does

- Downloads public SEC filings for a given company
- Parses and cleans raw HTML documents
- Splits documents into semantic chunks
- Generates embeddings using OpenAI
- Indexes content in a persistent vector database (ChromaDB)
- Answers user questions with **citations to original filings**

---

## 🧠 Architecture Overview
```text
SEC Filings (HTML)
  -> Parsing & Cleaning
  -> Chunking
  -> Embeddings
  -> ChromaDB
  -> Retrieval
  -> LLM Answer + Citations

```

---

## 📦 Tech Stack

- **Python**
- **OpenAI API** (embeddings + chat completions)
- **ChromaDB** (persistent vector store)
- **BeautifulSoup** (HTML parsing)
- **SEC EDGAR public data**

---

### 📁 Project Structure

```text
sec-rag/
  data/
    raw/            # downloaded filings (excluded from git)
  chroma/           # persistent vector DB (excluded from git)
  src/
    sec_download.py
    sec_ingest.py
    chroma_index_openai.py
    search_openai.py
    ask.py
  requirements.txt
  .env.example
  .gitignore

```
## 🚀 How to Run

### 1. Setup environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a .env file in the project root (see .env.example):
```text
OPENAI_API_KEY=your_openai_key_here
OPENAI_EMBED_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
SEC_USER_AGENT=YourName your@email.com
```
⚠️ Never commit .env. API keys must remain private.

### 3. Download SEC filings

Download recent SEC filings (10-K / 10-Q) for a company:

```bash
python src/sec_download.py --ticker TSLA --limit 10
```
### 4. Parse and chunk documents

Convert raw HTML filings into cleaned, chunked text:

```bash
python src/sec_ingest.py
```
### 5. Build vector index (OpenAI embeddings)

Generate embeddings and store them in a persistent ChromaDB index:
```bash
python src/chroma_index_openai.py
```
### 6. Ask questions (RAG)

Query the system using Retrieval-Augmented Generation:
```bash
python src/ask.py "What does Tesla say about Cybertruck production ramp?"
```
The assistant will return a grounded answer with citations to the original SEC filings.

Example output:
```text
Tesla states that it is ramping production of the Cybertruck by expanding
manufacturing capacity at its Gigafactories and improving production efficiency [1][2].
```

With cited sources:

TSLA 10-Q (2024-09-30)

TSLA 10-K (2024-12-31)

---

## 🛡️ Hallucination Control

The assistant is explicitly instructed to:

- Answer only using retrieved sources

- Return "Not found in provided sources" if information is missing

- Cite every factual statement

---

## 📈 Possible Improvements

- Section-aware chunking (e.g. Item 1A, Item 7)

- Hybrid search (BM25 + vectors)

- Reranking (cross-encoder)

- Evaluation pipeline with benchmark questions

- Web or API interface

---

## 📌 Use Cases

- Financial research assistants

- Compliance & regulatory analysis

- Internal document search

- Private knowledge-base chatbots