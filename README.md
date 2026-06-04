# Technical Support AI Bot for Upwork API

This project is a Retrieval-Augmented Generation (RAG) Streamlit app that answers developer questions using only the provided Upwork API documentation.

## Features

- Loads the provided Upwork API PDF.
- Prints a sanity check with total character count and a text sample.
- Splits documentation into 500-character chunks with 50-character overlap.
- Stores local embeddings in FAISS using `sentence-transformers/all-MiniLM-L6-v2`.
- Retrieves the top 3 relevant chunks for each question.
- Calls the DeepInfra OpenAI-compatible chat endpoint.
- Shows the generated answer, exact source snippets, and API latency.
- Uses a hallucination guard: if the retrieved snippets do not contain the answer, the model must say the required fallback sentence.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in `DEEPINFRA_API_KEY`.
4. Put the Upwork API PDF somewhere local and set `UPWORK_API_DOC_PATH` in `.env`.

## Build The Vector Store

```powershell
python ingest.py
```

## Run The App

```powershell
streamlit run app.py
```

## Run The Evaluation Helper

```powershell
python evaluate.py
```

The evaluation helper asks the three required ground-truth questions and prints answers with source snippets.
