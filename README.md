# Personal RAG Assistant

A local document question-answering assistant based on the referenced `03-personal-rag-assistant` project. It indexes Markdown, text, and PDF files with Chroma and local Hugging Face embeddings, then answers questions with Claude while retaining the last ten conversation turns.

## Setup

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `ANTHROPIC_API_KEY` in `.env`, then place `.md`, `.txt`, or `.pdf` files in `docs/`.

## Run

```bash
python rag.py
```

Useful options:

```bash
python rag.py --docs ./docs --db ./chroma_db --top-k 4
```

Answers include the source file and PDF page where available. The embedding model and vector database are local; only the question/answer call uses Anthropic.

## Test and lint

```bash
pytest -q
ruff check .
ruff format --check .
```

The first real indexing run downloads the configured embedding model and creates the ignored `chroma_db/` directory.
