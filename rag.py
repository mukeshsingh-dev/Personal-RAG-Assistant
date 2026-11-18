from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic


SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def load_documents(docs_path: str | Path = "./docs"):
    path = Path(docs_path)
    path.mkdir(parents=True, exist_ok=True)
    documents = []
    for file_path in sorted(path.rglob("*")):
        if (
            not file_path.is_file()
            or file_path.suffix.lower() not in SUPPORTED_SUFFIXES
        ):
            continue
        loader = (
            PyPDFLoader(str(file_path))
            if file_path.suffix.lower() == ".pdf"
            else TextLoader(str(file_path), encoding="utf-8")
        )
        documents.extend(loader.load())
    return documents


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def create_vectorstore(
    chunks, db_path: str | Path = "./chroma_db", embedding_model: str | None = None
):
    if not chunks:
        raise ValueError("At least one document chunk is required")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model
        or os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    return Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=str(db_path),
        collection_name="personal_documents",
    )


def build_chain(vectorstore, top_k: int = 4):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", return_messages=True, k=10
    )
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514", temperature=0.3, api_key=api_key
    )
    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": top_k}),
        memory=memory,
        return_source_documents=True,
    )


def format_sources(documents) -> str:
    sources = []
    for document in documents:
        source = document.metadata.get("source", "unknown source")
        page = document.metadata.get("page")
        label = f"{source} (page {page + 1})" if page is not None else source
        if label not in sources:
            sources.append(label)
    return ", ".join(sources) if sources else "no source metadata"


def run(docs_path: str, db_path: str, embedding_model: str, top_k: int) -> None:
    documents = load_documents(docs_path)
    if not documents:
        print(f"No .pdf, .md, or .txt files found in {Path(docs_path).resolve()}")
        return
    chunks = split_documents(documents)
    vectorstore = create_vectorstore(chunks, db_path, embedding_model)
    chain = build_chain(vectorstore, top_k)
    print(f"Loaded {len(documents)} documents into {len(chunks)} chunks.")
    print("Personal RAG Assistant ready. Type 'quit' to exit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        response = chain.invoke({"question": question})
        print(f"\nAssistant: {response['answer']}")
        print(f"Sources: {format_sources(response.get('source_documents', []))}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Chat with your local documents using retrieval-augmented generation"
    )
    parser.add_argument("--docs", default=os.getenv("RAG_DOCS_PATH", "./docs"))
    parser.add_argument("--db", default=os.getenv("RAG_DB_PATH", "./chroma_db"))
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    )
    parser.add_argument("--top-k", type=int, default=int(os.getenv("RAG_TOP_K", "4")))
    args = parser.parse_args()
    run(args.docs, args.db, args.embedding_model, args.top_k)


if __name__ == "__main__":
    main()
