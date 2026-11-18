from pathlib import Path

from langchain_core.documents import Document

from rag import format_sources, load_documents, split_documents


def test_load_documents_and_split(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("Personal RAG notes\n" * 100, encoding="utf-8")
    (docs / "ignored.csv").write_text("not supported", encoding="utf-8")

    documents = load_documents(docs)
    chunks = split_documents(documents, chunk_size=100, chunk_overlap=20)

    assert len(documents) == 1
    assert len(chunks) > 1
    assert chunks[0].metadata["source"].endswith("notes.md")


def test_split_rejects_invalid_overlap():
    try:
        split_documents(
            [Document(page_content="text")], chunk_size=10, chunk_overlap=10
        )
    except ValueError as error:
        assert "chunk_size" in str(error)
    else:
        raise AssertionError("Expected invalid chunk configuration to fail")


def test_format_sources_includes_pdf_page():
    source = format_sources(
        [Document(page_content="x", metadata={"source": "paper.pdf", "page": 0})]
    )
    assert source == "paper.pdf (page 1)"
