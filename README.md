# DocuMind AI

I built DocuMind to make long PDF notes easier to explore. Instead of searching page by page, you can upload a PDF and ask questions in normal language. The app finds relevant sections and sends that context to a local GPT4All model.

## What happens after an upload

- PDF upload and text extraction with PyMuPDF
- Semantic search with MiniLM embeddings and FAISS
- Conversational question answering with memory
- Fully local language-model inference
- File-type validation, 25 MB upload limit, and automatic upload cleanup
- Responsive Flask interface

## Architecture

```text
PDF → text extraction → chunking → embeddings → FAISS index
                                                ↓
Question → similarity retrieval → GPT4All → contextual answer
```

## Tech stack

- Python and Flask
- LangChain
- FAISS
- Sentence Transformers
- GPT4All
- PyMuPDF

## Quick start

```bash
git clone https://github.com/niharikavemula344-byte/-DocuMind-AI-Intelligent-PDF-Question-Answering-System.git
cd -- -DocuMind-AI-Intelligent-PDF-Question-Answering-System
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Download a GPT4All-compatible model, then configure its path:

```bash
# macOS/Linux
export GPT4ALL_MODEL_PATH="/absolute/path/to/model.bin"

# PowerShell
$env:GPT4ALL_MODEL_PATH="C:\path\to\model.bin"

python app.py
```

Open `http://127.0.0.1:5000`, upload a PDF, and ask questions about its contents.

## Repository structure

```text
.
├── app.py
├── requirements.txt
├── templates/index.html
├── pdf sample/
└── screenshot/
```

## What I learned

This project gave me hands-on experience with text chunking, embeddings, vector search, conversational memory, and the practical limits of local language models.

## Important limitations

- This is a single-user demonstration app; the in-memory vector store is shared by the running process.
- Answers can be incorrect. Verify important claims against the source PDF.
- Do not upload confidential documents to a publicly hosted instance without authentication and isolated storage.

## License

Licensed under the MIT License. See `LICENSE`.
