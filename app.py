from flask import Flask, request, render_template, jsonify
import os
import uuid
import traceback

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import GPT4All
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# === Setup Flask ===
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Keep local processing bounded and reject unexpectedly large uploads.
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024
MODEL_PATH = os.getenv('GPT4ALL_MODEL_PATH', './models/ggml-gpt4all-j-v1.3-groovy.bin')

# === Global variables ===
db = None
qa_chain = None
chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# === Routes ===
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_pdf():
    global db, qa_chain, chat_memory

    file = request.files.get('pdf')
    if not file or not file.filename:
        return jsonify({"error": "Choose a PDF to upload."}), 400
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files are supported."}), 400
    if not os.path.isfile(MODEL_PATH):
        return jsonify({"error": "GPT4All model not found. Configure GPT4ALL_MODEL_PATH first."}), 503

    # --- Save PDF safely ---
    try:
        filename = str(uuid.uuid4()) + '.pdf'
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save PDF: {str(e)}"}), 500

    try:
        # --- Load PDF ---
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
    except Exception as e:
        return jsonify({"error": f"Failed to read PDF: {str(e)}"}), 500

    try:
        # --- Split documents ---
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(documents)

        # --- Embeddings ---
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # --- Create FAISS vectorstore (in-memory) ---
        db = FAISS.from_documents(docs, embeddings)

        # --- Load GPT4All model properly ---
        llm = GPT4All(
            model=MODEL_PATH,
            model_type="ggml"
        )

        # Patch generate() to avoid unsupported 'max_tokens'
        original_generate = llm.client.generate
        def safe_generate(prompt, **kwargs):
            kwargs.pop("max_tokens", None)
            return original_generate(prompt, **kwargs)
        llm.client.generate = safe_generate

        # --- Conversational QA Chain ---
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=db.as_retriever(),
            memory=chat_memory
        )

        return jsonify({"message": "PDF processed successfully. Chat is ready."})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "The PDF could not be processed. Check the server log for details."}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.route('/chat', methods=['POST'])
def chat():
    global qa_chain
    if not qa_chain:
        return jsonify({'response': '⚠️ Please upload a PDF first.'}), 400

    payload = request.get_json(silent=True) or {}
    query = payload.get('message', '').strip()
    if not query:
        return jsonify({'response': '⚠️ Empty question'}), 400

    try:
        result = qa_chain.run(query)
        return jsonify({'response': result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'response': 'The question could not be processed. Check the server log.'}), 500


if __name__ == '__main__':
    print("🚀 Starting DocuMind AI Flask server...")
    print("📂 Upload folder:", UPLOAD_FOLDER)
    print("🌐 Open in browser: http://127.0.0.1:5000/")
    app.run(debug=os.getenv('FLASK_DEBUG') == '1')
