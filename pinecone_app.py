import os
import uuid
import json
import pandas as pd
import docx
from pinecone import Pinecone
from io import StringIO, BytesIO
import openai
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string
import base64

# --- Load environment variables from .env file ---
load_dotenv()

# --- Setup ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY:
    raise EnvironmentError("PINECONE_API_KEY environment variable not set.")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
if not INDEX_NAME:
    raise EnvironmentError("INDEX_NAME environment variable not set.")

pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
print("Pinecone client initialized.")
print("Available indexes:", pinecone_client.list_indexes().names())

pinecone_index = pinecone_client.Index(name=INDEX_NAME)

openai.api_key = OPENAI_API_KEY

# --- Flask App ---
app = Flask(__name__)

# --- Utilities ---
def embed_text(text):
    response = openai.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def create_metadata(file_name, content_type, chunk_id):
    return {
        "source": file_name,
        "type": content_type,
        "chunk_id": chunk_id
    }

def chunk_text(text, max_tokens=8000):
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    words = text.split()
    chunks = []
    current_chunk = []
    token_count = 0

    for word in words:
        word_tokens = len(enc.encode(word))
        if token_count + word_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            token_count = word_tokens
        else:
            current_chunk.append(word)
            token_count += word_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# --- File Handlers ---
def process_csv_file(file_obj, filename):
    df = pd.read_csv(file_obj)
    records = []
    for i, row in df.iterrows():
        text = row.astype(str).str.cat(sep=" ")
        vector = embed_text(text)
        metadata = create_metadata(filename, "csv", i)
        records.append((str(uuid.uuid4()), vector, metadata))
    return records

def process_json_file(file_obj, filename):
    data = json.load(file_obj)
    if isinstance(data, dict):
        data = [data]
    records = []
    for i, item in enumerate(data):
        text = json.dumps(item)
        chunks = chunk_text(text)
        for j, chunk in enumerate(chunks):
            vector = embed_text(chunk)
            metadata = create_metadata(filename, "json", f"{i}-{j}")
            records.append((str(uuid.uuid4()), vector, metadata))
    return records

def process_txt_file(file_obj, filename):
    content = file_obj.read().decode("utf-8").strip()
    chunks = chunk_text(content)
    records = []
    for i, chunk in enumerate(chunks):
        vector = embed_text(chunk)
        metadata = create_metadata(filename, "txt", i)
        records.append((str(uuid.uuid4()), vector, metadata))
    return records

def process_docx_file(file_obj, filename):
    doc = docx.Document(file_obj)
    full_text = "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
    chunks = chunk_text(full_text)
    records = []
    for i, chunk in enumerate(chunks):
        vector = embed_text(chunk)
        metadata = create_metadata(filename, "docx", i)
        records.append((str(uuid.uuid4()), vector, metadata))
    return records

def process_xlsx_file(file_obj, filename):
    xls = pd.read_excel(file_obj, sheet_name=None)
    records = []
    for sheet_name, df in xls.items():
        for i, row in df.iterrows():
            text = row.astype(str).str.cat(sep=" ")
            vector = embed_text(text)
            metadata = {
                "source": filename,
                "type": "xlsx",
                "chunk_id": f"{sheet_name}-{i}",
                "sheet": sheet_name
            }
            records.append((str(uuid.uuid4()), vector, metadata))
    return records

# --- Upload Endpoint ---
@app.route('/upload', methods=['POST'])
def upload_from_gpt_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file_obj = request.files['file']
    filename = file_obj.filename
    ext = os.path.splitext(filename)[1].lower()
    try:
        if ext == '.csv':
            records = process_csv_file(file_obj, filename)
        elif ext == '.json':
            records = process_json_file(file_obj, filename)
        elif ext == '.txt':
            records = process_txt_file(file_obj, filename)
        elif ext == '.docx':
            records = process_docx_file(file_obj, filename)
        elif ext == '.xlsx':
            records = process_xlsx_file(file_obj, filename)
        else:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400

        for i in range(0, len(records), 100):
            try:
                print("Sample record structure:", records[0] if records else 'No records to upload')
                pinecone_index.upsert(vectors=records[i:i+100])
            except Exception as e:
                print("[ERROR] During upsert:", e)
                return jsonify({"error": f"Upsert failed: {str(e)}"}), 500

        return jsonify({"message": f"Uploaded {len(records)} vectors from {filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Delete Endpoint ---
@app.route('/delete', methods=['POST'])
def delete_vectors_by_filename():
    data = request.get_json()
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "Filename is required."}), 400
    try:
        results = pinecone_index.query(
            vector=[0.0] * 1536,
            top_k=1000,
            include_metadata=True,
            filter={"source": filename}
        )
        ids_to_delete = [match["id"] for match in results["matches"]]
        if not ids_to_delete:
            return jsonify({"message": f"No vectors found for file '{filename}'"})
        pinecone_index.delete(ids=ids_to_delete)
        return jsonify({"message": f"Deleted {len(ids_to_delete)} vectors from '{filename}'"})
    except Exception as e:
        return jsonify({"error": f"Delete failed: {str(e)}"}), 500


# --- Upload Form UI ---
@app.route('/', methods=['GET'])
def index():
    with open("static/Foresight_Logo.PNG", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>QuoteFusion: Pinecone Interface</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(to bottom right, #f0f4f8, #d9e2ec);
                margin: 0;
                padding: 0;
            }
            .container {
                max-width: 700px;
                margin: 4em auto;
                padding: 2.5em;
                background: white;
                border-radius: 12px;
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
            }
            h1, h2, h3 {
                color: #003366;
            }
            input[type="file"], input[type="text"] {
                width: 100%;
                padding: 0.75em;
                margin: 1em 0;
                border: 1px solid #ccc;
                border-radius: 6px;
            }
            button {
                background-color: #0056b3;
                color: white;
                padding: 0.75em 1.5em;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                transition: background 0.3s ease;
            }
            button:hover {
                background-color: #004494;
            }
            #message {
                margin-top: 1.5em;
                padding: 1em;
                border-radius: 8px;
                font-weight: bold;
            }
            .success {
                background-color: #d4edda;
                color: #155724;
            }
            .error {
                background-color: #f8d7da;
                color: #721c24;
            }
            .dark-mode .container {
                background: #2d2d3f;
                box-shadow: 0 8px 20px rgba(255, 255, 255, 0.05);
            }
            .dark-mode input, .dark-mode button {
                background-color: #444;
                color: #fff;
                border-color: #666;
            }
            .dark-mode .success {
                background-color: #2e5134;
            }
            .dark-mode .error {
                background-color: #5b2e2e;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <img src="data:image/png;base64,{{ logo_base64 }}" alt="Foresight Logo" style="max-width: 220px; display: block; margin: 0 auto 1.5em auto;" />
            <h1>QuoteFusion: Pinecone Interface</h1>
            <p>Upload files to be indexed or delete previously uploaded data by filename.</p>
            <h2>Upload Documents</h2>
            <form id="uploadForm">
                <input type="file" name="file" id="file" multiple required />
                <button type="submit">Upload</button>
            </form>
            <div id="message"></div>
            <hr>
            <h3>Delete Vectors by Filename</h3>
            <input type="text" id="deleteFilename" placeholder="Enter filename (e.g., example.txt)" style="width:100%; padding:0.5em; margin-bottom:1em;" />
            <button id="deleteBtn">Delete</button>
        </div>
        <script>
            const form = document.getElementById("uploadForm");
            const messageBox = document.getElementById("message");
            form.addEventListener("submit", async (e) => {
                e.preventDefault();
                messageBox.innerHTML = "";
                const files = document.getElementById("file").files;
                if (files.length === 0) {
                    showMessage("Please select at least one file.", "error");
                    return;
                }
                for (const file of files) {
                    const formData = new FormData();
                    formData.append("file", file);
                    try {
                        const response = await fetch("/upload", {
                            method: "POST",
                            body: formData
                        });
                        let result;
                        try {
                            result = await response.json();
                        } catch (parseErr) {
                            const text = await response.text();
                            showMessage("Server returned invalid JSON response.", "error");
                            return;
                        }
                        if (!response.ok) {
                            showMessage(`Error: ${result.error || "Unknown error."}`, "error");
                        } else {
                            showMessage(`Success: ${result.message}`, "success");
                        }
                    } catch (err) {
                        showMessage(`Error: ${err.message}`, "error");
                    }
                }
            });
            function showMessage(message, type) {
                const div = document.createElement("div");
                div.className = type;
                div.textContent = message;
                messageBox.appendChild(div);
            }
            document.getElementById("deleteBtn").addEventListener("click", async () => {
                const filename = document.getElementById("deleteFilename").value.trim();
                messageBox.innerHTML = "";
                if (!filename) {
                    showMessage("Please enter a filename.", "error");
                    return;
                }
                try {
                    const response = await fetch("/delete", {
                        method: "POST",
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ filename })
                    });
                    const result = await response.json();
                    if (!response.ok) {
                        showMessage(`Error: ${result.error || "Unknown error."}`, "error");
                    } else {
                        showMessage(`Success: ${result.message}`, "success");
                    }
                } catch (err) {
                    showMessage(`Error: ${err.message}`, "error");
                }
            });
        </script>
    </body>
    </html>
    ''', logo_base64=encoded_string)

# --- Main ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
    
