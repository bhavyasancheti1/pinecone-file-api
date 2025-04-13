import os
import uuid
import pandas as pd
import fitz  # PyMuPDF
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# --- Config ---
PINECONE_API_KEY = os.getenv("")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = "quotefusion-main"  # CHANGE THIS

# --- Initialize Pinecone and model ---
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- FastAPI instance ---
app = FastAPI()

# --- Utilities ---
def embed_text(text):
    return model.encode(text).tolist()

def create_metadata(file_name, content_type, chunk_id):
    return {
        "source": file_name,
        "type": content_type,
        "chunk_id": chunk_id
    }

# --- CSV processing ---
def process_csv_file(file_obj, filename):
    df = pd.read_csv(file_obj)
    records = []
    for i, row in df.iterrows():
        text = row.astype(str).str.cat(sep=" ")
        vector = embed_text(text)
        metadata = create_metadata(filename, "csv", i)
        records.append((str(uuid.uuid4()), vector, metadata))
    return records

# --- PDF processing ---
def process_pdf_file(file_obj, filename):
    buffer = BytesIO(file_obj.read())
    buffer.seek(0)
    doc = fitz.open(stream=buffer, filetype="pdf")
    records = []

    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            vector = embed_text(text)
            metadata = create_metadata(filename, "pdf", i)
            records.append((str(uuid.uuid4()), vector, metadata))
    return records

# --- Unified handler ---
def upload_from_gpt_file(file_bytes, filename):
    ext = os.path.splitext(filename)[1].lower()
    file_obj = BytesIO(file_bytes)

    if ext == '.csv':
        records = process_csv_file(file_obj, filename)
    elif ext == '.pdf':
        records = process_pdf_file(file_obj, filename)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    for i in range(0, len(records), 100):
        index.upsert(vectors=records[i:i+100])
    return records

# --- Upload API endpoint ---
@app.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".csv", ".pdf"]:
        raise HTTPException(status_code=400, detail="Only .csv and .pdf files are supported.")

    try:
        await file.seek(0)
        records = upload_from_gpt_file(await file.read(), filename)
        return JSONResponse(content={"message": f"Uploaded {len(records)} chunks from {filename}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Delete API endpoint ---
@app.delete("/delete-vector/{vector_id}")
async def delete_vector(vector_id: str):
    try:
        index.delete(ids=[vector_id])
        return JSONResponse(content={"message": f"Vector {vector_id} deleted from index."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Search API endpoint ---
@app.get("/search-vector")
async def search_vector(q: str = Query(...)):
    try:
        query_vector = embed_text(q)
        result = index.query(vector=query_vector, top_k=5, include_metadata=True)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
