import os
import uuid
import shutil
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
# from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
app = FastAPI(title="Simple RAG API")

# Setup directories
UPLOAD_DIR = "./storage/uploads"
CHROMA_DB_DIR = "./storage/chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# Initialize Embeddings (No download, uses API)
# Ubinses HuggingFace Inference API
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
print("HF token:", hf_token)
if not hf_token or hf_token == "your_huggingface_api_token_here":
    print("Warning: HUGGINGFACEHUB_API_TOKEN is not set properly. Embeddings might fail.")

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=hf_token
)

# Initialize Vector Store
vectorstore = Chroma(
    collection_name="documents_collection",
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_DIR
)

# Initialize LLM (Open Source via Groq API)
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key or groq_api_key == "your_groq_api_key_here":
    print("Warning: GROQ_API_KEY is not set properly. LLM queries might fail.")

llm = ChatGroq(
    temperature=0, 
    model_name="Llama-3.1-8B-Instruct", 
    api_key=groq_api_key
)

# Setup prompts
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

class QueryRequest(BaseModel):
    document_id: str
    question: str

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document, extract text, chunk it, and store embeddings in ChromaDB.
    Returns a unique document ID.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    # Generate unique ID for the document
    doc_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    
    # Save file locally
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
        
    # Extract text from PDF
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            # Handle large files up to 200-300 pages seamlessly
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read PDF: {e}")
        
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
        
    # Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Prepare metadata so we can filter by doc_id
    metadatas = [{"document_id": doc_id} for _ in chunks]
    
    # Add to ChromaDB
    try:
        vectorstore.add_texts(texts=chunks, metadatas=metadatas)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index document: {e}")
        
    return {"message": "Document processed and indexed successfully", "document_id": doc_id}

@app.post("/query/")
async def query_document(request: QueryRequest):
    """
    Query a previously uploaded document using its ID.
    Generates an answer based on the retrieved context using Groq LLM.
    """
    # Setup retriever with filter for the specific document
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {"document_id": request.document_id}
        }
    )
    
    # Create QA chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    try:
        response = rag_chain.invoke({"input": request.question})
        return {
            "document_id": request.document_id,
            "question": request.question,
            "answer": response["answer"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
