from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from langchain.document_loaders import PyPDFLoader, CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os
from constants import CHROMA_SETTINGS
import logging
from typing import List
from langchain_community.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA


app = FastAPI()

persist_directory = "db"
logger = logging.getLogger(__name__)

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    try:
        upload_dir = "docs"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        return JSONResponse(content={"message": "File uploaded successfully", "filename": file.filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while uploading file: {str(e)}")

@app.get("/load_docs/")
async def load_documents():
    try:
        docs_dir = "docs"
        logger.info(f"Loading documents from directory: {docs_dir}")
        documents = []
        for root, _, files in os.walk(docs_dir):
            for file in files:
                if file.endswith((".pdf", ".csv", ".txt", ".xls", ".xlsx", ".docx")):
                    try:
                        loader = get_loader(file)
                        document = loader(os.path.join(root, file)).load()
                        logger.info(f"Document loaded successfully: {file}")
                        documents.append(document)
                    except Exception as e:
                        logger.error(f"Error loading document {file}: {e}")

        # Call process_documents function here
        result = process_documents(documents)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while loading documents: {str(e)}")
    
def get_loader(file):
    if file.endswith(".pdf"):
        return PDFMinerLoader
    elif file.endswith(".csv"):
        return CSVLoader
    elif file.endswith((".txt", ".log")):
        return TextLoader
    elif file.endswith((".xls", ".xlsx")):
        return UnstructuredExcelLoader
    elif file.endswith(".docx"):
        return Docx2txtLoader
    else:
        raise ValueError(f"Unsupported file type: {file}")

def process_documents(documents: List[str]):
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    # Load and split documents
    texts = []
    for document in documents:
        try:
            text = text_splitter.split_documents(document)
            texts.extend(text)
        except Exception as e:
            logger.error(f"Error splitting document: {e}")

    if not texts:
        logger.warning("No text extracted from the documents.")
        return None

    # Load Sentence Transformers model
    embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create vector store
    logger.info("Creating embeddings. May take some minutes...")
    db = Chroma.from_documents(
        texts, embeddings_model, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS
    )
    db.persist()
    db = None

    logger.info("Ingestion completed")
    return {"message": "Ingestion completed"}


@app.get("/answer/")
async def get_answer(question: str):
    
    persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
    embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load from disk
    db_chroma = Chroma(persist_directory=persist_directory, embedding_function=embeddings_model)

    # Use Ollama for language modeling
    ollama = Ollama(model="mistral")
    """
    Endpoint to get the answer to a question.
    """
    try:
        # Get document similarities based on the question
        docs = db_chroma.similarity_search(question)
        retriever = db_chroma.as_retriever()
        # Create RetrievalQA chain
        qachain = RetrievalQA.from_chain_type(ollama, retriever=retriever, chain_type="stuff",
                                               return_source_documents=True)

        # Get answers to the question using the RetrievalQA chain
        answers = qachain({"query": question})
        return {"answer": answers['result']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)