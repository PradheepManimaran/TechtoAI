from langchain.document_loaders import PyPDFLoader, CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
from constants import CHROMA_SETTINGS
import logging
import constants
from langchain.embeddings import SentenceTransformerEmbeddings


persist_directory = "db"
logger = logging.getLogger(__name__)
os.environ["OPENAI_API_KEY"] = constants.APIKEY
def main():
    documents = load_documents("docs")

    if not documents:
        print("No supported documents found.")
        return

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    # Load and split documents
    print("Loading and splitting documents")
    texts = []
    for document in documents:
        try:
            text = text_splitter.split_documents(document)
            texts.extend(text)
        except Exception as e:
            print(f"Error splitting document: {e}")

    if not texts:
        print("No text extracted from the documents.")
        return

    # Load Sentence Transformers model
    print("Loading Sentence Transformers model")
    embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create vector store
    print("Creating embeddings. May take some minutes...")
    db = Chroma.from_documents(
        texts, embeddings_model, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS
    )
    db.persist()
    db = None

    print("Ingestion completed")

def load_documents(directory):
    logger.info(f"Loading documents from directory: {directory}")
    documents = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".pdf", ".csv", ".txt", ".xls", ".xlsx", ".docx")):
                try:
                    loader = get_loader(file)
                    document = loader(os.path.join(root, file)).load()
                    logger.info(f"Document loaded successfully: {file}")
                    documents.append(document)
                except Exception as e:
                    logger.error(f"Error loading document {file}: {e}")
    return documents

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

if __name__ == "__main__":
    main()
