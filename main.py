
import os
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings import OpenAIEmbeddings
import constants


if __name__ == "__main__":

    persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
    # os.environ["OPENAI_API_KEY"] = constants.APIKEY

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

   # load from disk
    db_chroma = Chroma(persist_directory=persist_directory,  embedding_function=OpenAIEmbeddings())
   
    # Use Ollama for language modeling
    ollama = Ollama(model="mistral")
    # Example question
    question = "what is my name?"

    # Get document similarities based on the question
    docs = db_chroma.similarity_search(question)
    print(len(docs))
    retriever = db_chroma.as_retriever()
    # Create RetrievalQA chain
    qachain = RetrievalQA.from_chain_type(ollama, retriever=retriever,  chain_type = "stuff", return_source_documents=True)

    # Get answers to the question using the RetrievalQA chain
    answers = qachain({"query": question})
    print('Answer :',answers['result'])
