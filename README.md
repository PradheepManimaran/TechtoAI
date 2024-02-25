Welcome to the Techto AI 


References:

    qdrant implementation:  https https://github.com/AIAnytime/Build-your-first-RAG-using-Qdrant-Vector-Database/blob/main/ingest.py

    Step 1:

        pip install xlrd

        pip install fastapi uvicorn langchain

        pip install sentence-transformers

        pip install qdrant-client

    Step:2

        Command to run:
            docker info
            docker pull qdrant/qdrant 
            docker run -p 6333:6333 -v .:/qdrant/storage qdrant/qdrant
