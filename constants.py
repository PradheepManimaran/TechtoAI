
from chromadb.config import Settings 

CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory='db',
        anonymized_telemetry=False
)

APIKEY = "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"

TEST_KEY = "739fedsfsd779797c115d881bf30"

BASE_URL= "https://genaia.com/openai-chat"