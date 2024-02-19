
from chromadb.config import Settings 

CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory='db',
        anonymized_telemetry=False
)

APIKEY = "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"
