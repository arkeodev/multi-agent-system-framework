# database_service.py

from langchain.schema import Document
from sqlalchemy import create_engine


def process_database_and_add_to_faiss(connection_string: str, query: str, faiss_index):
    engine = create_engine(connection_string)
    with engine.connect() as connection:
        result = connection.execute(query)
        for row in result:
            content = str(row)
            doc = Document(page_content=content, metadata={"source": "database"})
            faiss_index.add_texts([doc.page_content], metadatas=[doc.metadata])
