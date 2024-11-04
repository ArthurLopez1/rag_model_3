# backend/vectorstore/vectorstore_manager.py
import os
import faiss
import numpy as np
from llm_models import HFModel, LLMModel
from file_handler import parse_pdf_with_pypdf
from settings import Config

class VectorStoreManager:
    def __init__(self):
        self.model = HFModel(Config.EMBEDDING_MODEL_NAME, Config.GENERATION_MODEL_NAME)
        self.embeddings = []
        self.metadata = []
        self.index = None

    def ingest_documents(self, data_folder=Config.DATA_FOLDER):
        for filename in os.listdir(data_folder):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(data_folder, filename)
                documents = parse_pdf_with_pypdf(pdf_path)
                for doc in documents:
                    embedding = self.model.embed_text(doc.page_content)
                    self.embeddings.append(embedding)
                    self.metadata.append(doc.page_content)
        self.build_faiss_index()

    def build_faiss_index(self):
        embeddings_array = np.array(self.embeddings).astype("float32")
        self.index = faiss.IndexFlatL2(embeddings_array.shape[1])
        self.index.add(embeddings_array)

    def retrieve_documents(self, query, top_k=5):
        query_embedding = self.model.embed_text(query)
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        return [self.metadata[idx] for idx in indices[0]]
