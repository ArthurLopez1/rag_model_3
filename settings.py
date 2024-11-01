# config/settings.py

import os
import getpass

class Config:

    DATA_FOLDER = "data"
    EMBEDDING_MODEL_PATH = "models/sentence_transformer"
    GENERATION_MODEL_PATH = "models/generation_model"
    FAISS_INDEX_PATH = "vectorstore/faiss_index.bin"
    CROSS_VALIDATION_SPLITS = 5
    @staticmethod
    def set_env(var: str):
        """Prompts for environment variables if they are not already set."""
        if not os.environ.get(var):
            os.environ[var] = getpass.getpass(f"{var}: ")


    @staticmethod
    def initialize():
        """Sets all required environment variables."""
        Config.set_env("TAVILY_API_KEY")
        Config.set_env("LANGCHAIN_API_KEY")
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "local-llama32-rag"

