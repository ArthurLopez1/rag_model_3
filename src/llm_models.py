# models/llm_model.py
from langchain_ollama import ChatOllama

class LLMModel:
    def __init__(self, model_name="llama3.2:1b-instruct-fp16", temperature=0, format=None):
        self.model = ChatOllama(model=model_name, temperature=temperature, format=format)

    def generate_response(self, prompt):
        return self.model(prompt)

    def generate_json_response(self, prompt):
        json_model = ChatOllama(model=self.model.model, temperature=self.model.temperature, format="json")
        return json_model(prompt)
    

from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, pipeline, T5Model
import torch

class HFModel:
    def __init__(self, embedding_model_name="llama3.2:1b-instruct-fp16",
                 generation_model_name="facebook/blenderbot-400M-distill"):
        
        # Initialize embedding model
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, local_files_only=True)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name, local_files_only=True)

        # Initialize text generation model
        self.generation_tokenizer = AutoTokenizer.from_pretrained(generation_model_name, local_files_only=True)
        self.generation_model = AutoModelForSeq2SeqLM.from_pretrained(generation_model_name, local_files_only=True)
        self.generator = pipeline("text2text-generation", model=self.generation_model, tokenizer=self.generation_tokenizer)

    def embed_text(self, text):
        inputs = self.embedding_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = self.embedding_model(**inputs).last_hidden_state.mean(dim=1).squeeze()
        return embeddings.numpy()

    def generate_response(self, prompt):
        response = self.generator(prompt, max_length=100, num_return_sequences=1)
        return response[0]["generated_text"]

