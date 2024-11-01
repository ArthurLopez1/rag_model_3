import os
from file_handler import parse_pdf_with_pypdf
from vectorstore import VectorStoreManager
from llm_models import HFModel 
from router import get_rag_prompt
from router import get_retrieval_grader_prompt
from router import get_hallucination_grader_prompt


class DataIngestor:
    def __init__(self, data_folder="data"):
        self.data_folder = data_folder
        self.vectorstore_manager = VectorStoreManager()

    def ingest_new_data(self):
        for filename in os.listdir(self.data_folder):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(self.data_folder, filename)
                documents = parse_pdf_with_pypdf(pdf_path)
                self.vectorstore_manager.add_documents(documents)
                print(f"Added {filename} to vector store.")


def train_vector_store():
    # Initialize the ingestor to load and process PDFs
    ingestor = DataIngestor()
    ingestor.ingest_new_data()

    # Use vector store manager to build embeddings
    vector_store_manager = VectorStoreManager()
    vector_store_manager.build_embeddings()
    print("Vector store training complete.")


class Evaluator:
    def __init__(self):
        self.llm = HFModel()
        self.vectorstore_manager = VectorStoreManager()

    def test_query(self, question):
        # Retrieve documents and format the context
        docs = self.vectorstore_manager.retrieve_documents(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        # Generate the response
        rag_prompt = get_rag_prompt(context, question)
        response = self.llm.generate_response(rag_prompt)
        print("Response:", response)

        # Relevance grading
        retrieval_grader_prompt = get_retrieval_grader_prompt(context, question)
        relevance_grade = self.llm.generate_json_response(retrieval_grader_prompt)
        print("Relevance Grade:", relevance_grade['binary_score'])

        # Hallucination grading
        hallucination_grader_prompt = get_hallucination_grader_prompt(response)
        hallucination_grade = self.llm.generate_json_response(hallucination_grader_prompt)
        print("Hallucination Grade:", hallucination_grade['binary_score'])
        print("Explanation:", hallucination_grade['explanation'])

if __name__ == "__main__":
    evaluator = Evaluator()
    test_question = "What are the rules for a weather situation to count for a full hour?"
    evaluator.test_query(test_question)
