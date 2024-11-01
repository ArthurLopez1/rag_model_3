# main.py
from backend.config.settings import Config
from backend.models import LLMModel
from backend.vectorstore.vectorstore_manager import VectorStoreManager
from backend.edges import get_router_prompt, get_rag_prompt
from backend.edges import get_retrieval_grader_prompt

def main():
    Config.initialize()

    llm = LLMModel()
    vector_manager = VectorStoreManager("backend/data/ersattningsmodell_vaders_2019.pdf")

    # Example Question
    question = "What are the rules for a weather situation to count for a full hour?"
    
    # Router Node
    router_prompt = get_router_prompt(question)
    route_decision = llm.generate_json_response(router_prompt)
    print("Routing to:", route_decision)

    # Retrieval Node (VectorStore interaction example)
    docs = vector_manager.load_documents("backend/data/ersattningsmodell_vaders_2019.pdf")
    context = format_docs(docs)

    # RAG Node
    rag_prompt = get_rag_prompt(context, question)
    answer = llm.generate_response(rag_prompt)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
