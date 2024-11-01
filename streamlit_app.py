# streamlit_app.py
import streamlit as st
from settings import Config
from llm_models import HFModel   
from vectorstore import VectorStoreManager
from router import get_router_prompt, get_retrieval_grader_prompt, get_rag_prompt, get_hallucination_grader

# Initialize environment and settings
Config.initialize()

# Load the LLM model
llm = HFModel()

# Vector store setup
vector_manager = VectorStoreManager("backend/data/ersattningsmodell_vaders_2019.pdf")

# Streamlit UI
st.title("Local RAG Chatbot")
st.write("Ask questions related to Swedish winter road maintenance and meteorology.")

# User input
question = st.text_input("Enter your question:")

if st.button("Submit"):
    if question:
        # Routing decision
        router_prompt = get_router_prompt(question)
        route_decision = llm.generate_json_response(router_prompt)
        
        if route_decision['datasource'] == 'vectorstore':
            # Retrieve relevant documents
            docs = vector_manager.load_documents("backend/data/ersattningsmodell_vaders_2019.pdf")
            context = format_docs(docs)

            # Generate RAG response
            rag_prompt = get_rag_prompt(context, question)
            response = llm.generate_response(rag_prompt)
            st.subheader("Response:")
            st.write(response)

            # Relevance grading
            retrieval_grader_prompt = get_retrieval_grader_prompt(context, question)
            relevance_grade = llm.generate_json_response(retrieval_grader_prompt)
            st.subheader("Relevance Grade:")
            st.write(f"Relevant: {relevance_grade['binary_score']}")

            # Hallucination grading
            hallucination_grader_prompt = get_hallucination_grader_prompt(response)
            hallucination_grade = llm.generate_json_response(hallucination_grader_prompt)
            st.subheader("Hallucination Grade:")
            st.write(f"Grounded: {hallucination_grade['binary_score']}")
            st.write(f"Explanation: {hallucination_grade['explanation']}")
        
        else:
            st.write("For this question, a web search might be more appropriate.")
    else:
        st.write("Please enter a question.")
