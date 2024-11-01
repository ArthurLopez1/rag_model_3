# prompts/router.py
from langchain_core.messages import SystemMessage, HumanMessage

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

router_instructions = """You are an expert on Swedish winter road maintenance and meteorology. 
Use 'vectorstore' for questions related to this, and 'websearch' for others."""

def get_router_prompt(question):
    return [
        SystemMessage(content=router_instructions),
        HumanMessage(content=question)
    ]

doc_grader_instructions = """You are grading the relevance of a retrieved document. Use 'yes' or 'no'."""

def get_retrieval_grader_prompt(document, question):
    doc_grader_prompt = f"Document: {document} \nQuestion: {question}"
    return [
        SystemMessage(content=doc_grader_instructions),
        HumanMessage(content=doc_grader_prompt)
    ]


rag_prompt_template = """Here is the context: {context}\n\nQuestion: {question}\nAnswer in concise Swedish."""

def get_rag_prompt(context, question):
    return rag_prompt_template.format(context=context, question=question)
