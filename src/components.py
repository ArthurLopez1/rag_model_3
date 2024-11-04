# prompts/router.py
from langchain_core.messages import SystemMessage, HumanMessage
import json

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Prompt
router_instructions = """You are an expert at Swedish and winter road maintenance employed by Klimator.

The vectorstore contains documents related to meteorological facts, winter road maintance laws, Road Weather Forecast and related fields of study.

Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.

Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""

def get_router_prompt(question):
    return [
        SystemMessage(content=router_instructions),
        HumanMessage(content=question)
    ]

# Doc grader instructions
doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

def get_retrieval_grader_prompt(document, question):
    doc_grader_prompt = f"Document: {document} \nQuestion: {question}"
    return [
        SystemMessage(content=doc_grader_instructions),
        HumanMessage(content=doc_grader_prompt)
    ]


# Grader prompt
doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

This carefully and objectively assess whether the document contains at least some information that is relevant to the question.

Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""


# Prompt
rag_prompt = """You are an assistant for question-answering tasks in swedish. 

Here is the context to use to answer the question:

{context}

Think carefully about the above context. 

Now, review the user question:

{question}

Provide an answer to this questions using only the above context. 

Use three sentences maximum and keep the answer concise.

Answer:"""

def get_rag_prompt(context, question):
    return rag_prompt.format(context=context, question=question)

# Hallucination grader instructions
hallucination_grader_instructions = """

You are a teacher grading a quiz in Swedish. 

You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader prompt
hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""


# Answer grader instructions
answer_grader_instructions = """You are a teacher grading a quiz in Swedish. 

You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) The STUDENT ANSWER helps to answer the QUESTION

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader prompt
answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""


