import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_mistralai import ChatMistralAI

from core.vector_store import build_vector_store, get_retriever, load_vector_store


def get_llm():
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set in environment / .env")
    return ChatMistralAI(
        model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        mistral_api_key=api_key,
        temperature=0.3,
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def _make_rag_chain(vector_store):
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert meeting assistant. Answer the user's question
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say:
\"I could not find this information in the meeting transcript.\"

Always be concise and precise. Do not invent names, facts, quotes, or decisions.

Context from meeting transcript:
{context}""",
            ),
            ("human", "{question}"),
        ]
    )

    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def build_rag_chain(transcript: str):
    return _make_rag_chain(build_vector_store(transcript))


def load_rag_chain():
    """Reload the most recently built persisted RAG collection."""
    return _make_rag_chain(load_vector_store())


def ask_question(rag_chain, question: str) -> str:
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    print(f"Question: {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")
    return answer
