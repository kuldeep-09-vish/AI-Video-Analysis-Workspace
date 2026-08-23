import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_llm():
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set in environment / .env")
    return ChatMistralAI(
        model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        mistral_api_key=api_key,
        temperature=0.3,
    )


def split_transcript(transcript: str) -> list[str]:
    transcript = transcript.strip()
    if not transcript:
        raise ValueError("Transcript is empty.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_llm()
    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize this portion of a meeting transcript concisely."),
            ("human", "{text}"),
        ]
    )
    map_chain = map_prompt | llm | StrOutputParser()
    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in split_transcript(transcript)]
    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Combine the partial summaries into one professional meeting summary in concise bullet points. "
                "Do not add facts that are not present in the source summaries.",
            ),
            ("human", "{text}"),
        ]
    )
    return (combined_prompt | llm | StrOutputParser()).invoke({"text": combined})


def generate_title(transcript: str) -> str:
    transcript = transcript.strip()
    if not transcript:
        raise ValueError("Transcript is empty.")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(maximum 8 words). Return only the title.",
            ),
            ("human", "{text}"),
        ]
    )
    return (prompt | get_llm() | StrOutputParser()).invoke({"text": transcript[:2000]}).strip()
