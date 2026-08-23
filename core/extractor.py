import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI


def get_llm():
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set in environment / .env")
    return ChatMistralAI(
        model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        mistral_api_key=api_key,
        temperature=0.2,
    )


def build_chain(system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{text}")]
    )
    return prompt | get_llm() | StrOutputParser()


def _invoke(chain, transcript: str) -> str:
    transcript = transcript.strip()
    if not transcript:
        raise ValueError("Transcript is empty.")
    return chain.invoke({"text": transcript})


def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, extract all action items. "
        "For each provide task description, owner, and deadline. If owner or deadline is not mentioned, "
        "write 'Not specified'. Format as a numbered list. If none are found, say 'No action items found.' "
        "Do not invent missing details."
    )
    return _invoke(chain, transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. Extract only decisions explicitly made in the transcript. "
        "Format as a numbered list. If none are found, say 'No key decisions found.' Do not invent details."
    )
    return _invoke(chain, transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "Extract unresolved questions or topics explicitly needing follow-up from the transcript. "
        "Format as a numbered list. If none are found, say 'No open questions found.' Do not invent details."
    )
    return _invoke(chain, transcript)
