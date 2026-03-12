"""
Scribe agent – converts a clinical transcript into a structured SOAP note.
"""

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig

from src.config import AGENT_MODEL, scribe_prompt_str

try:
    from langchain_groq import ChatGroq
    _llm = ChatGroq(model=AGENT_MODEL, temperature=0.1, max_tokens=2048)
except Exception:
    _llm = None  # allow import without credentials

_scribe_prompt = ChatPromptTemplate.from_messages([
    ("system", scribe_prompt_str),
    MessagesPlaceholder("messages"),
])

_chain = _scribe_prompt | _llm if _llm else None


def parse_soap(content: str) -> str | None:
    """Return the SOAP note if it contains all required sections (S, O, A, P), else None."""
    required = ("S:", "O:", "A:", "P:")
    if all(section in content for section in required):
        return content
    return None


def scribe_node(state: dict, config: RunnableConfig) -> dict:
    """LangGraph node: generate a SOAP note from the clinical transcript."""
    response = _chain.invoke({"messages": state["messages"]}, config)
    content = response.content.strip()
    return {
        "messages": [AIMessage(content=content, name="scribe")],
        "soap_note": parse_soap(content),
    }
