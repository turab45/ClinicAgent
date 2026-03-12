from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# ────────────────────────────────────────────────
#  CONFIG
# ────────────────────────────────────────────────
load_dotenv()

SCRIBE_MODEL = "llama-3.3-70b-versatile"


def get_scribe_model() -> ChatGroq:
    return ChatGroq(
        model=SCRIBE_MODEL,
        temperature=0.1,
        max_tokens=2048,
    )


try:
    llm = get_scribe_model()
except Exception:
    llm = None  # Allow import without credentials; will fail at runtime if not set

# ────────────────────────────────────────────────
#  STATE
# ────────────────────────────────────────────────
class ScribeState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "add_messages"]
    transcript: str
    soap_note: str | None


# ────────────────────────────────────────────────
#  PROMPT
# ────────────────────────────────────────────────
SCRIBE_SYSTEM_PROMPT = """You are an expert medical scribe. Convert the raw conversation transcript into a clean, structured SOAP note.
Use standard format:
S: Subjective – patient's chief complaint, history, and symptoms in their own words
O: Objective – measurable findings: vitals, physical exam, lab results
A: Assessment – differential or working diagnosis
P: Plan – ordered tests, medications, referrals, and follow-up

Be concise, professional, and accurate. Do NOT diagnose or invent information not present in the transcript.
Output ONLY the SOAP note."""

scribe_prompt = ChatPromptTemplate.from_messages([
    ("system", SCRIBE_SYSTEM_PROMPT),
    MessagesPlaceholder("messages"),
])

scribe_chain = scribe_prompt | llm if llm else None


# ────────────────────────────────────────────────
#  PARSER
# ────────────────────────────────────────────────
def parse_soap(content: str) -> str | None:
    """Return the SOAP note if it contains all required sections (S, O, A, P), else None."""
    required = ("S:", "O:", "A:", "P:")
    if all(section in content for section in required):
        return content
    return None


# ────────────────────────────────────────────────
#  AGENT NODE
# ────────────────────────────────────────────────
def scribe_node(state: ScribeState, config: RunnableConfig) -> dict:
    """Convert the clinical transcript into a structured SOAP note."""
    response = scribe_chain.invoke(
        {"messages": state["messages"]},
        config,
    )
    content = response.content.strip()
    soap = parse_soap(content)

    return {
        "messages": [AIMessage(content=content, name="scribe")],
        "soap_note": soap,
    }


# ────────────────────────────────────────────────
#  GRAPH
# ────────────────────────────────────────────────
workflow = StateGraph(state_schema=ScribeState)
workflow.add_node("scribe", scribe_node)
workflow.add_edge(START, "scribe")
workflow.add_edge("scribe", END)

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)


# ────────────────────────────────────────────────
#  ENTRY POINT
# ────────────────────────────────────────────────
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "scribe_session_1"}}

    sample_transcript = """
    Patient: I've been feeling tired, thirsty a lot, and peeing more frequently. I've also lost about 8 pounds in the last 3 months.
    Doctor: Do you have any family history of diabetes? Any blurry vision or numbness in your feet?
    Patient: My mom had type 2 diabetes. No numbness yet, but my vision has been slightly blurry lately.
    Doctor: Blood pressure is 138/84, weight 210 lb, height 5'10". I'll order an A1c, lipid panel, and urine microalbumin today.
    """

    initial_state: ScribeState = {
        "messages": [HumanMessage(content=f"Please scribe the following clinical transcript:\n{sample_transcript}")],
        "transcript": sample_transcript,
        "soap_note": None,
    }

    print("Running scribe agent...\n")

    final_state = graph.invoke(initial_state, config)

    print("=" * 70)
    print("SOAP NOTE")
    print("=" * 70)
    print(final_state.get("soap_note") or final_state["messages"][-1].content)
