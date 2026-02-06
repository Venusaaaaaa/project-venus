import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from tools import search_sensor_info, summarize_for_user

load_dotenv()

# ----- STATE -----
class AgentState(dict):
    input: str
    retrieved: str
    simplified: str
    final_answer: str


# ----- NODES -----

def node_retrieve(state: AgentState):
    """Always retrieve info from FAISS first."""
    result = search_sensor_info.run(state["input"])
    return {"retrieved": result}


def node_simplify(state: AgentState):
    """Simplify the retrieved information."""
    text = state.get("retrieved", "")
    simplified = summarize_for_user.run(text)
    return {"simplified": simplified}


def node_finish(state: AgentState):
    """Return the simplified explanation as final answer."""
    answer = state.get("simplified") or state.get("retrieved")
    return {"final_answer": answer}


# ----- GRAPH -----

def create_agent():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", node_retrieve)
    graph.add_node("simplify", node_simplify)
    graph.add_node("finish", node_finish)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "simplify")
    graph.add_edge("simplify", "finish")
    graph.add_edge("finish", END)

    return graph.compile()
