from langchain_core.tools import Tool
from rag import get_retriever


def _search_sensor_info(query: str) -> str:
    retriever = get_retriever()
    docs = retriever.invoke(query)   # <-- FIXED
    return "\n\n".join([d.page_content for d in docs])


def _summarize_for_user(text: str) -> str:
    return (
        "Simplified explanation:\n\n" +
        text
    )


search_sensor_info = Tool(
    name="search_sensor_info",
    description="Retrieve relevant sensor troubleshooting information.",
    func=_search_sensor_info
)

summarize_for_user = Tool(
    name="summarize_for_user",
    description="Simplify retrieved technical information.",
    func=_summarize_for_user
)
