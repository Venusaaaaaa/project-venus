from langchain.tools import tool
from rag import get_retriever

retriever = get_retriever()

@tool
def search_sensor_info(query: str) -> str:
    """
    Retrieve relevant information about a car sensor problem from the RAG knowledge base.
    """
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([d.page_content for d in docs])

@tool
def summarize_for_user(text: str) -> str:
    """
    Simplify and re-explain retrieved information for a non-technical user.
    """
    return (
        "Here is a simplified explanation based on the retrieved information:\n\n"
        + text
    )
