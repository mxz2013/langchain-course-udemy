import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.agents import create_agent

from langchain.chat_models import init_chat_model

from langchain.messages import ToolMessage
from langchain.tools import tool

from langchain_pinecone import PineconeVectorStore

from langchain_openai import OpenAIEmbeddings

# load environment variable from .env
load_dotenv()

# Initialize embeddings, note that we have to use the same imbeddings as ingestions.py in order to make the search meaningful.
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)  # do we also provide the embedding dimension to be the same as the ingestion?

# Initialize vector store
vector_store = PineconeVectorStore(
    index_name="langchain-documentation-2026-april", embedding=embeddings
)

# Initialize the chat model
model = init_chat_model("gpt-oss:latest", model_provider="ollama")


# we want both content and artifact
@tool(response_format="content_and_artifact")
def retrieve_context(query: str) -> (str, List[str]):
    """Retrieve relevant documentation to help the LLM answer user queries about LangChain.

    Args:
        query (str): user query
        List (_type_): serialized content and raw documents
    """

    # retrieve top 4 most similar documents
    # as_retriever() is better than similarity_search funciton, because it works better with langsmigh tracing.
    retrieved_docs = vector_store.as_retriever().invoke(query, k=4)

    # serialize documents for the model

    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'Unknow')} \n\n Content: {doc.page_content}"
            for doc in retrieved_docs
        )
    )

    # Return both serialized content and raw documents

    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """Run the RAG pipeline to answer a query using retrieved documentation.

    Args:
        query (str): user input query

    Returns:
        Dict[str, Any]: Dictonary containing
        - answer: The generated answer
        - context: List of retrieved documents
    """

    # create a system prompt for the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChian documentation."
        "You have access to a tool that retrieves relevant documentation."
        "Use the tool to find relevant information before answering questions."
        "Always cite the sources you have used in your answer."
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    # Build messages list
    messages = [{"role": "user", "content": query}]

    # Invoke the agent
    response = agent.invoke({"messages": messages})

    # Extract the answer from the last AI message
    answer = response["messages"][-1].content

    # Extract the context documents form ToolMessage Artifacts

    context_docs = []

    for message in response["messages"]:
        # check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # the artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {"answer": answer, "context": context_docs}


if __name__ == "__main__":
    result = run_llm(query="what are deep agents?")
    print(result)
