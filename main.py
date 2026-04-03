from dotenv import load_dotenv

load_dotenv()

from typing import List, Optional
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.tools import tool

from langchain_core.messages import HumanMessage

from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI

from langchain_tavily import TavilySearch

from tavily import TavilyClient

tavily = TavilyClient()


class Source(BaseModel):
    """Schema for a source used by the agent

    Args:
        BaseModel (_type_): _description_

    Returns:
        _type_: _description_
    """

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response

    Args:
        BaseModel (_type_): _description_

    Returns:
        _type_: _description_
    """

    answer: str = Field(description="The agent answer to the user query")
    sources: List[Source] = Field(
        default_factory=list,
        description="The list of the sources used to generate the answer",
    )


# implent a search tool that uses the Ollama API to search for information


# The description of the tool is very important, it should be clear and concise, and it should explain what the tool does and how to use it. The description will be used by the agent to decide when to use the tool and how to use it.
@tool
def search(query: str) -> str:
    """
    Tool that searches over internet for information.
    Args:
        query (str): The query to search for

    Returns:
        str: the search results as a string
    """
    print(f"Search for {query}")
    return tavily.search(query=query)


# define the LLM to use the Ollama API with the gpt-oss:latest model and a temperature of 0.0
llm = ChatOllama(model="gpt-oss:latest", temperature=0.0)
# llm = ChatOpenAI(model="gpt-5", temperature=0.0)
# create an agent that uses the LLM and the search tool
# tools = [TavilySearch(max_results=1, include_raw_response=False, include_anser=True)]
tools = [search]

agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")

    results = agent.invoke(
        {"messages": HumanMessage(content="What is the weather of Paris today?")}
    )
    print(results)


if __name__ == "__main__":
    main()
