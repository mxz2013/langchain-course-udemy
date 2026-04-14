from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


load_dotenv()


# declare the function as a tool so that the LLM can make tool call later
@tool
def triple(num: float) -> float:
    """triple the num and return it

    Args:
        num (float): input number

    Returns:
        float: result of tripling the input number
    """
    return float(num) * 3


tools = [TavilySearch(max_results=1), triple]

# llm = ChatOllama(model="gpt-oss:latest", temperature=0).bind_tools(tools)
llm = ChatOpenAI(model="gpt-5", temperature=0).bind_tools(tools)
