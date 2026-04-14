from dotenv import load_dotenv

from langgraph.graph import MessagesState

from langgraph.prebuilt import ToolNode

from react import llm, tools


load_dotenv()

SYSTEM_MESSAGE = """
You are a helpful assitant that can use tools to answer questions.
"""


def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """Run the agent reasoning node

    Args:
        state (MessagesState): _description_

    Returns:
        MessagesState: _description_
    """

    response = llm.invoke(
        [{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]]
    )

    return {"messages": [response]}


tool_node = ToolNode(tools)
