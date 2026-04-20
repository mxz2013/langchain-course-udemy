from typing import Literal
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, MessagesState

from chains import revisor, first_responder

from tool_executor import execute_tools

MAX_ITERATIONS = 2  # the maximum iterations the reviser will do to stop the loop. In later sessions (agentic RAG)
# the ending condition can be made by the LLM.


def draft_node(state: MessagesState) -> dict:
    """Draft the initial response.

    Args:
        state (MessagesState): _description_

    Returns:
        dict: _description_
    """
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def revise_node(state: MessagesState) -> dict:
    """Revise the answer based on tool results.

    Args:
        state (MessagesState): _description_

    Returns:
        dict: _description_
    """
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def event_loop(state: MessagesState) -> Literal["execute_tools", END]:
    """Determine whether to continue or end based on iteration count."""
    count_tool_visites = sum(
        isinstance(item, ToolMessage) for item in state["messages"]
    )  # the number of tool call

    num_iterations = count_tool_visites
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_node)
builder.add_edge(START, "draft")  # from start to draft
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])

graph = builder.compile()


if __name__ == "__main__":
    print("Hello reflexion agent")
    print(graph.get_graph().draw_mermaid())
    res = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital.",
                }
            ]
        }
    )
    # Extract the final answer from the last message with tool calls
    last_message = res["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print(last_message.tool_calls[0]["args"]["answer"])
    print(res)
