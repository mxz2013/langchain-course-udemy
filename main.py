from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.graph import END, StateGraph

from langgraph.graph import add_messages
from chains import generate_chain, reflect_chain

load_dotenv()
REFLECT = "reflect"
GENERATE = "generat"


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def generate_node(state: MessageGraph) -> dict:
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflect_node(state: MessageGraph) -> dict:
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {
        "messages": [HumanMessage(content=res.content)]
    }  # here it supposed to be a AI message
    # but we want to make the LLM to take it as a uers message while thinking


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generate_node)
builder.add_node(REFLECT, reflect_node)
builder.set_entry_point(GENERATE)


def should_continue(state: MessageGraph) -> str:
    """the condition to stop the thinking loop

    Args:
        state (MessageGraph): state of the thinking loop

    Returns:
        str: either END or keep thinking
    """

    if len(state["messages"]) > 6:
        return END
    else:
        return REFLECT


# we need to use path_map to tell the system how to continue the thinking loop
builder.add_conditional_edges(
    GENERATE, should_continue, path_map={END: END, REFLECT: REFLECT}
)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

print(
    graph.get_graph().draw_mermaid()
)  # can be used to visualize the thinking loop by excalidraw: Mermaid to Excalidraw

graph.get_graph().print_ascii()


if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """
            )
        ]
    }
    response = graph.invoke(inputs)
    print(response)
