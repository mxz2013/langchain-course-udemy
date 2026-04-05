from dotenv import load_dotenv

# from _typeshed import OpenBinaryMode

load_dotenv()

from langchain.chat_models import init_chat_model

from langchain.tools import tool

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from langsmith import traceable

Max_ITERATIONS = 10

MODEL = "qwen3:1.7b"


# ------ Tools LangChain tool decorator ------


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog

    Args:
        product (str): the name of the product

    Returns:
        float: price of the product
    """

    print(f"  >>> Executing get product_price(product={product}) ")

    prices = {
        "iPhone 14": 999.99,
        "MacBook Pro": 1999.99,
        "AirPods Pro": 249.99,
        "Apple Watch Series 7": 399.99,
    }

    return prices.get(product, 0.0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price and return the discounted price

    Args:
        price (float): the original price
        discount_tier (str): the discount tier to apply, bronze, silver or gold

    Returns:
        float: the discounted price
    """

    print(
        f"  >>> Executing apply_discount(price={price}, discount_tier={discount_tier}) "
    )
    discount_percentage = {
        "bronze": 5,
        "silver": 10,
        "gold": 15,
    }
    discount = discount_percentage.get(discount_tier, 0)

    discounted_price = price * (1 - discount / 100)
    return round(discounted_price, 2)


### --- Agent Loop ------


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tool_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)

    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("-" * 50)
    messages = [
        SystemMessage(
            content="You are a helpful shopping assistant."
            "YOu have access to a product catalog tool"
            "and a discount tool. \n\n"
            "STRICT RULES - You must fllow these exactly:\n"
            "1. NEVER guess or assume any product price."
            "2. Only call apply_discount After you have recieved a price from get_product_price."
            "Pass the exact price returned by get_product_price - do NOT pass a made-up number. \n"
            "3. NEVER calculate discounts yourself using math. Always use the apply_discount tool to calculate the discounted price. \n"
            "4. If the use does not specify a discount tier, ask them which tier to use - do NOT assume one."
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(Max_ITERATIONS):
        print(f"===== Iteration {iteration + 1} =======:")
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls
        # if no tool calls, then we are done and can return the final response to the user
        if not tool_calls:
            print(f" \n Final Answer: {ai_message.content}")
            return ai_message.content

        # process only the first too call here
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f" [Tool Call] {tool_name} with args {tool_args} ")

        tool_to_use = tool_dict.get(tool_name)
        if not tool_to_use:
            raise ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use.invoke(tool_args)
        print(f" [Observation] {observation} \n")

        # append the tool call and observation to the messages for the next iteration
        # so that the llm can see the result of the tool call and decide what to do next
        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached without a final answer.")
    return None


if __name__ == "__main__":
    question = "What is the price of iPhone 14 with a silver discount?"
    run_agent(question)
