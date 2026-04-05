from dotenv import load_dotenv

load_dotenv()

import ollama

from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# Tracable by langsmith as a tool
@traceable(run_type="tool")
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


# Tracable by langsmith as a tool
@traceable(run_type="tool")
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


# difference 2: Without @tool decorator, we must MANUALLY define the JSON schema for eahch function
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "the name of the product",
                    }
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount to a price and return the discounted price",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "the original price",
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "the discount tier to apply, bronze, silver or gold",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

# NOTE THAT Ollama can also auto-generate these schemas if you pass the functions directly as tools
# similar to LangChain's @tool decorator, i.e., tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions form the Args section.


# ---- Helper: Traced Ollama call ------


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(
        model=MODEL,
        messages=messages,
        tools=tools_for_llm,
    )


@traceable(name="Ollama Chat with Tools")
def run_agent(question: str):
    tool_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")
    print("-" * 50)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful shopping assistant."
            "YOu have access to a product catalog tool"
            "and a discount tool. \n\n"
            "STRICT RULES - You must fllow these exactly:\n"
            "1. NEVER guess or assume any product price."
            "2. Only call apply_discount After you have recieved a price from get_product_price."
            "Pass the exact price returned by get_product_price - do NOT pass a made-up number. \n"
            "3. NEVER calculate discounts yourself using math. Always use the apply_discount tool to calculate the discounted price. \n"
            "4. If the use does not specify a discount tier, ask them which tier to use - do NOT assume one.",
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(MAX_ITERATIONS):
        print(f"===== Iteration {iteration + 1} =======:")

        response = ollama_chat_traced(messages=messages)
        ai_message = response.message
        tool_calls = ai_message.tool_calls
        # if no tool calls, then we are done and can return the final response to the user
        if not tool_calls:
            print(f" \n Final Answer: {ai_message.content}")
            return ai_message.content

        # process only the first too call here
        tool_call = tool_calls[0]
        # Different 6: Attribute access .function.name and .function.arguments instead of .get("name") and .get("args") since we are using Ollama's raw function calling here, not LangChain's tool calling
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments or {}

        print(f" [Tool Call] {tool_name} with args {tool_args} ")

        tool_to_use = tool_dict.get(tool_name)
        if not tool_to_use:
            raise ValueError(f"Tool {tool_name} not found")
        # Difference 7: Direct functional call instead of LangChain's tool.invoke() method, since we don't have LangChain tools here
        observation = tool_to_use(**tool_args)
        print(f" [Observation] {observation} \n")

        # append the tool call and observation to the messages for the next iteration
        # so that the llm can see the result of the tool call and decide what to do next
        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        )

    print("ERROR: Max iterations reached without a final answer.")
    return None


if __name__ == "__main__":
    question = "What is the price of iPhone 14 with a silver discount?"
    run_agent(question)
