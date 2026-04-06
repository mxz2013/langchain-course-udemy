import re
import inspect
from dotenv import load_dotenv

load_dotenv()
import re

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

    price = float(
        price
    )  # ensure price is a float in case it's passed as a string from the LLM

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

tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

# Here we delete the JSON schemas. Tools now live inside the prompt as plain text.
# We derive descriptions from the functions themselves using inspect


def get_tool_descriptions(tools_dict: dict) -> str:
    descriptions = []
    for name, func in tools_dict.items():
        # functions are wrapped by langsmith's @traceable decorator, so we need to access the original function using .__wrapped__ to get the correct docstring and type hints
        original_func = getattr(func, "__wrapped__", func)
        signature = inspect.signature(original_func)
        docstring = inspect.getdoc(original_func) or ""
        descriptions.append(f"{name}{signature}: {docstring}")

    return "\n".join(descriptions)


tool_descriptions = get_tool_descriptions(tools)

tool_names = ", ".join(tools.keys())

react_prompt = f"""
"STRICT RULES - You must fllow these exactly:\n"
            "1. NEVER guess or assume any product price."
            "2. Only call apply_discount After you have recieved a price from get_product_price."
            "Pass the exact price returned by get_product_price - do NOT pass a made-up number. \n"
            "3. NEVER calculate discounts yourself using math. Always use the apply_discount tool to calculate the discounted price. \n"
            "4. If the use does not specify a discount tier, ask them which tier to use - do NOT assume one.",

Answer the following questions as best you can. You have access to the following tools:
{tool_descriptions}
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:
"""

# Here we drop tools = from ollama.chat(). THe LLM has no idea it's an agent
# all agency comes form the prompt above and our regex parsing below


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(
        model=model,
        messages=messages,
        options=options,  # we still need to pass the tools to ollama so it can execute them when we call them in the prompt, but the LLM has no idea they are tools, they are just black box functions it can call using the format we defined in the prompt
    )


@traceable(name="Ollama Chat raw ReAct Prompt")
def run_agent(question: str):

    print(f"Question: {question}")
    print("-" * 50)
    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for iteration in range(MAX_ITERATIONS):
        print(f"===== Iteration {iteration + 1} =======:")
        full_prompt = prompt + scratchpad
        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={
                "stop": ["\nObservation"],
                "temperature": 0,
            },  # we have to stop at Observation so that we can parse the tool result before giving the next prompt to the LLM. Otherwise, the LLM will keep going and generate the next Thought and Action without waiting for our tool result, which will break the loop since we need the tool result to decide what to do next. By stopping at Observation, we can ensure that we get the tool result in the response and can parse it before continuing with the next iteration.
        )
        output = response.message.content

        print(f"LLM Output:\n{output}\n")
        # in output we need to find "Final Answer: " and extract the answer that comes after it. If we find it, we are done and can return the final answer to the user. If we don't find it, then we need to continue with the next iteration of the loop and call the tool specified in the output.
        print("  [Parsing] Looking for Final Answer in the output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)

        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print("  [Parsed] Final Answer found! Ending loop.")
            print("\n " + "=" * 50 + "\n")
            print(f"Final Answer: {final_answer}")
            return final_answer

        # Action, and Action Input
        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(
                "  [Parsed] ERROR No Action or Action Input found. Ending loop to avoid infinite loop."
            )
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(
            f"  [Tool Selected] Tool to call: {tool_name} with input: {tool_input_raw} "
        )

        # to get rid of tool_input_raw = 'product = "iPhone 14"' and turn it into {"product": "iPhone 14"}, we can use eval() since we trust the LLM not to do anything malicious here. In production, you would want a more robust and secure way to parse the tool input.

        # Split comma-separated args; strip key= prefix if LLM outputs key=value format
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"  [Tool Executing] {tool_name}({args})...")
        if (
            tool_name not in tools
        ):  # when LLM hallucinates a tool name that is not in our tools dict, we should handle it gracefully instead of crashing. We can return an error message as the observation and let the LLM decide what to do next, maybe it will try a different tool or ask the user for clarification.
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))

        print(f"  [Tool Result] {observation}")

        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    print("ERROR: Max iterations reached without a final answer.")
    return None


if __name__ == "__main__":
    question = "What is the price of iPhone 14 with a silver discount?"
    run_agent(question)
