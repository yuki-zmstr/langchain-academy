from langgraph.graph import START, StateGraph
from IPython.display import Image, display
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv("../.env")


def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# This will be a tool


def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b


tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-4.1-nano")

llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)


# System message
sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node


def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()

messages = [HumanMessage(
    content="Add 3 and 4. Multiply the output by 2. Divide the output by 5")]
messages = react_graph.invoke({"messages": messages})

if __name__ == "__main__":
    # script_name = os.path.splitext(os.path.basename(__file__))[0]
    # output_filename = f"{script_name}.png"
    # output_path = os.path.join("../images", output_filename)
    # # Generate the image data
    # image_data = react_graph.get_graph().draw_mermaid_png()

    # # Save the image
    # with open(output_path, "wb") as f:
    #     f.write(image_data)

    # print(f"Graph saved to {output_path}")
    for m in messages['messages']:
        m.pretty_print()
