from langchain_core.messages import HumanMessage
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
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


llm = ChatOpenAI(model="gpt-4.1-nano")
llm_with_tools = llm.bind_tools([multiply])


# Node

def tool_calling_llm(state: MessagesState):
    # Will return either a tool call or regular message
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)  # acts like a router
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", END)
graph = builder.compile()


if __name__ == "__main__":
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_filename = f"{script_name}.png"
    output_path = os.path.join("../images", output_filename)
    # Generate the image data
    image_data = graph.get_graph().draw_mermaid_png()

    # Save the image
    with open(output_path, "wb") as f:
        f.write(image_data)

    print(f"Graph saved to {output_path}")

    # messages = [HumanMessage(content="Hello, what is 2 multiplied by 2?")]
    # messages = graph.invoke({"messages": messages})
    # for m in messages['messages']:
    #     m.pretty_print()
