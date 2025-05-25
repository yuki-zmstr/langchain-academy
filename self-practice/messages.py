import os
from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from typing import Annotated
from langgraph.graph.message import add_messages
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv("../.env")


# class MessagesState(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]


class MessagesState(MessagesState):
    # Add any keys needed beyond messages, which is pre-built
    pass


# messages = [AIMessage(
#     content=f"So you said you like to code?", name="Model")]
# messages.append(HumanMessage(content=f"Yes, that's right.", name="Yuki"))
# messages.append(
#     AIMessage(content=f"Great, what would you like to learn about.", name="Model"))
# messages.append(HumanMessage(
#     content=f"I want to learn about how LangGraph implements memory", name="Yuki"))

llm = ChatOpenAI(model="gpt-4.1-nano")
# result = llm.invoke(messages)


def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


llm_with_tools = llm.bind_tools([multiply])
tool_call = llm_with_tools.invoke(
    [HumanMessage(content=f"What is 2 multiplied by 3", name="Yuki")])

# Node


def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)
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

    # The LLM chooses to use a tool when it determines that the input or task requires the functionality provided by that tool.
    # messages = graph.invoke({"messages": HumanMessage(content="Hello!")})
    # for m in messages['messages']:
    #     m.pretty_print()

    # messages = graph.invoke(
    #     {"messages": HumanMessage(content="Multiply 2 and 3")})
    # for m in messages['messages']:
    #     m.pretty_print()
