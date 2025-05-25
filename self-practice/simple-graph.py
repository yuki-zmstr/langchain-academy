import random
import os
from typing_extensions import TypedDict
from typing import Literal

from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

# State of the graph


class State(TypedDict):
    graph_state: str


# Nodes, implemented as functions
def node_1(state):
    print("---Node 1---")
    return {"graph_state": state['graph_state'] + " I am"}


def node_2(state):
    print("---Node 2---")
    return {"graph_state": state['graph_state'] + " happy!"}


def node_3(state):
    print("---Node 3---")
    return {"graph_state": state['graph_state'] + " sad!"}

# A conditional edge, implemented as a function


def decide_mood(state) -> Literal["node_2", "node_3"]:

    # Often, we will use state to decide on the next node to visit
    user_input = state['graph_state']

    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:

        # 50% of the time, we return Node 2
        return "node_2"

    # 50% of the time, we return Node 3
    return "node_3"


# We start to construct our graph
builder = StateGraph(State)

# Nodes
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Edges
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Compile
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
