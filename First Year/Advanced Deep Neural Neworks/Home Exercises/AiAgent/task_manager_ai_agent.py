from typing import List, Literal, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from uuid import uuid4

# ----- Define State and Task -----
class Task(TypedDict):
    id: str
    description: str
    priority: Literal["low", "medium", "high"]
    completed: bool

class GraphState(TypedDict):
    tasks: List[Task]
    messages: List[HumanMessage | AIMessage]

# ----- Add Task Tool -----
def add_task(state: GraphState) -> GraphState:
    last_input = state["messages"][-1].content
    try:
        _, description, priority = last_input.split(":", 2)
        priority = priority.strip().lower()
        if priority not in ["low", "medium", "high"]:
            raise ValueError
    except ValueError:
        return add_messages(state, AIMessage(content="Invalid priority level. Use: low, medium, or high."))

    task = {
        "id": str(uuid4())[:8],
        "description": description.strip(),
        "priority": priority,
        "completed": False
    }
    state["tasks"].append(task)
    return add_messages(state, AIMessage(content=f"Task added successfully with ID {task['id']}"))

# ----- Complete Task Tool -----
def complete_task(state: GraphState) -> GraphState:
    last_input = state["messages"][-1].content
    try:
        _, task_id = last_input.split(":", 1)
        task_id = task_id.strip()
    except ValueError:
        return add_messages(state, AIMessage(content="Invalid format. Use: complete: <task_id>"))

    for task in state["tasks"]:
        if task["id"] == task_id:
            if task["completed"]:
                return add_messages(state, AIMessage(content="Task is already marked as completed."))
            task["completed"] = True
            return add_messages(state, AIMessage(content="Task marked as completed."))

    return add_messages(state, AIMessage(content="Task with the given ID was not found."))

# ----- List Tasks Tool -----
def list_tasks(state: GraphState) -> GraphState:
    last_input = state["messages"][-1].content
    try:
        _, filter_type = last_input.split(":", 1)
        filter_type = filter_type.strip().lower()
    except ValueError:
        return add_messages(state, AIMessage(content="Invalid format. Use: list: all / completed / pending"))

    if filter_type not in ["all", "completed", "pending"]:
        return add_messages(state, AIMessage(content="Invalid filter type. Choose from: all / completed / pending"))

    if filter_type == "all":
        tasks = state["tasks"]
    elif filter_type == "completed":
        tasks = [t for t in state["tasks"] if t["completed"]]
    else:
        tasks = [t for t in state["tasks"] if not t["completed"]]

    if not tasks:
        return add_messages(state, AIMessage(content="No tasks to display under the selected filter."))

    output = "\n".join(
        f"[{t['id']}] {t['description']} (Priority: {t['priority']}, Status: {'Completed' if t['completed'] else 'Pending'})"
        for t in tasks
    )
    return add_messages(state, AIMessage(content=output))

# ----- Input Router -----
def route_user_input(state: GraphState) -> str:
    user_input = state["messages"][-1].content.lower()
    if user_input.startswith("add"):
        return "add"
    elif user_input.startswith("complete"):
        return "complete"
    elif user_input.startswith("list"):
        return "list"
    elif user_input in ["exit", "quit", "bye"]:
        return END
    else:
        state["messages"].append(AIMessage(content="Unrecognized command. Please use: add, complete, or list."))
        return "router"

# ----- Graph Definition -----
builder = StateGraph(GraphState)

builder.add_node("router", RunnableLambda(route_user_input))
builder.add_node("add", RunnableLambda(add_task))
builder.add_node("complete", RunnableLambda(complete_task))
builder.add_node("list", RunnableLambda(list_tasks))

builder.set_entry_point("router")

# Define a function to extract next step from the result
get_next = lambda x: x  # route_user_input returns a string directly

builder.add_conditional_edges(
    "router",
    get_next,
    {
        "add": "add",
        "complete": "complete",
        "list": "list",
        "router": "router",
        END: END,
    },
)

builder.add_edge("add", "router")
builder.add_edge("complete", "router")
builder.add_edge("list", "router")

app = builder.compile()

# ----- Interactive Loop -----
if __name__ == "__main__":
    state = {"tasks": [], "messages": []}
    print("Welcome to TaskMaster! Type 'add: description : priority', 'complete: <id>', 'list: all/completed/pending', or 'exit'")
    while True:
        user_input = input(">>> ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        state = app.invoke(add_messages(state, HumanMessage(content=user_input)))
        print(state["messages"][-1].content)
