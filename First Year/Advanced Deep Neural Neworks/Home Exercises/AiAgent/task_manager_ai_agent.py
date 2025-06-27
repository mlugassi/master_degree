import re
from typing import List, Literal, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from uuid import uuid4

# ----- Define State and Task -----
class Task(TypedDict):
    id: str
    name: str
    description: str
    priority: Literal["low", "medium", "high"]
    completed: bool

class GraphState(TypedDict):
    finished: bool
    next_task_id: int
    tasks: List[Task]
    messages: List[HumanMessage | AIMessage]

# ----- Init State Node -----
def init_state(state: GraphState) -> GraphState:
    state["finished"] = False if "finished" not in state else state["finished"]
    state["next_task_id"] = 0 if "next_task_id" not in state else state["next_task_id"]
    state["tasks"] = [] if "tasks" not in state else state["tasks"]
    state["messages"] = [] if "messages" not in state else state["messages"]
    return state

def unrecognize_state(state: GraphState) -> GraphState:
    state["messages"].append(AIMessage(content=(
        "\n\nOopsie, looks like that didn’t go through quite right 😅\n"
        "Just a quick reminder, here’s how you can ask me things:\n"
        "  • To add a task: add_task name:\"<task name>\" description:\"<optional description>\" priority:\"<optional priority[hight|medium|low]>\"\n"
        "  • To mark a task complete: mark_complete <task ID>\n"
        "  • To view tasks: get_tasks [all|completed|pending]\n"
        "  • Type 'exit/quit/bye' anytime to quit.\n"
        "So, what would you like to do next? 😊\n"
    )))   
    return state

# ----- Init State Node -----
def finish_state(state: GraphState) -> GraphState:
    state["finished"] = True
    state["messages"] = add_messages(state["messages"], AIMessage(content="Goodbye"))
    return state  

def wait_user_state(state: GraphState) -> GraphState:
    state["messages"][-1].content += "\n\nWhat would you like to do next? add a new task (add_task), mark as completed (mark_complete), or list your tasks (get_tasks)."
    return state

# ----- Add Task Tool -----
def add_task(state: GraphState) -> GraphState:
    last_input = state["messages"][-1].content
    name_match = re.search(r'(?:name):\s*"([^"]+)"', last_input, re.IGNORECASE)
    first_valid_format_match = re.search(r'(?:name):\s*"([^"]+)"\s*(?:desc):\s*"([^"]*)"', last_input, re.IGNORECASE)
    second_valid_format_match = re.search(r'(?:name):\s*"([^"]+)"\s*(?:prior):\s*"([^"]+)"', last_input, re.IGNORECASE)
    third_valid_format_match = re.search(r'(?:name):\s*"([^"]+)"\s*(?:desc):\s*"([^"]*)"\s*(?:prior):\s*"([^"]+)"', last_input, re.IGNORECASE)

    if not last_input.startswith("add_task") or name_match is None or (first_valid_format_match is None and second_valid_format_match is None and third_valid_format_match is None):
        state["messages"] = add_messages(state["messages"], AIMessage(content="\n\nOopsie, looks like that didn’t go through quite right 😅\nJust a reminder you should use: add_task name:<task_name> description:<optional_description> priority:<optional_priority>"))
    else:
        desc_match = re.search(r'(?:desc):\s*"([^"]*)"', last_input, re.IGNORECASE)
        prio_match = re.search(r'(?:prior):\s*"([^"]+)"', last_input, re.IGNORECASE)

        name = name_match.group(1)
        description = desc_match.group(1) if desc_match else ""
        priority = prio_match.group(1).lower() if prio_match else None

        if (priority is None and "prior" in last_input) or (priority not in ["low", "medium", "high"]):
            state["messages"] = add_messages(state["messages"], AIMessage(content="\n\nOopsie, looks like that didn’t go through quite right 😅\nJust a reminder we have only priority levels of low/medium/high"))
        else:
            state["tasks"].append({
                "id": state["next_task_id"],
                "name": name,
                "description": description,
                "priority": "medium" if priority is None else priority,
                "completed": False
            })
            state["messages"] = add_messages(state["messages"], AIMessage(content=f"YAY! Task '{name}' added with ID {state['next_task_id']} (Priority: {priority})"))
            state["next_task_id"] += 1
    return state

# ----- Complete Task Tool -----
def complete_task(state: GraphState) -> GraphState:
    last_input = state["messages"][-1].content
    parts = last_input.strip().split()
    if len(parts) != 2 or parts[0] != "mark_complete":
        state["messages"] = add_messages(state["messages"], AIMessage(content="\n\nOopsie, looks like that didn’t go through quite right 😅\nJust a reminder you should use: mark_complete <task_id>"))
    else:
        task_id = int(parts[1].strip())
        for i, task in enumerate(state["tasks"]):
            if task["id"] == task_id:
                if task["completed"]:
                    state["messages"] = add_messages(state["messages"], AIMessage(content="Great :) Task is already marked as completed."))
                task["completed"] = True
                state["messages"] = add_messages(state["messages"], AIMessage(content="YAY! Task marked as completed."))
                break
        else:
            state["messages"] = add_messages(state["messages"], AIMessage(content="Hey... I did't find that task :/"))

    return state

# ----- List Tasks Tool -----
def list_tasks(state: GraphState) -> GraphState:
    last_input = state["messages"][-1].content
    parts = last_input.strip().split()
    if len(parts) != 2 or parts[0] != "get_tasks":
        state["messages"] = add_messages(state["messages"], AIMessage(content="\n\nOopsie, looks like that didn’t go through quite right 😅\nJust a reminder you should use: get_tasks [all|completed|pending]"))
    else:
        filter_type = parts[1].strip().lower()
        if filter_type not in ["all", "completed", "pending"]:
            state["messages"] = add_messages(state["messages"], AIMessage(content="\n\nOopsie, looks like that didn’t go through quite right 😅\nYou probably used an invalid filter type. Choose one from: [all|completed|pending]"))
        else:
            if filter_type == "all":
                tasks = state["tasks"]
            elif filter_type == "completed":
                tasks = [t for t in state["tasks"] if t["completed"]]
            else:
                tasks = [t for t in state["tasks"] if not t["completed"]]

            if not tasks:
                output = "Hi!\n Seems that there is no tasks to display under the selected filter.\n"
            else:
                output = "\n".join(f"[{t['id']}] {t['name']} - {t['description']} (Priority: {t['priority']}, Status: {'Completed' if t['completed'] else 'Pending'})" for t in tasks)
            state["messages"] = add_messages(state["messages"], AIMessage(content=output))
    return state

def route_user_input(state: GraphState) -> GraphState:
    last_input = state["messages"][-1].content
    if last_input.startswith("add_task"):
        return "add"
    elif last_input.startswith("mark_complete"):
        return "complete"
    elif last_input.startswith("get_tasks") or last_input.strip() == "get_tasks":
        return "list"
    elif last_input in ["exit", "quit", "bye"]:
        return "finish"
    return "unrecognize"

# ----- Graph Definition -----
builder = StateGraph(GraphState)

builder.add_node("init", RunnableLambda(init_state))
builder.add_node("router", RunnableLambda(lambda state: state))
builder.add_node("add", RunnableLambda(add_task))
builder.add_node("complete", RunnableLambda(complete_task))
builder.add_node("list", RunnableLambda(list_tasks))
builder.add_node("finish", RunnableLambda(finish_state))
builder.add_node("unrecognize", RunnableLambda(unrecognize_state))
builder.add_node("wait", RunnableLambda(wait_user_state))

builder.set_entry_point("init")

builder.add_conditional_edges(
    "router",
    route_user_input,
    {
        "wait": "wait",
        "add": "add",
        "complete": "complete",
        "list": "list",
        "unrecognize": "unrecognize",
        "finish": "finish",
    },
)

builder.add_edge("init", "router")
builder.add_edge("add", "wait")
builder.add_edge("complete", "wait")
builder.add_edge("list", "wait")
builder.add_edge("wait", END)
builder.add_edge("finish", END)

app = builder.compile()

# ----- Interactive Loop -----
if __name__ == "__main__":
    state: GraphState = {"finished": False, "messages": []}

    print(
        "Hi there! I'm TaskMaster – your personal task assistant.\n"
        "You can type commands like:\n"
        "  • add_task name:<task_name> description:<optional_description> priority:<optional_priority[high|medium|low]>\n"
        "  • mark_complete <task ID>\n"
        "  • get_tasks [all|completed|pending]\n"
        "  • Type 'exit/quit/bye' anytime to quit.\n"
        "So, what would you like to do? 😊\n"        
    )

    while not state["finished"]:
        user_input = input(">>> ")
        state["messages"] = add_messages(state["messages"], HumanMessage(content=user_input))
        state = app.invoke(state)
        print(state["messages"][-1].content)
