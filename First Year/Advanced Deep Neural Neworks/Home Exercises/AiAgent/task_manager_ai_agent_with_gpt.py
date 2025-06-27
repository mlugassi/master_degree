import re
from typing import List, Literal, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolExecutor
from langgraph.graph import MessageGraph

# ----- Setup LLM -----
llm = ChatOpenAI(model="gpt-4", temperature=0)

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
    messages: List[HumanMessage | AIMessage | ToolMessage | SystemMessage]

# ----- Tools Definitions -----
@tool
def add_task_tool(name: str, description: str = "", priority: str = "medium") -> str:
    if priority not in ["low", "medium", "high"]:
        return "Invalid priority. Please use 'low', 'medium', or 'high'."
    task = {
        "id": str(uuid4()),
        "name": name,
        "description": description,
        "priority": priority,
        "completed": False
    }
    TASK_LIST.append(task)
    return f"✅ Task added: {name} (Priority: {priority})"

@tool
def mark_task_tool(task_id: str) -> str:
    for task in TASK_LIST:
        if task["id"] == task_id:
            if task["completed"]:
                return "Task is already completed."
            task["completed"] = True
            return f"✅ Task {task_id} marked as completed."
    return "❌ Task not found."

@tool
def list_tasks_tool(filter_type: str = "all") -> str:
    if filter_type not in ["all", "completed", "pending"]:
        return "Invalid filter. Use 'all', 'completed', or 'pending'."
    
    if filter_type == "all":
        tasks = TASK_LIST
    elif filter_type == "completed":
        tasks = [t for t in TASK_LIST if t["completed"]]
    else:
        tasks = [t for t in TASK_LIST if not t["completed"]]

    if not tasks:
        return "No tasks found for the selected filter."

    return "\n".join([
        f"[{t['id']}] {t['name']} - {t['description']} (Priority: {t['priority']}, Status: {'Completed' if t['completed'] else 'Pending'})"
        for t in tasks
    ])

# ----- Task List -----
TASK_LIST: List[Task] = []

# ----- Tool Executor -----
tool_executor = ToolExecutor(tools=[add_task_tool, mark_task_tool, list_tasks_tool])

# ----- Message Graph with GPT and Tools -----
message_graph = MessageGraph.from_llm(llm, tool_executor=tool_executor)

# ----- Init State Node -----
def init_state(state: GraphState) -> GraphState:
    state["finished"] = False
    state["next_task_id"] = 0
    state["tasks"] = []
    state["messages"] = []
    return state

# ----- GPT Reasoning Node -----
def gpt_decide(state: GraphState) -> GraphState:
    state["messages"] = message_graph.invoke(state["messages"])
    return state

# ----- Finish Node -----
def finish_state(state: GraphState) -> GraphState:
    state["finished"] = True
    state["messages"] = add_messages(state["messages"], AIMessage(content="Goodbye! 👋"))
    return state

# ----- Route Node -----
def route_user_input(state: GraphState) -> str:
    if state["finished"]:
        return "finish"
    return "gpt_decide"

# ----- Graph Definition -----
builder = StateGraph(GraphState)

builder.add_node("init", RunnableLambda(init_state))
builder.add_node("gpt_decide", RunnableLambda(gpt_decide))
builder.add_node("finish", RunnableLambda(finish_state))

builder.set_entry_point("init")

builder.add_conditional_edges(
    "gpt_decide",
    route_user_input,
    {"gpt_decide": "gpt_decide", "finish": "finish"},
)

builder.add_edge("init", "gpt_decide")

app = builder.compile()

# ----- Interactive Loop -----
if __name__ == "__main__":
    state: GraphState = {"finished": False, "messages": []}

    print(
        "Hi there! I'm TaskMaster – your personal task assistant powered by GPT-4.\n"
        "You can say things like:\n"
        "  • add_task name:\"Buy milk\" description:\"from the store\" priority:\"high\"\n"
        "  • mark_complete <task_id>\n"
        "  • get_tasks all\n"
        "  • Or just talk to me naturally!\n"
        "Type 'exit' or 'quit' anytime to finish.\n"
    )

    while not state["finished"]:
        user_input = input(">>> ")
        if user_input.strip().lower() in ["exit", "quit", "bye"]:
            state["finished"] = True
        else:
            state["messages"] = add_messages(state["messages"], HumanMessage(content=user_input))
            state = app.invoke(state)
            print(state["messages"][-1].content)
