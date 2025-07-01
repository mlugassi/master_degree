from langchain_core.tools import tool
import os
import re
from typing import List, Literal
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode

from uuid import uuid4
from dotenv import load_dotenv


env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print("Loaded OPENAI_API_KEY successfully!")
else:
    print("Failed to load OPENAI_API_KEY.")
    
# ----- Define State and Task -----
class Task(TypedDict):
    id: int
    name: str
    description: str
    priority: Literal["low", "medium", "high"]
    completed: bool

class TaskAgentState(TypedDict):
    debug_mode: bool  # For debugging purposes, can be set to True to see more detailed logs
    finished: bool
    next_task_id: int
    tasks: List[Task]
    messages: List[HumanMessage | AIMessage]
    command: str | None  # To store the last command issued by the user
    tool_calls: List[dict] | None  # To store tool calls from the last message


@tool
def add_task(*, name: str, description: str, priority: str, state: TaskAgentState) -> TaskAgentState:
    """
    Adds a new task to the user's task list.

    Parameters:
        name (str): The name or title of the task to be added.
        description (str): A short description or context for the task.
        priority (str): The priority level of the task. Must be one of: 'low', 'medium', or 'high'.
        state (TaskAgentState): The current state of the agent, including the task list, messages, and metadata.

    Returns:
        TaskAgentState: The updated state with the new task added to the task list and an appropriate confirmation message.
    
    Behavior:
        - Appends the new task with a unique ID to the list of tasks in the state.
        - Updates the 'next_task_id' counter.
        - Adds a confirmation message to the messages list.
    """
    if state["debug_mode"]:
        print("<< calling complete_task with name:", name, "description:", description, "priority:", priority)

    state["messages"][-1].tool_calls = None  # Clear tool calls from the last message
    state["tasks"].append({
        "id": int(state["next_task_id"]),
        "name": name,
        "description": description,
        "priority": priority,
        "completed": False
    })
    state["next_task_id"] += 1
    state["messages"].append(AIMessage(content=f"Added task '{name}' with priority {priority}"))
    return state   

@tool
def complete_task(*, task_id: int, state: TaskAgentState) -> TaskAgentState:
    """
    Marks a task as completed based on its task ID.
    """
    if state["debug_mode"]:
        print("<< calling complete_task with task_id:", task_id)

    state["messages"][-1].tool_calls = None  # Clear tool calls from the last message
    for i, task in enumerate(state["tasks"]):
        if task["id"] == task_id:
            if task["completed"]:
                system_prompt = SystemMessage(content=(
                    "The user tried to mark a task as complete, but it is already marked completed. "
                    "Kindly acknowledge that it's already done in a friendly way."
                ))
            else:
                task["completed"] = True
                system_prompt = SystemMessage(content=(
                    "The user has successfully marked a task as completed. "
                    "Generate a cheerful confirmation message to acknowledge that."
                ))
            break
    else:
        system_prompt = SystemMessage(content=(
            "The user attempted to complete a task using a task ID that doesn't exist. "
            "Generate a helpful and understanding message to inform them that the task could not be found."
        ))
    response = llm.invoke([system_prompt])
    state["messages"] = add_messages(state["messages"], AIMessage(content=response.content))
    return state

@tool
def list_tasks(*, filter_type:str, state: TaskAgentState) -> TaskAgentState:
    """
    Displays the list of tasks based on a filter: all, completed, or pending.
    """
    if state["debug_mode"]:
        print("<< calling list_tasks with filter_type:", filter_type)

    state["messages"][-1].tool_calls = None  # Clear toll calls from the last message    

    if filter_type == "all":
        tasks = state["tasks"]
    elif filter_type == "completed":
        tasks = [t for t in state["tasks"] if t["completed"]]
    else:
        tasks = [t for t in state["tasks"] if not t["completed"]]

    if not tasks:
        system_prompt = SystemMessage(content=(
            "The selected filter returned no tasks. Generate a message to inform the user that no tasks were found."
        ))
        response = llm.invoke([system_prompt])
        msg_content = response.content
    else:
        msg_content="\n".join(f"[{t['id']}] {t['name']} - {t['description']} (Priority: {t['priority']}, Status: {'Completed' if t['completed'] else 'Pending'})" for t in tasks)
    
    state["messages"] = add_messages(state["messages"], AIMessage(content=msg_content))
    return state

@tool
def finish_state(state: TaskAgentState) -> TaskAgentState:
    """
    Marks the session as finished and sends a goodbye message to the user.
    """
    if state["debug_mode"]:
        print("<< calling finish_state")
    state["messages"][-1].tool_calls = None  # Clear tool calls from the last message
    system_prompt = SystemMessage(content=(
        "The user has chosen to end the session. Generate a polite and friendly goodbye message to conclude the interaction."
    ))
    state["finished"] = True
    response = llm.invoke([system_prompt])        
    state["messages"] = add_messages(state["messages"], AIMessage(content=response.content))
    return state


def init_state(state: TaskAgentState) -> TaskAgentState:
    state["finished"] = False if "finished" not in state else state["finished"]
    state["next_task_id"] = 0 if "next_task_id" not in state else state["next_task_id"]
    state["tasks"] = [] if "tasks" not in state else state["tasks"]
    state["messages"] = [] if "messages" not in state else state["messages"]
    state["command"] = None  # Initialize command to None
    state["tool_calls"] = None  # Initialize tool calls to None
    state["debug_mode"] = False if "debug_mode" not in state else state["debug_mode"]
    return state

def init_llm() -> str:
   system_prompt = SystemMessage(content=
    "You are a smart and kindly and interactive Task Manager assistant."
    " Your job is to help the user manage their tasks efficiently and naturally."
    " The user can interact with you in free-form language, and your responsibility is to understand their intent and perform the appropriate action."
    " Here are the core actions you can perform:"
    " 1. **Add a new task**, which includes:"
    "    - Task name"
    "    - Description"
    "    - Priority - low / normal / high"
    " 2. **Mark a task as completed** — based on its name or identifier."
    " 3. **Show tasks**, with different filters:"
    "    - All tasks"
    "    - Only completed tasks"
    "    - Only incomplete (pending) tasks"
    " 4. **Disconnect** — The user can choose to end the session at any time."
    " At the beginning of the conversation, greet the user and present these options clearly. Let them know they can ask you to do any of these actions, and you will guide them through it."
    " Throughout the conversation:"
    "   - Analyze the user's input and understand which action they want to perform."
    "   - Ask follow-up questions if you need clarification."
    "   - Respond in a friendly, concise, and helpful manner."
    "   - If the user's input is completely unclear, gently let them know that their intent was not understood and remind them of the valid commands."
    "   - If the input is partially understood, explain what is missing or incorrect and suggest how to fix it so you can help them effectively."
    "   - If you remember information about the user's tasks from the session context (such as existing tasks, task names, or IDs), use it to support the user's request and do not ignore it."
    "     Based on that context, call the appropriate tool to fulfill their intent."
    "\n\nNow, greet the user and explain what you can do for them." )
   return llm.invoke([system_prompt]).content

def extract_command(state: TaskAgentState) -> TaskAgentState:
    base_messages = state["messages"]

    system_prompt = SystemMessage(content=(
        "You are a strict but intelligent command converter for a Task Manager Agent. You receive the full history of the conversation, including previous user and assistant messages, and your job is to extract the user's intent based on their latest message."

        "You must translate their request into one of the following valid command formats:\n"
        "  • add_task name:<task_name> description:<optional_description> priority:<optional_priority[high|medium|low]>\n"
        "  • mark_complete <task ID>\n"
        "  • get_tasks [all|completed|pending]\n"
        "  • exit\n\n"

        "Strict rules:\n"
        "1. Return a valid command in exactly one of the above formats — OR respond with UNRECOGNIZED_COMMAND: <brief explanation>.\n"
        "2. Your response must always be a single line — either a valid command or a one-line error in the above format. No greetings, explanations, or extra comments.\n"
        "3. Do not simulate any tool behavior. Never say a task was added or shown unless a tool was called.\n"
        "4. Validate required fields strictly — task name, numeric task ID, and valid priority (low, medium, high). Reject truly invalid values.\n"
        "5. DO NOT be overly strict about superficial issues. If the user's meaning is clear but includes common typos, inconsistent spacing, lowercase or misspelled values (like 'proir' instead of 'priority'), you must correct them silently and return a valid command.\n"
        "6. Never reject a command just because of formatting, if the user's intent is clearly recoverable. Be strict with structure — not pedantic about spelling.\n"
        "7. If a value is truly unrecognized or missing (e.g., priority is 'fastest', or description is completely missing), respond as: UNRECOGNIZED_COMMAND: <brief reason>.\n"
        "8. Do not guess missing values — only infer meaning where clearly implied (e.g., 'low proir' → 'priority: low').\n"
        "9. Vague requests like 'show urgent' or 'what’s left' must be rejected if they cannot be matched to an exact command format.\n"
        "10. Final output must always be a single line and nothing else.\n"
    ))

    messages = [system_prompt] + base_messages
    response = llm.invoke(messages)
    state["command"] = None if response.content.startswith("UNRECOGNIZED_COMMAND") else response.content.strip()
    if state["debug_mode"]:
        if state["command"] is None:
            print("<< command not found: " + response.content[len("UNRECOGNIZED_COMMAND"):].strip())
        else:
            print("<< command found: ", state["command"])
    return state

def call_llm(state: TaskAgentState) -> TaskAgentState:
    base_messages = state["messages"]

    # Combine the instruction + conversation so far (state["messages"])
    if state["command"] is None:
        tool_use_instruction = SystemMessage(content=(
            "You received a user request that is unclear, invalid, or missing information. Identify the issue — whether it's vague, incomplete, incorrect, or does not match a supported action — and respond accordingly.\n\n"

            "Your job is to interpret the user's intent and convert it into one of the following valid command formats:\n"
            "  • add_task name:<task_name> description:<optional> priority:<low|medium|high>\n"
            "  • mark_complete <task_id>\n"
            "  • get_tasks [all|completed|pending]\n"
            "  • exit | quit | bye\n\n"

            "STRICT RULES:\n"
            "• 🚫 Never say you added, completed, or showed a task unless you actually called the relevant tool.\n"
            "• 🚫 Never display task information from memory — only from tool responses.\n"
            "• 🚫 Never guess task content, structure, filters, or ID — only act on explicit user input.\n"
            "• 🚫 Do not transform vague or informal inputs like 'urgent stuff' or 'not done tasks' into tool calls unless they clearly match the expected format.\n"
            "• 🚫 Do not hallucinate valid-sounding filters or task fields — only accept exactly: name, description, priority (low/medium/high), or task_id.\n"
            "• ✅ Validate all inputs before calling a tool. If priority is invalid, or ID is non-numeric, explain and ask the user to fix it.\n"
            "• ✅ If required fields are missing, clearly explain what is missing and ask for clarification.\n"
            "• ✅ If the message is completely unrecognized or off-topic, say so and remind the user of the valid actions.\n"
            "• ✏️ If the input includes typos or vague phrasing, rephrase it clearly before proceeding.\n"
            "• ⚠️ You must always act through tools — never simulate, describe, or fabricate tool behavior.\n"
            "• 🧠 You are not allowed to use memory of past tasks, actions, or assistant messages when generating results. Only tools are permitted for accessing task state.\n"
            "• ✅ Output must be accurate, tool-driven, and grounded in validated inputs — no assumptions, no shortcuts."
        ))
        response = llm.invoke([tool_use_instruction] + base_messages)
        state["messages"] = add_messages(state["messages"], AIMessage(content=response.content))        
    else:
        response = llm.invoke([state["command"]])
        state["tool_calls"] = response.tool_calls
    return state

def final_state(state: TaskAgentState) -> TaskAgentState:
    return state  

def should_call_tool(state: TaskAgentState) -> str:
    if not state["tool_calls"]:
        return "final"
    else:
        return "tools"

tools = [finish_state, add_task, complete_task, list_tasks]
tool_node = ToolNode(tools=tools)
tool_map = {tool.name: tool for tool in tools}

def wrap_tool_node(state: TaskAgentState) -> TaskAgentState:
    for call in state["tool_calls"]:
        tool_name = call["name"]
        tool_func = tool_map.get(tool_name)
        if tool_func:
            args = call.get("args", {})
            args["state"] = state
            try:
                state = tool_func.invoke(args)
            except Exception as e:
                print(f"Error invoking tool '{tool_name}': {e}")
        else:
            print(f"Tool '{tool_name}' not found in tool map.")
    return state

llm: ChatOpenAI = ChatOpenAI(model="gpt-4-turbo", temperature=0).bind_tools(tools)

# ----- Graph Definition -----
builder = StateGraph(TaskAgentState)

builder.add_node("init", RunnableLambda(init_state))
builder.add_node("call_llm", RunnableLambda(call_llm))
builder.add_node("extract", RunnableLambda(extract_command))
builder.add_node("tools", RunnableLambda(wrap_tool_node))
builder.add_node("final", RunnableLambda(final_state))

builder.add_edge("init", "extract")
builder.add_edge("extract", "call_llm")
builder.add_conditional_edges("call_llm", should_call_tool, {
    "tools": "tools",
    "final": "final"
})
builder.add_edge("final", END)

builder.set_entry_point("init")

app = builder.compile()

# ----- Interactive Loop -----
if __name__ == "__main__":
    hello_message = init_llm()
    state: TaskAgentState = {"debug_mode": True, "finished": False, "messages": [AIMessage(content=hello_message)]}

    print(state["messages"][-1].content)
    while not state["finished"]:
        user_input = input(">>> ")
        state["messages"] = add_messages(state["messages"], HumanMessage(content=user_input))
        state = app.invoke(state)
        print(state["messages"][-1].content)
