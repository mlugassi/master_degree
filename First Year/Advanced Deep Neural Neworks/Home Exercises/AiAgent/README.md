# TaskMaster AI Agent

TaskMaster is an interactive AI-based task management assistant built using [LangGraph](https://www.langchain.com/langgraph). It allows users to add tasks, mark them as complete, and view task lists—all through a natural language interface.

---

## 💡 Features

- Add tasks with a name, optional description, and optional priority (low / medium / high).
- Mark tasks as completed using their task ID.
- View all tasks or filter by completed / pending.
- Friendly and forgiving command syntax.
- Natural conversational prompts from the AI assistant.

---

## 🧠 Architecture

This project uses `LangGraph` to define a stateful AI agent that moves between defined nodes:

### State Structure (`GraphState`)

```python
class GraphState(TypedDict):
    finished: bool
    next_task_id: int
    tasks: List[Task]
    messages: List[HumanMessage | AIMessage]
```

### Task Format

```python
class Task(TypedDict):
    id: str
    name: str
    description: str
    priority: Literal["low", "medium", "high"]
    completed: bool
```

### Graph Nodes

- **init** – Initializes state with defaults.
- **router** – Determines what command the user gave and routes accordingly.
- **add** – Adds a task to the task list.
- **complete** – Marks a task as completed.
- **list** – Displays tasks based on a filter.
- **wait** – Prompts the user for the next step.
- **finish** – Ends the session.
- **unrecognize** – Handles unknown commands.

---

## 🧾 Supported Commands

```bash
add_task name:"<task name>" description:"<optional description>" priority:"<optional priority>"
mark_complete <task ID>
get_tasks [all|completed|pending]
exit
```

Examples:

```bash
add_task name:"Buy groceries" desc:"Milk, eggs, bread" prior:"high"
mark_complete 2
get_tasks completed
```

---

## ⚙️ How to Run

Make sure you have `langgraph`, `langchain-core`, and required Python dependencies installed.

Then run the script:

```bash
python your_script_name.py
```

You will interact with TaskMaster through your terminal.

---

## 🎯 Notes

- The assistant adds friendly feedback messages for each interaction.
- If an invalid command or malformed syntax is detected, it suggests corrections and provides usage hints.
- The agent state is persistent during the session, keeping track of all messages and tasks.

Enjoy managing your tasks with TaskMaster! 🎉
