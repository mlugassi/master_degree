from typing import TypedDict
from langgraph.graph import StateGraph


# Define our state structure
class AgentState(TypedDict):
    numbers: list
    idx: int
    op: str
    res: str


def init_node(state: AgentState) -> AgentState:
    print("init_node")
    state['idx'] = 2
    if state['op'] == "+":
        state['res'] = state['numbers'][0] + state['numbers'][1]
    else:
        state['res'] = state['numbers'][0] - state['numbers'][1]
    return state

def op_add_node(state: AgentState) -> AgentState:
    print("op_add_node")
    state["res"] += state['numbers'][state['idx']]
    state['idx'] += 1
    return state

def op_sub_node(state: AgentState) -> AgentState:
    print("op_sub_node")
    state["res"] -= state['numbers'][state['idx']]
    state['idx'] += 1
    return state

def res_node(state: AgentState) -> AgentState:
    print("res_node")
    return state["res"]

def decide_next_node(state: AgentState) -> AgentState:
    if state['idx'] >= len(state['numbers']):
        print("decide_next_node - END")
        return "END"
    
    if state['res'] > 5:
        print("decide_next_node - +")
        return "+"
    else:
        print("decide_next_node - -")
        return "-"

# Connect nodes in sequence
graph = StateGraph(AgentState)
graph.add_node("init_node", init_node)
graph.add_node("op_add_node", op_add_node)
graph.add_node("op_sub_node", op_sub_node)
graph.add_node("res_node", res_node)
graph.add_node("router", lambda state: state)

graph.set_entry_point("init_node")
graph.add_edge("init_node", "router")
graph.add_conditional_edges(
    'router',
    decide_next_node,
    {
        "+": "op_add_node",
        "-": "op_sub_node",
    }
)
graph.set_finish_point("res_node")

# Compile and run
app = graph.compile()
result = app.invoke({"numbers": [1,5,3], "op": "+"})
print(result['res']) # "Hey Bob, how is your day?"