from typing import List, Optional, TypedDict


class NodeState(TypedDict):
    id: str
    goal: str
    parent_id: Optional[str]
    children_ids: List[str]
    status: str
    result: str
    summary: str
    depth: int
    task_type: str
    vram_points: int


def reduce_node_state(current: NodeState, action: dict) -> NodeState:
    """
    Pure reducer function for state updates.

    Args:
        current: Current state
        action: Dict containing 'type' and 'payload'
    """
    new_state = current.copy()

    match action.get("type"):
        case "START":
            new_state["status"] = "IN_PROGRESS"
        case "SPLIT":
            new_state["status"] = "SPLIT"
            if "children" in action["payload"]:
                new_state["children_ids"] = action["payload"]["children"]
        case "COMPLETE":
            new_state["status"] = "COMPLETED"
            new_state["result"] = action["payload"].get("result", "")
            new_state["summary"] = action["payload"].get("summary", "")
        case "FAIL":
            new_state["status"] = "FAILED"
            new_state["result"] = action["payload"].get("error", "Unknown error")
        case "CANCEL":
            new_state["status"] = "CANCELLED"
            new_state["result"] = "Execution cancelled"
        case "WAIT_FOR_USER":
            new_state["status"] = "WAITING_FOR_USER"
            new_state["result"] = action["payload"].get("reason", "Waiting for user feedback")
        case "UPDATE_CONTEXT":
            # Context is not strictly in NodeState TypedDict above (it was separate in class),
            # but if we move it here:
            pass

    return new_state
