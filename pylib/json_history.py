import json
import os
from pathlib import Path

HISTORY_FILENAME = "history.json"

def load_history(folder_path):
    """Loads history.json from the given folder. Returns empty dict if not found."""
    path = Path(folder_path) / HISTORY_FILENAME
    if not path.exists():
        return {}
    
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"⚠️ Warning: Corrupt history.json in {folder_path}. Returning empty.")
        return {}

def save_history(folder_path, data):
    """Saves data to history.json in the given folder."""
    path = Path(folder_path) / HISTORY_FILENAME
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        # print(f"✅ History saved to {path}")
    except Exception as e:
        print(f"❌ Error saving history to {path}: {e}")

def init_new_history(new_folder, parent_folder, parent_history=None):
    """
    Initializes history for a new experiment.
    Inherits session_id and summary (as context) from parent.
    """
    if parent_history is None and parent_folder:
        parent_history = load_history(parent_folder)
    
    parent_history = parent_history or {}
    
    new_data = {
        "gemini_session_id": parent_history.get("gemini_session_id"), 
        "claude_session_id": parent_history.get("claude_session_id"),
        
        "hypothesis": parent_history.get("hypothesis", parent_history.get("goal", "Initialize strategy.")),
        "parent_exp": str(parent_folder.name) if parent_folder else "example_workspace",
        "exp_design": parent_history.get("exp_design", ""),
        "hint": parent_history.get("hint", ""),
        
        "if_improved": False,
        "metrics": {},
        "result_analysis": parent_history.get("result_analysis", parent_history.get("summary", ""))
    }
    
    save_history(new_folder, new_data)
    return new_data
