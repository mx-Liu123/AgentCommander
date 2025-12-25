import os
from .config import HISTORY_FILE

def init_history_if_needed():
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'w') as f:
            f.write("# Improvement History Log\n\n")
        print(f"📄 Created new history file: {HISTORY_FILE}")

def get_history_context():
    if not os.path.exists(HISTORY_FILE):
        return "(No history yet)"
    
    with open(HISTORY_FILE, 'r') as f:
        lines = f.readlines()
    
    if len(lines) <= 100:
        return "".join(lines)
    
    head = lines[:50]
    tail = lines[-50:]
    omission = ["\n... (Intermediate history omitted for brevity) ...\n"]
    return "".join(head + omission + tail)

def append_history_entry(cycle_num, analysis, modified_files, execution_summary, session_id):
    entry = f"\n\n---\n### [CYCLE] {cycle_num}\n"
    if session_id:
        entry += f"🔗 Session ID: {session_id}\n\n"
    
    entry += f"**Gemini Analysis (Final Summary):**\n{analysis}\n\n"
    
    entry += "**Modified Files:**\n"
    if modified_files:
        for f in modified_files:
            entry += f"- {f}\n"
    else:
        entry += "- (None)\n"
        
    entry += f"\n**Execution Summary:**\n\n```\n{execution_summary}\n```\n---"
    
    with open(HISTORY_FILE, 'a') as f:
        f.write(entry)
    print(f"📝 Cycle {cycle_num} recorded to history.")

