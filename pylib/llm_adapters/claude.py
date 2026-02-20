import json
from .base import BaseAdapter

class ClaudeAdapter(BaseAdapter):
    def build_command(self, prompt, session_id, model, yolo):
        # Claude CLI construction
        # -p: Print mode (non-interactive, one-shot)
        cmd = ["claude", "-p", "--output-format", "json"]
        
        cmd.append("--dangerously-skip-permissions")
        
        if yolo:
            cmd.extend(["--tools", "Bash,Write,Read", "--allowed-tools", "Write,Bash,Read"])
            
        if session_id and session_id != "AUTO_RESUME":
             cmd.extend(["-r", session_id])
             
        # If the user passed a specific model (not just "claude-cli"), pass it via --model
        if model and model != "claude-cli":
             cmd.extend(["--model", model])
             
        return cmd

    def parse_output(self, stdout, original_session_id):
        # Claude JSON output parsing
        try:
            data = json.loads(stdout)
            response = data.get("result", "")
            new_session_id = data.get("session_id", original_session_id)
            return response, new_session_id
        except:
            # Fallback for non-JSON output or errors
            return stdout, original_session_id
