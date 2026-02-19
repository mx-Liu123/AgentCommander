import json
from .base import BaseAdapter

class GeminiAdapter(BaseAdapter):
    def build_command(self, prompt, session_id, model, yolo):
        cmd = ["gemini", "-o", "stream-json"]
        
        # Original logic: if cli_model and cli_model != binary ...
        # Here 'model' is the target model string.
        if model and model != "gemini" and not model.startswith("auto-gemini"):
             # auto-gemini is handled by the binary usually or passed as -m? 
             # In old logic: is_gemini_prefix = target_model.startswith("gemini") or "auto-gemini"
             # If it is auto-gemini-3, we pass it as -m auto-gemini-3
             cmd.extend(["-m", model])
        elif model and model.startswith("auto-gemini"):
             cmd.extend(["-m", model])
            
        if session_id:
            if session_id == "AUTO_RESUME":
                cmd.extend(["--resume", "latest"])
            else:
                cmd.extend(["--resume", session_id])
                
        if yolo:
            cmd.append("-y")
            
        return cmd

    def parse_output(self, stdout, original_session_id):
        new_session_id = None
        full_response = ""
        
        for line in stdout.splitlines():
            if not line.strip(): continue
            try:
                event = json.loads(line)
                event_type = event.get("type")
                
                if event_type == "init":
                    new_session_id = event.get("session_id")
                elif event_type == "message" and event.get("role") == "assistant":
                    content = event.get("content", "")
                    full_response += content
                    
            except json.JSONDecodeError:
                pass

        if not new_session_id and original_session_id:
            new_session_id = original_session_id

        return full_response, new_session_id
