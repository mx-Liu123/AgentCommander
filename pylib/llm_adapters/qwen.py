import json
from .base import BaseAdapter

class QwenAdapter(BaseAdapter):
    def build_command(self, prompt, session_id, model, yolo):
        cmd = ["qwen", "-o", "stream-json"]
        
        # Strip 'qwen:' prefix if present
        final_model = model
        if model and model.startswith("qwen:"):
            final_model = model[5:]
            
        if final_model and final_model != "qwen":
            cmd.extend(["-m", final_model])
            
        if session_id:
            if session_id == "AUTO_RESUME":
                cmd.append("-c")
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
                
                if event_type == "system" and event.get("subtype") == "init":
                    new_session_id = event.get("session_id")

                elif event_type == "assistant":
                    msg_obj = event.get("message", {})
                    content_list = msg_obj.get("content", [])
                    
                    text_chunk = ""
                    if isinstance(content_list, list):
                        for item in content_list:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_chunk += item.get("text", "")
                    
                    if text_chunk:
                        full_response += text_chunk

            except json.JSONDecodeError:
                pass
                
        if not new_session_id and original_session_id:
            new_session_id = original_session_id

        return full_response, new_session_id
