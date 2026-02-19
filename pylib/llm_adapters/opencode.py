from .base import BaseAdapter

class OpenCodeAdapter(BaseAdapter):
    def build_command(self, prompt, session_id, model, yolo):
        cmd = ["opencode", "run"]
        
        if model:
            cmd.extend(["-m", model])
            
        if session_id:
            cmd.append("-c")
            
        return cmd

    def parse_output(self, stdout, original_session_id):
        return stdout.strip(), original_session_id
