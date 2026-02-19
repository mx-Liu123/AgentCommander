import abc
import os

class BaseAdapter(abc.ABC):
    @abc.abstractmethod
    def build_command(self, prompt, session_id, model, yolo):
        """Builds the CLI command list. Returns list of strings."""
        pass

    @abc.abstractmethod
    def parse_output(self, stdout, original_session_id):
        """Parses stdout to return (response_text, new_session_id)."""
        pass
        
    def get_run_kwargs(self, prompt, env):
        """Returns extra kwargs for subprocess.run (e.g. env, input)."""
        # Default implementation for Stdin based CLIs
        return {
            "capture_output": True,
            "text": True,
            "env": env,
            "input": prompt
        }
