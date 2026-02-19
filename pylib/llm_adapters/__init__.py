from .gemini import GeminiAdapter
from .qwen import QwenAdapter
from .claude import ClaudeAdapter
from .opencode import OpenCodeAdapter

def get_adapter(model_name):
    """Factory to get the correct adapter based on model name."""
    if not model_name:
        return GeminiAdapter() # Default
        
    # 1. Custom Format Parsing: custom:<cli>:<model_name>
    if model_name.startswith("custom:"):
        parts = model_name.split(":", 2)
        if len(parts) >= 2:
            cli_type = parts[1]
            if cli_type == "qwen": return QwenAdapter()
            if cli_type == "claude": return ClaudeAdapter()
            if cli_type == "opencode": return OpenCodeAdapter()
            return GeminiAdapter()

    # 2. Standard Prefix Detection
    if model_name.startswith("opencode"):
        return OpenCodeAdapter()
    
    if model_name.startswith("qwen:") or model_name == "qwen":
        return QwenAdapter()
        
    if model_name == "claude-cli":
        return ClaudeAdapter()
        
    # Default fallback (Gemini)
    return GeminiAdapter()
