import subprocess
import os
import sys
from .config import GEMINI_MODEL, DEBUG
from . import llm_adapters

def call_gemini(prompt, session_id=None, timeout=None, model=None, cwd=None, yolo=True):
    """
    Wrapper for call_llm to maintain backward compatibility.
    """
    return call_llm(prompt, session_id, timeout, model, cwd, yolo)

def call_llm(prompt, session_id=None, timeout=None, model=None, cwd=None, yolo=True):
    """
    Unified LLM caller using Adapter Pattern.
    """
    target_model = model if model else GEMINI_MODEL
    
    # 1. Get Adapter
    adapter = llm_adapters.get_adapter(target_model)
    
    # 2. Clean Model Name (Handle custom: prefix)
    clean_model = target_model
    if target_model.startswith("custom:"):
        parts = target_model.split(":", 2)
        if len(parts) >= 3: clean_model = parts[2]
    
    # 3. Build Command
    cmd = adapter.build_command(prompt, session_id, clean_model, yolo)
    
    # 4. Prepare Execution Args
    run_kwargs = adapter.get_run_kwargs(prompt, os.environ)
    if cwd: run_kwargs["cwd"] = cwd
    if timeout: run_kwargs["timeout"] = timeout
    
    if DEBUG:
        sys.stderr.write(f"\n🐛 [DEBUG] Executing: {cmd}\n")
        if len(prompt) > 1000: sys.stderr.write(f"Prompt len: {len(prompt)}\n")

    try:
        # Execute
        result = subprocess.run(cmd, **run_kwargs)
        
        # 5. Session Recovery (Generic)
        if result.returncode != 0 and session_id == "AUTO_RESUME":
            sys.stderr.write(f"⚠️ Auto-resume failed: {result.stderr.strip()}. Retrying as NEW session...\n")
            
            # Rebuild command without session
            cmd_new = adapter.build_command(prompt, None, clean_model, yolo)
            result = subprocess.run(cmd_new, **run_kwargs)
            
            if result.returncode == 0:
                sys.stderr.write("✅ Started new session.\n")
        
        if result.returncode != 0:
            err_msg = result.stderr.strip() if result.stderr else "(No stderr)"
            # Check for specific session not found error to be friendly
            if "Session not found" in err_msg:
                 sys.stderr.write("⚠️ Session not found. (Recommendation: Reset session)\n")
            
            print(f"❌ CLI Error: {err_msg}")
            raise RuntimeError(f"CLI Error: {err_msg}")

        # 6. Parse Output
        return adapter.parse_output(result.stdout, session_id)

    except OSError as e:
        if e.errno == 7: # E2BIG
            sys.stderr.write(f"\n⚠️ [Robustness] Caught E2BIG. Retrying with truncated prompt...\n")
            cut_size = max(2000, int(len(prompt) * 0.1))
            new_prompt = prompt[:-cut_size] + "\n...[TRUNCATED]..."
            return call_llm(new_prompt, session_id, timeout, model, cwd, yolo)
        raise e
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"❌ Call Timed Out after {timeout}s")
    except Exception as e:
        if isinstance(e, RuntimeError): raise e
        raise RuntimeError(f"❌ Failed to call LLM: {e}")
