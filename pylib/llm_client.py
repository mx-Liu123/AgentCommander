import subprocess
import json
import os
import sys
from .config import GEMINI_MODEL, DEBUG

def call_gemini(prompt, session_id=None, timeout=None, model=None, cwd=None, yolo=True):
    """
    Wrapper for call_llm to maintain backward compatibility.
    """
    return call_llm(prompt, session_id, timeout, model, cwd, yolo)

def call_llm(prompt, session_id=None, timeout=None, model=None, cwd=None, yolo=True):
    """
    Unified LLM caller supporting Gemini, Qwen, and custom CLI wrappers.
    """
    target_model = model if model else GEMINI_MODEL
    
    # Defaults
    binary = "gemini"
    cli_model = target_model
    is_qwen = False

    # 1. Custom Format Parsing: custom:<cli>:<model_name>
    if target_model.startswith("custom:"):
        parts = target_model.split(":", 2) # Limit split to 2 in case model name has colons
        if len(parts) >= 3:
            binary = parts[1] # e.g. 'gemini', 'qwen', 'claude'
            cli_model = parts[2]
            if binary == "qwen": is_qwen = True
        else:
            # Fallback if format is weird
            cli_model = target_model[7:] # Just strip 'custom:'
            
    # 2. Legacy / Standard Logic
    else:
        is_explicit_qwen = target_model.startswith("qwen:")
        is_gemini_prefix = target_model.startswith("gemini") or target_model.startswith("auto-gemini")
        
        # Determine if Qwen
        if is_explicit_qwen or (target_model == "qwen") or (not is_gemini_prefix and target_model != "claude-cli"):
             is_qwen = True
             binary = "qwen"
        
        if target_model == "claude-cli": 
            binary = "claude"
            is_qwen = False
            
        # Strip prefix if explicit qwen
        if is_explicit_qwen:
            cli_model = target_model[5:]

    # Base command
    cmd = [binary, "-o", "stream-json"]
    
    # Model handling (-m)
    # Both Gemini and Qwen support -m for specifying model
    # Only skip if the model name is exactly the binary name (e.g. just 'qwen') which implies default
    if cli_model and cli_model != binary and cli_model != "claude-cli":
        cmd.extend(["-m", cli_model])
    
    run_kwargs = {
        "capture_output": True,
        "text": True,
        "env": os.environ
    }
    
    if cwd:
        run_kwargs["cwd"] = cwd

    if timeout:
        run_kwargs["timeout"] = timeout

    # Session Management Flags
    if session_id:
        if session_id == "AUTO_RESUME":
            # Auto-resume latest session
            if is_qwen:
                cmd.append("-c") # Qwen uses -c for continue
            else:
                cmd.extend(["--resume", "latest"]) # Gemini explicitly uses --resume latest
                
            # Changed to stdin to avoid E2BIG
            if yolo: cmd.append("-y")
            run_kwargs["input"] = prompt
        else:
            # Resume specific session
            cmd.extend(["--resume", session_id])
            # Changed to stdin to avoid E2BIG
            if yolo: cmd.append("-y")
            run_kwargs["input"] = prompt
    else:
        # New session: Pass prompt via stdin
        if yolo: cmd.append("-y")
        run_kwargs["input"] = prompt

    if DEBUG:
        sys.stderr.write(f"\n🐛 [DEBUG] ({binary}) PROMPT START -----------------\n")
        sys.stderr.write(prompt[:1000] + "... (truncated)" if len(prompt) > 1000 else prompt)
        sys.stderr.write(f"\n🐛 [DEBUG] ({binary}) PROMPT END -------------------\n")

    try:
        # Run command
        result = subprocess.run(cmd, **run_kwargs)
        
        # Session Recovery Logic (Simplified for brevity, similar to original)
        if result.returncode != 0 and session_id == "AUTO_RESUME":
            sys.stderr.write(f"⚠️ Auto-resume failed: {result.stderr.strip()}. Starting NEW session...\n")
            
            # Fallback: New Session
            new_cmd = [binary, "-o", "stream-json"]
            if not is_qwen and target_model: new_cmd.extend(["-m", target_model])
            if yolo: new_cmd.append("-y")
            
            new_kwargs = run_kwargs.copy()
            new_kwargs["input"] = prompt 
            
            final_result = subprocess.run(new_cmd, **new_kwargs)
            if final_result.returncode == 0:
                sys.stderr.write("✅ Started new session.\n")
                result = final_result
            else:
                sys.stderr.write(f"❌ New session creation failed: {final_result.stderr.strip()}\n")
                result = final_result

        elif result.returncode != 0 and session_id and "Session not found" in result.stderr:
            sys.stderr.write(f"⚠️ Session '{session_id}' not found. Attempting to attach to last active session...\n")
            
            # Retry logic... (omitted full complexity, falling back to new session usually safer)
            # For now, let's keep it simple or user might get confused by too many retries.
            pass

        stdout = result.stdout
        stderr = result.stderr
        
        if result.returncode != 0:
            err_msg = f"❌ {binary.upper()} CLI Error: {stderr.strip()}"
            print(err_msg) # Keep printing to stdout for CLI usage
            raise RuntimeError(err_msg)

        # Parse Stream-JSON Output
        if is_qwen:
            return parse_qwen_output(stdout, session_id)
        else:
            return parse_gemini_output(stdout, session_id)

    except subprocess.TimeoutExpired:
        err_msg = f"❌ {binary.upper()} Call Timed Out after {timeout}s"
        print(err_msg)
        raise RuntimeError(err_msg)
    except Exception as e:
        # If it's already our RuntimeError, re-raise
        if isinstance(e, RuntimeError) and str(e).startswith("❌"):
            raise e
        err_msg = f"❌ Failed to call {binary.upper()}: {e}"
        print(err_msg)
        raise RuntimeError(err_msg)

def parse_gemini_output(stdout, original_session_id):
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

def parse_qwen_output(stdout, original_session_id):
    new_session_id = None
    full_response = ""
    
    for line in stdout.splitlines():
        if not line.strip(): continue
        try:
            event = json.loads(line)
            event_type = event.get("type")
            
            # 1. Detect Session ID (Qwen: type='system', subtype='init')
            if event_type == "system" and event.get("subtype") == "init":
                new_session_id = event.get("session_id")

            # 2. Detect Content (Qwen: type='assistant', nested message structure)
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
