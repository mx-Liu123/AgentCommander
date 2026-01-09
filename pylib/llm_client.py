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
    Unified LLM caller supporting both Gemini and Qwen CLIs.
    """
    # Determine which CLI to use
    target_model = model if model else GEMINI_MODEL
    is_qwen = target_model and ("qwen" in target_model.lower() or target_model == "qwen")
    
    binary = "qwen" if is_qwen else "gemini"
    
    # Base command
    cmd = [binary, "-o", "stream-json"]
    
    # Model handling
    if not is_qwen and target_model:
        # Gemini uses -m
        cmd.extend(["-m", target_model])
    elif is_qwen:
        # Qwen ignores -m or uses default if not specified
        pass 
    
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
                cmd.append("-r") # Gemini uses -r
                
            cmd.extend(["-p", prompt])
            if yolo: cmd.append("-y")
            run_kwargs["input"] = ""
        else:
            # Resume specific session
            cmd.extend(["--resume", session_id])
            cmd.extend(["-p", prompt])
            if yolo: cmd.append("-y")
            run_kwargs["input"] = ""
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
            print(f"❌ {binary.upper()} CLI Error: {stderr}")
            return stdout, None

        # Parse Stream-JSON Output
        if is_qwen:
            return parse_qwen_output(stdout, session_id)
        else:
            return parse_gemini_output(stdout, session_id)

    except subprocess.TimeoutExpired:
        print(f"❌ {binary.upper()} Call Timed Out after {timeout}s")
        return "[TIMEOUT]", None
    except Exception as e:
        print(f"❌ Failed to call {binary.upper()}: {e}")
        return "", None

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
