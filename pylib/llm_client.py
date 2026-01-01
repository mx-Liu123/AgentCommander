import subprocess
import json
import os
import sys
from .config import GEMINI_MODEL, DEBUG

def call_gemini(prompt, session_id=None, timeout=None, model=None, cwd=None):
    # Always use stream-json to capture Session ID reliably
    cmd = ["gemini", "-o", "stream-json"]
    
    target_model = model if model else GEMINI_MODEL
    if target_model:
        cmd.extend(["-m", target_model])
    
    run_kwargs = {
        "capture_output": True,
        "text": True,
        "env": os.environ
    }
    
    if cwd:
        run_kwargs["cwd"] = cwd

    if timeout:
        run_kwargs["timeout"] = timeout

    if session_id:
        if session_id == "AUTO_RESUME":
            # Auto-resume latest session
            cmd.append("-r")
            cmd.extend(["-p", prompt])
            cmd.append("-y")
            run_kwargs["input"] = ""
        else:
            # Resume specific session
            cmd.extend(["--resume", session_id])
            cmd.extend(["-p", prompt])
            cmd.append("-y") # Enable YOLO mode for tools
            run_kwargs["input"] = "" # Prevent hanging by closing stdin
    else:
        # New session: Pass prompt via stdin
        cmd.append("-y") # Enable YOLO mode
        run_kwargs["input"] = prompt

    if DEBUG:
        sys.stderr.write(f"\n🐛 [DEBUG] ----------------- PROMPT START -----------------\\n")
        sys.stderr.write(prompt[:1000] + "... (truncated)" if len(prompt) > 1000 else prompt)
        sys.stderr.write(f"\n🐛 [DEBUG] ----------------- PROMPT END -------------------\\n")

    try:
        # Run command
        result = subprocess.run(cmd, **run_kwargs)
        
        # Session Recovery Logic
        if result.returncode != 0 and session_id == "AUTO_RESUME":
            sys.stderr.write(f"⚠️ Auto-resume failed: {result.stderr.strip()}. Starting NEW session...\n")
            
            # Fallback: New Session
            new_cmd = ["gemini", "-o", "stream-json"]
            if target_model: new_cmd.extend(["-m", target_model])
            new_cmd.append("-y")
            
            # Update kwargs: prompt goes to stdin for new session
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
            
            # Retry 1: Auto-resume (-r without ID)
            retry_cmd = ["gemini", "-o", "stream-json"]
            if target_model: retry_cmd.extend(["-m", target_model])
            
            # Auto-resume flags
            retry_cmd.append("-r") 
            retry_cmd.extend(["-p", prompt])
            retry_cmd.append("-y")
            
            # Use same run_kwargs (input is empty string for resume mode)
            retry_result = subprocess.run(retry_cmd, **run_kwargs)
            
            if retry_result.returncode == 0:
                sys.stderr.write("✅ Successfully attached to last session.\n")
                result = retry_result
            else:
                sys.stderr.write(f"⚠️ Auto-resume failed: {retry_result.stderr.strip()}. Starting NEW session...\n")
                
                # Retry 2: New Session
                new_cmd = ["gemini", "-o", "stream-json"]
                if target_model: new_cmd.extend(["-m", target_model])
                new_cmd.append("-y")
                
                # Update kwargs: prompt goes to stdin for new session
                new_kwargs = run_kwargs.copy()
                new_kwargs["input"] = prompt 
                
                final_result = subprocess.run(new_cmd, **new_kwargs)
                if final_result.returncode == 0:
                    sys.stderr.write("✅ Started new session.\n")
                    result = final_result
                else:
                    sys.stderr.write(f"❌ New session creation failed: {final_result.stderr.strip()}\n")
                    # result remains as the failed attempt (or we could set it to final_result)
                    result = final_result 

        stdout = result.stdout
        stderr = result.stderr
        
        if result.returncode != 0:
            print(f"❌ Gemini CLI Error: {stderr}")
            # Attempt to return partial stdout if available, but usually error means failure
            return stdout, None

        # Parse Stream-JSON Output
        new_session_id = None
        full_response = ""
        
        for line in stdout.splitlines():
            if not line.strip(): continue
            try:
                event = json.loads(line)
                event_type = event.get("type")
                
                if event_type == "init":
                    # Capture Session ID from init event
                    new_session_id = event.get("session_id")
                    
                elif event_type == "message" and event.get("role") == "assistant":
                    # Aggregate assistant response
                    content = event.get("content", "")
                    full_response += content
                    
            except json.JSONDecodeError:
                pass # Ignore malformed lines

        # Fallback: If we resumed a session, we might want to keep using the old ID 
        # if the CLI didn't emit a new one (though init event should always be there).
        if not new_session_id and session_id:
            new_session_id = session_id

        return full_response, new_session_id

    except subprocess.TimeoutExpired:
        print(f"❌ Gemini Call Timed Out after {timeout}s")
        return "[TIMEOUT]", None # Return specific marker
    except Exception as e:
        print(f"❌ Failed to call Gemini: {e}")
        return "", None