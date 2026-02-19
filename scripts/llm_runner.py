import argparse
import os
import sys
from pathlib import Path

# Add project root to path
if 'AGENT_APP_ROOT' in os.environ:
    sys.path.append(os.environ['AGENT_APP_ROOT'])
else:
    sys.path.append(str(Path(__file__).parent.parent))

try:
    from pylib import llm_task
except ImportError:
    print("Error: Could not import pylib.llm_task")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-2.0-flash-exp")
    parser.add_argument("--whitelist", default="")
    parser.add_argument("--blacklist", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--lock-parent", action="store_true", help="Temporarily make parent directory read-only")
    parser.add_argument("--no-exec", default="", help="Comma-separated list of files to remove execute permission from")
    
    args = parser.parse_args()
    
    whitelist = [f.strip() for f in args.whitelist.split(',')] if args.whitelist else None
    blacklist = [f.strip() for f in args.blacklist.split(',')] if args.blacklist else None
    no_exec_list = [f.strip() for f in args.no_exec.split(',')] if args.no_exec else []

    print(f"[LLM Runner] Model: {args.model}", flush=True)
    print(f"[LLM Runner] Timeout: {args.timeout}s", flush=True)
    print(f"[LLM Runner] CWD: {args.cwd}", flush=True)
    print(f"[LLM Runner] Whitelist: {whitelist if whitelist else 'None (Read-Only)'}", flush=True)
    if no_exec_list:
        print(f"[LLM Runner] No-Exec List: {no_exec_list}", flush=True)
    if args.lock_parent:
        print(f"[LLM Runner] Lock Parent: Enabled", flush=True)
    
    # Read prompt from stdin
    prompt = sys.stdin.read()
    
    try:
        response, session_id = llm_task.run_task(
            prompt=prompt,
            model=args.model,
            cwd=args.cwd,
            permission_mode="whitelist" if whitelist else ("blacklist" if blacklist else "open"),
            whitelist=whitelist,
            blacklist=blacklist,
            allow_new_files=False, # Strict mode: Only allow new files if whitelisted
            timeout=args.timeout,
            session_id="AUTO_RESUME" if args.resume else None,
            yolo=True,
            lock_parent=args.lock_parent,
            no_exec_list=no_exec_list
        )
        
        print("[LLM Runner] Task Completed.", flush=True)
    except Exception as e:
        print(f"[LLM Runner] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
