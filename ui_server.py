import os
import sys
import json
import glob
import multiprocessing
import queue
import base64
import subprocess
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from datetime import datetime

# Add current dir to path
sys.path.append(os.getcwd())
import agent_service

# Setup Flask
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

# Global State
agent_process = None
script_process = None # For running auxiliary scripts
log_queue = multiprocessing.Queue()
stop_event = multiprocessing.Event()
recent_logs = []

# ... (Existing Code) ...

def scan_scripts_with_config(root_dir):
    """Scans for .sh files with <GEMINI_UI_CONFIG> block."""
    scripts = []
    try:
        # We always scan from the APP_ROOT/scripts to ensure consistency
        search_path = Path(os.getcwd()) / "scripts"
        if not search_path.exists(): 
            print(f"Debug: scripts folder not found at {search_path}")
            return []
        
        for path in search_path.rglob("*.sh"):
            try:
                content = path.read_text(encoding='utf-8')
                start_marker = "<GEMINI_UI_CONFIG>"
                end_marker = "</GEMINI_UI_CONFIG>"
                
                s_idx = content.find(start_marker)
                e_idx = content.find(end_marker)
                
                if s_idx != -1 and e_idx != -1:
                    # Extract everything between markers
                    json_block = content[s_idx + len(start_marker):e_idx]
                    
                    # Clean lines: remove leading '#' and whitespace
                    lines = []
                    for line in json_block.splitlines():
                        # Remove leading '#' if present, then strip whitespace
                        clean_line = line.strip()
                        if clean_line.startswith('#'):
                            clean_line = clean_line[1:].strip()
                        if clean_line:
                            lines.append(clean_line)
                    
                    clean_json = "".join(lines)
                    config = json.loads(clean_json)
                    
                    # Store path relative to project root for execution
                    config['path'] = os.path.relpath(str(path), os.getcwd())
                    scripts.append(config)
                    print(f"✅ Loaded script config: {config['name']} from {config['path']}")
            except Exception as e:
                print(f"❌ Error parsing script {path}: {e}")
                
    except Exception as e:
        print(f"❌ Error scanning scripts: {e}")
        
    return scripts

def script_runner_thread(cmd, env_vars, cwd):
    """Runs a script and pipes output to socketio."""
    global script_process
    
    # Merge env
    full_env = os.environ.copy()
    
    # Ensure all env vars are strings (subprocess requires string values)
    clean_env_vars = {k: str(v) for k, v in env_vars.items()}
    full_env.update(clean_env_vars)
    full_env["NON_INTERACTIVE"] = "1"
    full_env["AGENT_APP_ROOT"] = os.getcwd() # Inject App Root for scripts to reference
    
    try:
        # Use Popen to capture output in real-time
        # Use pty to force line buffering (optional, but subprocess usually buffers)
        # For simplicity, use standard PIPE and iterate lines
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=full_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Merge stderr to stdout
            text=True,
            bufsize=1 # Line buffered
        )
        script_process = process
        
        socketio.emit('script_status', {'status': 'running', 'pid': process.pid})
        
        for line in process.stdout:
            socketio.emit('script_log', {'data': line})
            
        process.wait()
        rc = process.returncode
        
        if rc == 0:
            socketio.emit('script_log', {'data': '\n✅ Script finished successfully.\n'})
            socketio.emit('script_status', {'status': 'completed', 'code': 0})
        else:
            socketio.emit('script_log', {'data': f'\n❌ Script failed with exit code {rc}.\n'})
            socketio.emit('script_status', {'status': 'error', 'code': rc})
            
    except Exception as e:
        socketio.emit('script_log', {'data': f'\n❌ Script execution error: {e}\n'})
        socketio.emit('script_status', {'status': 'error'})
    finally:
        script_process = None

@app.route('/api/scripts/list', methods=['GET'])
def list_scripts_api():
    return jsonify(scan_scripts_with_config(CURRENT_ROOT_DIR))

@app.route('/api/scripts/run', methods=['POST'])
def run_script_api():
    global script_process
    if script_process and script_process.poll() is None:
         return jsonify({"error": "A script is already running"}), 400
         
    data = request.json
    script_rel_path = data.get('path')
    env_vars = data.get('env', {})
    
    if not script_rel_path: return jsonify({"error": "Path required"}), 400
    
    # Scripts are always relative to the APP ROOT (where ui_server.py is), not the task root
    script_path = (Path(os.getcwd()) / script_rel_path).resolve()
    
    if not script_path.exists(): 
        print(f"Debug: Script not found at {script_path}")
        return jsonify({"error": f"Script not found at {script_path}"}), 404
    
    # Security Check: Script must be within app root
    if str(Path(os.getcwd()).resolve()) not in str(script_path):
         return jsonify({"error": "Access denied: Script outside project root"}), 403

    # Start Background Thread
    # We still run the script inside CURRENT_ROOT_DIR so it creates files in the right place
    cmd = ["bash", str(script_path)]
    thread = threading.Thread(target=script_runner_thread, args=(cmd, env_vars, CURRENT_ROOT_DIR))
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "started"})

@app.route('/api/scripts/stop', methods=['POST'])
def stop_script_api():
    global script_process
    if script_process and script_process.poll() is None:
        script_process.terminate()
        return jsonify({"status": "terminated"})
    return jsonify({"status": "no_process"})

# Default root dir
CURRENT_ROOT_DIR = os.path.join(os.getcwd(), 'Find_Transform')
if not os.path.exists(CURRENT_ROOT_DIR):
    CURRENT_ROOT_DIR = os.getcwd()

def get_branches(root_dir):
    try:
        root = Path(root_dir)
        branches = sorted(root.glob("Branch*"), key=lambda p: p.name)
        return [b.name for b in branches if b.is_dir()]
    except:
        return []

def background_log_emitter():
    global recent_logs
    while True:
        try:
            msg = log_queue.get(timeout=0.1)
            
            # Add timestamp
            if 'time' not in msg:
                msg['time'] = datetime.now().strftime('%H:%M:%S')

            # Store history
            if msg.get('type') != 'status':
                recent_logs.append(msg)
                if len(recent_logs) > 1000: recent_logs.pop(0)
            
            if msg.get('type') == 'status':
                 socketio.emit('status', msg)
            else:
                 socketio.emit('log', msg)
        except queue.Empty:
            socketio.sleep(0.1)
        except Exception as e:
            print(f"Log emitter error: {e}")
            socketio.sleep(1)

import shutil
from pylib import llm_client

# Constants
APP_ROOT = Path(os.getcwd()).resolve()
CACHE_DIR = Path("pylib/.cache")
CACHE_FILE = CACHE_DIR / "current_graph.json"
DEFAULT_TEMPLATE = Path("pylib/default_graph.json")
CONFIG_FILE = APP_ROOT / "config.json"

# Load Config
def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            # If config.json exists but is invalid JSON
            print(f"Error loading config from existing file: {e}")
            raise ValueError(f"Error loading config.json. Please check its format. Error: {e}")
    else:
        # If config.json does not exist, do not create it.
        # Instead, raise an error and tell the user to rename config_template.json.
        raise FileNotFoundError("config.json not found. Please rename config_template.json to config.json and configure it.")

config_data = load_config()
CURRENT_ROOT_DIR = config_data.get('root_dir', os.getcwd())
if not os.path.exists(CURRENT_ROOT_DIR):
    CURRENT_ROOT_DIR = os.getcwd()

# Allow override from UI_CONFIG_FILE for legacy support or local override
UI_CONFIG_FILE = Path(".ui_server_config.json")
if UI_CONFIG_FILE.exists():
    try:
        with open(UI_CONFIG_FILE, 'r') as f:
            saved_conf = json.load(f)
            if 'root_dir' in saved_conf and os.path.exists(saved_conf['root_dir']):
                CURRENT_ROOT_DIR = saved_conf['root_dir']
    except: pass

def ensure_cache():
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not CACHE_FILE.exists():
        if DEFAULT_TEMPLATE.exists():
            shutil.copy(DEFAULT_TEMPLATE, CACHE_FILE)
        else:
            # Create empty structure if default missing
            with open(CACHE_FILE, 'w') as f:
                json.dump({"nodes": [], "edges": []}, f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    global agent_process
    is_running = agent_process is not None and agent_process.is_alive()
    return jsonify({
        "isRunning": is_running,
        "logs": recent_logs
    })

@app.route('/api/config', methods=['GET'])
def get_config():
    # Load fresh config
    try:
        conf = load_config()
    except (FileNotFoundError, ValueError) as e:
        # If config.json is missing or invalid, return error to frontend
        return jsonify({"error": str(e)}), 500
    
    # Load default template for initial structure (nodes/edges)
    default_workflow = {}
    if DEFAULT_TEMPLATE.exists():
        try:
            with open(DEFAULT_TEMPLATE, 'r') as f:
                data = json.load(f)
                if 'nodes' in data: default_workflow = data
                elif 'workflow' in data: default_workflow = data['workflow']
        except: pass
    
    # Merge config into response
    # Start with full conf to include mode, resume_branch_id, etc.
    response = conf.copy()
    response.update({
        "root_dir": CURRENT_ROOT_DIR,
        "branches": get_branches(CURRENT_ROOT_DIR),
        "global_vars": conf.get('global_vars', {}),
        "workflow": "pylib/.cache/current_graph.json"
    })
    return jsonify(response)

@app.route('/api/save_config', methods=['POST'])
def save_config_api():
    global CURRENT_ROOT_DIR # Moved to top
    try:
        new_conf = request.json
        
        # Sanitize: Ensure 'workflow' is NOT saved to config.json (Hardcoded now)
        if 'workflow' in new_conf:
            del new_conf['workflow']

        # Merge with existing to avoid losing keys not sent
        current_conf = load_config()
        current_conf.update(new_conf)
        
        # If root_dir changed, update global state
        if 'root_dir' in new_conf and os.path.exists(new_conf['root_dir']):
             CURRENT_ROOT_DIR = new_conf['root_dir']
             
        with open(CONFIG_FILE, 'w') as f:
            json.dump(current_conf, f, indent=2)
            
        return jsonify({"status": "ok", "config": current_conf})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/save_config_as', methods=['POST'])
def save_config_as_api():
    global CURRENT_ROOT_DIR # Moved to top
    try:
        data = request.json
        path_str = data.get('path')
        new_conf = data.get('config')
        
        if not path_str or not new_conf:
            return jsonify({"error": "Path and config required"}), 400
            
        target_path = Path(os.getcwd()) / path_str
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. Save to Target
        with open(target_path, 'w') as f:
            json.dump(new_conf, f, indent=2)
            
        # 2. Update Active Config (Sync)
        # We also treat 'Save As' as 'Switch To', so we update the main config.json
        current_conf = load_config()
        current_conf.update(new_conf)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(current_conf, f, indent=2)
            
        return jsonify({"status": "ok", "saved_path": str(target_path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/load_config_from', methods=['POST'])
def load_config_from_api():
    global CURRENT_ROOT_DIR # Moved to top
    try:
        data = request.json
        path_str = data.get('path')
        if not path_str: return jsonify({"error": "Path required"}), 400
        
        target_path = Path(os.getcwd()) / path_str
        if not target_path.exists():
            return jsonify({"error": "File not found"}), 404
            
        with open(target_path, 'r') as f:
            new_conf = json.load(f)
            
        # Clean workflow if present (Hardcoded path usage)
        if 'workflow' in new_conf: del new_conf['workflow']
            
        # Update Active Config
        with open(CONFIG_FILE, 'w') as f:
            json.dump(new_conf, f, indent=2)
            
        # Update Global State if root_dir changed
        if 'root_dir' in new_conf and os.path.exists(new_conf['root_dir']):
             CURRENT_ROOT_DIR = new_conf['root_dir']
             
        return jsonify({"status": "ok", "config": new_conf})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/set_root', methods=['POST'])
def set_root():
    global CURRENT_ROOT_DIR
    data = request.json
    new_root = data.get('path')
    
    if not new_root: return jsonify({"error": "Path required"}), 400

    if os.path.exists(new_root):
        CURRENT_ROOT_DIR = new_root
        # Persist to config.json
        try:
            current_conf = load_config()
            current_conf['root_dir'] = CURRENT_ROOT_DIR
            with open(CONFIG_FILE, 'w') as f:
                json.dump(current_conf, f, indent=2)
        except Exception as e:
            print(f"Error saving config.json: {e}")
        
        return jsonify({"status": "ok", "branches": get_branches(new_root)})
    return jsonify({"error": f"Path does not exist: {new_root}"}), 400

@app.route('/api/workflow', methods=['GET', 'POST'])
def workflow_api():
    ensure_cache()
    
    if request.method == 'GET':
        # Always return the CACHE content by default
        try:
            with open(CACHE_FILE, 'r') as f: return jsonify(json.load(f))
        except Exception as e: return jsonify({"error": str(e)}), 500

    elif request.method == 'POST':
        data = request.json
        action = data.get('action', 'update')
        
        try:
            if action == 'update':
                # content from UI -> Cache
                # We expect data to contain: content (graph), global_vars, llm_changeable_vars
                save_data = data.get('content', {}) # This is the graph (nodes/edges)
                
                # Defensive Check: Prevent overwriting cache with empty/invalid graph
                if not isinstance(save_data, dict) or 'nodes' not in save_data: # Allow empty nodes list if valid structure, but structure must exist
                     print(f"WARNING: Refusing to save invalid graph to cache. Data type: {type(save_data)}")
                     return jsonify({"error": "Invalid graph data: missing 'nodes'"}), 400
                
                # Decoupling: Do NOT save global_vars/llm_changeable_vars to workflow file
                # They are managed by config.json
                
                with open(CACHE_FILE, 'w') as f: json.dump(save_data, f, indent=2)
                
            elif action == 'save_as':
                # UI Data -> Target Path AND Cache
                path_str = data.get('path')
                if not path_str: return jsonify({"error": "Path required"}), 400
                
                target_path = Path(CURRENT_ROOT_DIR) / path_str
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Construct data
                save_data = data.get('content', {})
                # Decoupling: Do NOT save global_vars/llm_changeable_vars to workflow file
                
                # Write to Target
                with open(target_path, 'w') as f: json.dump(save_data, f, indent=2)
                
                # Update Cache as well (Sync)
                with open(CACHE_FILE, 'w') as f: json.dump(save_data, f, indent=2)
                
                return jsonify({"status": "ok", "saved_path": str(target_path.resolve())})
                
            elif action == 'load':
                # Source Path -> Cache
                src_path_str = data.get('path')
                if src_path_str == 'default':
                    src_path = DEFAULT_TEMPLATE
                elif src_path_str == 'parallel':
                    src_path = Path("pylib/parallel_graph.json")
                else:
                    src_path = Path(CURRENT_ROOT_DIR) / src_path_str
                
                if src_path.exists():
                    shutil.copy(src_path, CACHE_FILE)
                else:
                    return jsonify({"error": f"File not found: {src_path_str}"}), 404
                    
            elif action == 'reset':
                if DEFAULT_TEMPLATE.exists():
                    shutil.copy(DEFAULT_TEMPLATE, CACHE_FILE)
            
            return jsonify({"status": "ok"})
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.json
    user_msg = data.get('message', '')
    session_id = data.get('session_id')
    model = data.get('model', 'auto')
    target_cwd = data.get('cwd')
    yolo_param = data.get('yolo', True)
    
    # Ensure cache exists so AI finds the file (only critical for Workflow mode, but harmless otherwise)
    ensure_cache()
    
    if target_cwd:
        # File Explorer Mode
        cwd_str = str(Path(target_cwd).resolve())
        if not os.path.exists(cwd_str):
             return jsonify({"error": f"Directory not found: {cwd_str}"}), 404
             
        context_instruction = (
            f"You are an AI Assistant working in directory: {cwd_str}. "
            "You can read/write files and execute commands in this directory. "
            f"\nUser Request: {user_msg}"
        )
    else:
        # Workflow Editor Mode (Default)
        cwd_str = str(CACHE_DIR.resolve())
        context_instruction = (
            "You are the Workflow Editor Assistant. "
            "You are working in a directory containing the active workflow file 'current_graph.json'. "
            "User requests will be about modifying this workflow. "
            "1. Always read 'current_graph.json' first to understand the structure (nodes, edges, IDs). "
            "2. Directly modify 'current_graph.json' using 'replace' or 'write_file'. "
            "3. Respond concisely. "
            f"\nUser Request: {user_msg}"
        )
    
    try:
        response, new_sid = llm_client.call_gemini(
            prompt=context_instruction,
            session_id=session_id,
            model=model,
            cwd=cwd_str,
            timeout=600,
            yolo=yolo_param
        )
        return jsonify({"response": response, "session_id": new_sid})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def scan_directory(path):
    """Scans a single directory non-recursively."""
    path_obj = Path(path)
    if not path_obj.exists(): return []
    
    results = []
    try:
        with os.scandir(path_obj) as entries:
            for entry in entries:
                if entry.name.startswith('.') or entry.name == '__pycache__': continue
                
                item = {
                    "name": entry.name,
                    "path": entry.path, # Absolute path for backend usage, or relative? Let's use relative for API consistency if possible, but absolute works if we map it back.
                    # Ideally, frontend sends relative path.
                    # Let's send relative path from CURRENT_ROOT_DIR for ID purposes
                    "rel_path": os.path.relpath(entry.path, CURRENT_ROOT_DIR),
                    "type": "folder" if entry.is_dir() else "file"
                }
                # For folder, we mark children as empty list to indicate it CAN have children
                if entry.is_dir():
                    item["children"] = [] 
                
                results.append(item)
                
        results.sort(key=lambda x: (x['type'] != 'folder', x['name']))
    except Exception as e:
        print(f"Scan error: {e}")
        
    return results

@app.route('/api/files', methods=['GET'])
def list_files_api():
    path_param = request.args.get('path', '')
    
    # 1. Determine target path
    if path_param == '':
        target_path = Path(CURRENT_ROOT_DIR)
    else:
        p_param = Path(path_param)
        p_root = Path(CURRENT_ROOT_DIR)
        
        # If absolute, use as is (pathlib / operator does this, but explicit logic is clearer)
        if p_param.is_absolute():
            target_path = p_param
        # If it matches root or starts with root (naive string check for relative paths)
        elif str(p_param) == str(p_root) or str(p_param).startswith(str(p_root) + os.sep):
            target_path = p_param
        else:
            # Assume it is a sub-path relative to root
            target_path = p_root / path_param

    # 2. Security & Existence Check
    try:
        target_path_abs = target_path.resolve()
        root_path_abs = Path(CURRENT_ROOT_DIR).resolve()
        
        # Security: Must be inside root or equal to root
        if target_path_abs != root_path_abs and root_path_abs not in target_path_abs.parents:
             return jsonify({"error": "Access denied"}), 403
             
        if not target_path.exists():
             # Fallback: Maybe it WAS a relative path but our heuristic failed?
             # Try forcing append one last time if we didn't already
             retry_path = Path(CURRENT_ROOT_DIR) / path_param
             if retry_path.exists() and retry_path.resolve() == target_path_abs:
                 # It was the same path, really doesn't exist
                 return jsonify({"error": f"Path not found: {target_path}"}), 404
             elif retry_path.exists():
                 # Oh, appending worked! Use that.
                 target_path = retry_path
             else:
                 return jsonify({"error": f"Path not found: {target_path}"}), 404

    except Exception as e:
        return jsonify({"error": f"Invalid path resolution: {e}"}), 400

    if path_param == '':
        # Root request wrapper
        return jsonify({
            "name": Path(CURRENT_ROOT_DIR).name,
            "type": "folder",
            "path": str(Path(CURRENT_ROOT_DIR)),
            "rel_path": "",
            "children": scan_directory(target_path)
        })
    else:
        return jsonify(scan_directory(target_path))

@app.route('/api/file_content', methods=['GET'])
def get_file_content():
    path_str = request.args.get('path')
    if not path_str: return jsonify({"error": "Path required"}), 400
    
    # Improved Path Resolution
    p_param = Path(path_str)
    p_root = Path(CURRENT_ROOT_DIR)
    
    full_path = None
    
    if p_param.is_absolute():
        full_path = p_param.resolve()
    elif str(p_param) == str(p_root) or str(p_param).startswith(str(p_root) + os.sep):
        full_path = p_param.resolve()
    else:
        full_path = (p_root / p_param).resolve()
    
    print(f"DEBUG: Request content for {path_str}")
    print(f"DEBUG: Full path resolved to {full_path}")
    
    # Security check
    if str(Path(CURRENT_ROOT_DIR).resolve()) not in str(full_path):
         return jsonify({"error": "Access denied"}), 403
         
    if not full_path.exists():
        return jsonify({"error": "File not found"}), 404
        
    # Read content
    try:
        # Check for image extension to prioritize binary read
        is_image = full_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.bmp']
        
        if is_image:
            with open(full_path, 'rb') as f:
                content = base64.b64encode(f.read()).decode('utf-8')
            return jsonify({"content": content, "encoding": "base64"})

        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({"content": content, "encoding": "utf-8"})
    except UnicodeDecodeError:
        # Fallback for other binary files (non-image but binary)
        try:
             with open(full_path, 'rb') as f:
                content = base64.b64encode(f.read()).decode('utf-8')
             return jsonify({"content": content, "encoding": "base64"})
        except Exception as e:
             return jsonify({"content": f"[Error reading binary file: {e}]"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/scan_progress', methods=['GET'])
def scan_progress_api():
    nodes = []
    edges = []
    
    # 1. Scan for Branch*/exp*
    root_path = Path(CURRENT_ROOT_DIR)
    if not root_path.exists(): return jsonify({"nodes": [], "edges": []})
    
    exp_dirs = sorted(root_path.glob("Branch*/exp*"))
    
    node_map = {} # path -> node_id
    branch_info = {} # branch -> {name: 'MyBranch', hint: 'Focus on...'}
    
    for exp_dir in exp_dirs:
        try:
            name = exp_dir.name
            rel_path = exp_dir.relative_to(root_path)
            branch = exp_dir.parent.name
            
            # Read history
            hist_path = exp_dir / "history.json"
            metrics = "N/A"
            is_improved = False
            parent_exp = None
            full_history = {}
            
            if hist_path.exists():
                try:
                    with open(hist_path, 'r') as f:
                        full_history = json.load(f)
                        
                        # Collect Branch Info (from any node, prefer first)
                        if 'branch_name' in full_history and branch not in branch_info:
                             branch_info[branch] = full_history['branch_name']
                        
                        # Extract Metrics (heuristic)
                        if 'metrics' in full_history and full_history['metrics']:
                             metrics = str(full_history['metrics'])[:50]
                        
                        # Check experiment_result for more details
                        if metrics == "N/A" and 'experiment_result' in full_history and isinstance(full_history['experiment_result'], list) and len(full_history['experiment_result']) > 0:
                            last_res = full_history['experiment_result'][-1]
                            # Handle stringified JSON in history
                            if isinstance(last_res, str):
                                try:
                                    last_res = json.loads(last_res)
                                except: pass
                            
                            if isinstance(last_res, dict):
                                ana = last_res.get('analysis', '')
                                if not ana: ana = str(last_res)
                                metrics = ana[:50] + "..."
                                is_improved = last_res.get('improved', is_improved)
                            else:
                                metrics = str(last_res)[:50] + "..."
                                
                        if metrics == "N/A" and 'result_analysis' in full_history and full_history['result_analysis']:
                             metrics = str(full_history['result_analysis'])[:50] + "..."
                            
                        if 'if_improved' in full_history: is_improved = full_history['if_improved']
                        parent_exp = full_history.get('parent_exp')
                except Exception as e:
                    print(f"Error reading {hist_path}: {e}")

            node_id = str(rel_path)
            node_map[name] = node_id # Map short name to full rel path for parent linking if needed
            
            nodes.append({
                "id": node_id,
                "name": name,
                "branch": branch,
                "metrics": metrics,
                "is_improved": is_improved,
                "full_history": full_history,
                "parent_exp_raw": parent_exp
            })
            
        except Exception as e:
            print(f"Error processing {exp_dir}: {e}")

    # 2. Build Edges
    for node in nodes:
        parent_raw = node['parent_exp_raw']
        if parent_raw:
            # Try to resolve parent
            # parent_raw could be absolute path, relative path, or just folder name
            parent_id = None
            
            # Strategy 1: Try as node_id (rel path) directly
            if parent_raw in [n['id'] for n in nodes]:
                parent_id = parent_raw
            # Strategy 2: Try as folder name (e.g. "exp1.1.1")
            elif parent_raw in node_map:
                parent_id = node_map[parent_raw]
            # Strategy 3: Try to find by path suffix
            else:
                 for n in nodes:
                     if str(parent_raw).endswith(n['name']): # loose match
                         parent_id = n['id']
                         break
            
            if parent_id:
                edges.append({"source": parent_id, "target": node['id']})
            else:
                print(f"Warning: Parent '{parent_raw}' not found for node '{node['name']}'")

    return jsonify({"nodes": nodes, "edges": edges, "branch_info": branch_info})

@app.route('/api/save_file', methods=['POST'])
def save_file_api():
    try:
        data = request.json
        rel_path = data.get('path')
        content = data.get('content')
        
        if not rel_path: return jsonify({"error": "No path provided"}), 400
        
        # Improved Path Resolution
        p_param = Path(rel_path)
        p_root = Path(CURRENT_ROOT_DIR)
        
        full_path = None
        if p_param.is_absolute():
            full_path = p_param.resolve()
        elif str(p_param) == str(p_root) or str(p_param).startswith(str(p_root) + os.sep):
            full_path = p_param.resolve()
        else:
            full_path = (p_root / p_param).resolve()
        
        # Security check
        if str(Path(CURRENT_ROOT_DIR).resolve()) not in str(full_path):
             return jsonify({"error": "Access denied"}), 403
             
        if not full_path.parent.exists():
            return jsonify({"error": "Parent directory does not exist"}), 400
            
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/update_vars', methods=['POST'])
def update_vars_api():
    try:
        data = request.json
        vars_dict = data.get('global_vars', {})
        # Save to runtime file that agent monitors
        with open(Path(CURRENT_ROOT_DIR) / 'runtime_vars.json', 'w') as f:
            json.dump(vars_dict, f)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/start', methods=['POST'])
def start_agent():
    global agent_process
    ensure_cache() # Ensure default cache exists if targeted
    if agent_process and agent_process.is_alive():
        return jsonify({"error": "Agent already running"}), 400

    config = request.json
    print(f"DEBUG: start_agent received config. workflow type: {type(config.get('workflow'))}")
    
    # FORCE HARDCODED WORKFLOW PATH
    # We ignore whatever the frontend or config.json says about 'workflow' path.
    # We always use the active graph in the cache.
    wf_path_str = "pylib/.cache/current_graph.json"
    wf_path = (APP_ROOT / wf_path_str).resolve()
    
    if not wf_path.exists():
        print(f"DEBUG: {wf_path} not found. Trying default.")
        wf_path = (APP_ROOT / "pylib/default_graph.json").resolve()
        
    if not wf_path.exists():
        return jsonify({"error": f"Workflow file not found at {wf_path}"}), 400
            
    try:
        with open(wf_path, 'r', encoding='utf-8') as f:
            config['workflow'] = json.load(f)
            print(f"DEBUG: Loaded workflow from HARDCODED path: {wf_path}")
            print(f"DEBUG: Nodes count: {len(config['workflow'].get('nodes', []))}")
    except Exception as e:
        return jsonify({"error": f"Failed to load workflow file: {e}"}), 400

    # Inject environment python path if not set by user
    if 'venv' not in config.get('global_vars', {}):
        if 'global_vars' not in config: config['global_vars'] = {}
        config['global_vars']['venv'] = sys.executable 
    
    stop_event.clear()
    
    agent_process = multiprocessing.Process(
        target=agent_service.agent_process_wrapper,
        args=(CURRENT_ROOT_DIR, config, log_queue, stop_event)
    )
    agent_process.start()
    return jsonify({"status": "started", "pid": agent_process.pid})

@app.route('/api/stop', methods=['POST'])
def stop_agent():
    stop_type = request.args.get('type', 'force')
    
    log_queue.put({"type": "log", "data": "💀 Force stop requested."})
    if agent_process and agent_process.is_alive():
        stop_event.set() 
        agent_process.join(timeout=1.0)
        if agent_process.is_alive(): agent_process.terminate()
        socketio.emit('status', {'type': 'status', 'data': 'stopped'})
    return jsonify({'status': 'stopped', 'mode': 'force'})

@app.route('/api/delete_path', methods=['DELETE'])
def delete_path_api():
    try:
        data = request.json
        rel_path = data.get('path')
        if not rel_path: return jsonify({"error": "Path required"}), 400
        
        # Improved Path Resolution
        p_param = Path(rel_path)
        p_root = Path(CURRENT_ROOT_DIR)
        
        full_path = None
        if p_param.is_absolute():
            full_path = p_param.resolve()
        elif str(p_param) == str(p_root) or str(p_param).startswith(str(p_root) + os.sep):
            full_path = p_param.resolve()
        else:
            full_path = (p_root / p_param).resolve()
        
        # Security check
        if str(Path(CURRENT_ROOT_DIR).resolve()) not in str(full_path):
             return jsonify({"error": "Access denied"}), 403
             
        if not full_path.exists():
            return jsonify({"error": "Path not found"}), 404
            
        if full_path.is_dir():
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)
            
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/rename_path', methods=['POST'])
def rename_path_api():
    try:
        data = request.json
        old_rel = data.get('old_path')
        new_rel = data.get('new_path')
        
        if not old_rel or not new_rel: return jsonify({"error": "Paths required"}), 400
        
        p_root = Path(CURRENT_ROOT_DIR)

        # Resolve Old
        p_old = Path(old_rel)
        if p_old.is_absolute(): old_path = p_old.resolve()
        elif str(p_old) == str(p_root) or str(p_old).startswith(str(p_root) + os.sep): old_path = p_old.resolve()
        else: old_path = (p_root / p_old).resolve()

        # Resolve New
        p_new = Path(new_rel)
        if p_new.is_absolute(): new_path = p_new.resolve()
        elif str(p_new) == str(p_root) or str(p_new).startswith(str(p_root) + os.sep): new_path = p_new.resolve()
        else: new_path = (p_root / p_new).resolve()
        
        # Security check
        root_path = Path(CURRENT_ROOT_DIR).resolve()
        if str(root_path) not in str(old_path) or str(root_path) not in str(new_path):
             return jsonify({"error": "Access denied"}), 403
             
        if not old_path.exists():
            return jsonify({"error": "Source path not found"}), 404
            
        if new_path.exists():
            return jsonify({"error": "Destination already exists"}), 400

        os.rename(old_path, new_path)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/copy_path', methods=['POST'])
def copy_path_api():
    try:
        data = request.json
        src_rel = data.get('src_path')
        dest_rel = data.get('dest_path')
        
        if not src_rel or not dest_rel: return jsonify({"error": "Paths required"}), 400
        
        p_root = Path(CURRENT_ROOT_DIR)

        # Resolve Src
        p_src = Path(src_rel)
        if p_src.is_absolute(): src_path = p_src.resolve()
        elif str(p_src) == str(p_root) or str(p_src).startswith(str(p_root) + os.sep): src_path = p_src.resolve()
        else: src_path = (p_root / p_src).resolve()

        # Resolve Dest
        p_dest = Path(dest_rel)
        if p_dest.is_absolute(): dest_path = p_dest.resolve()
        elif str(p_dest) == str(p_root) or str(p_dest).startswith(str(p_root) + os.sep): dest_path = p_dest.resolve()
        else: dest_path = (p_root / p_dest).resolve()
        
        # Security check
        root_path = Path(CURRENT_ROOT_DIR).resolve()
        if str(root_path) not in str(src_path) or str(root_path) not in str(dest_path):
             return jsonify({"error": "Access denied"}), 403
             
        if not src_path.exists():
            return jsonify({"error": "Source path not found"}), 404
            
        if dest_path.exists():
            return jsonify({"error": "Destination already exists"}), 400
            
        if src_path.is_dir():
            shutil.copytree(src_path, dest_path)
        else:
            shutil.copy2(src_path, dest_path)
            
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    socketio.start_background_task(background_log_emitter)
    port = config_data.get('port', 8080)
    print(f"Starting UI Server on http://127.0.0.1:{port}")
    socketio.run(app, host='127.0.0.1', port=port, debug=True, use_reloader=False)