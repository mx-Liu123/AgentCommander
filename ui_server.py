import os
import sys
import json
import glob
import multiprocessing
import queue
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
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# Global State
agent_process = None
log_queue = multiprocessing.Queue()
stop_event = multiprocessing.Event()
recent_logs = []

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
CACHE_DIR = Path("pylib/.cache")
CACHE_FILE = CACHE_DIR / "current_graph.json"
DEFAULT_TEMPLATE = Path("pylib/default_graph.json")
CONFIG_FILE = Path("config.json")

# Load Config
def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
    
    # Create default config if missing
    default_config = {
        "root_dir": os.path.join(os.getcwd(), "example/diabetes_sklearn"),
        "n_cycles": 5,
        "port": 8080,
        "global_vars": {
            "DEFAULT_SYS": "You are an expert AI Data Scientist. Your goal is to minimize Mean Squared Error (MSE) on the Diabetes dataset. Modify `strategy.py` to improve the `Strategy` class (sklearn model). Metric: MSE (Lower is better).",
        "venv": "python"
        },
        "llm_changeable_vars": ["hypothesis", "exp_design", "result_analysis", "hint"]
    }
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"Created default config: {CONFIG_FILE}")
    except Exception as e:
        print(f"Error creating default config: {e}")
        
    return default_config

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
    conf = load_config()
    
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
    return jsonify({
        "root_dir": CURRENT_ROOT_DIR,
        "branches": get_branches(CURRENT_ROOT_DIR),
        "global_vars": conf.get('global_vars', {}),
        "llm_changeable_vars": conf.get('llm_changeable_vars', []),
        "n_cycles": conf.get('n_cycles', 1),
        "port": conf.get('port', 8080)
    })

@app.route('/api/save_config', methods=['POST'])
def save_config_api():
    global CURRENT_ROOT_DIR # Moved to top
    try:
        new_conf = request.json
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
            
        target_path = Path(CURRENT_ROOT_DIR) / path_str
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
        
        target_path = Path(CURRENT_ROOT_DIR) / path_str
        if not target_path.exists():
            return jsonify({"error": "File not found"}), 404
            
        with open(target_path, 'r') as f:
            new_conf = json.load(f)
            
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
    if os.path.exists(new_root):
        CURRENT_ROOT_DIR = new_root
        # Persist to config.json
        try:
            current_conf = load_config()
            current_conf['root_dir'] = CURRENT_ROOT_DIR
            with open(CONFIG_FILE, 'w') as f:
                json.dump(current_conf, f, indent=2)
        except: pass
        
        return jsonify({"status": "ok", "branches": get_branches(new_root)})
    return jsonify({"error": "Path does not exist"}), 400

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
                
                # Merge extra fields if provided
                if 'global_vars' in data: save_data['global_vars'] = data['global_vars']
                if 'llm_changeable_vars' in data: save_data['llm_changeable_vars'] = data['llm_changeable_vars']
                
                with open(CACHE_FILE, 'w') as f: json.dump(save_data, f, indent=2)
                
            elif action == 'save_as':
                # UI Data -> Target Path AND Cache
                path_str = data.get('path')
                if not path_str: return jsonify({"error": "Path required"}), 400
                
                target_path = Path(CURRENT_ROOT_DIR) / path_str
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Construct data
                save_data = data.get('content', {})
                if 'global_vars' in data: save_data['global_vars'] = data['global_vars']
                if 'llm_changeable_vars' in data: save_data['llm_changeable_vars'] = data['llm_changeable_vars']
                
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
    
    # Ensure cache exists so AI finds the file
    ensure_cache()
    
    # Context instruction
    context_instruction = (
        "You are the Workflow Editor Assistant. "
        "You are working in a directory containing the active workflow file 'current_graph.json'. "
        "User requests will be about modifying this workflow. "
        "1. Always read 'current_graph.json' first to understand the structure (nodes, edges, IDs). "
        "2. Directly modify 'current_graph.json' using 'replace' or 'write_file'. "
        "3. Respond concisely. "
        f"\nUser Request: {user_msg}"
    )
    
    # The 'cwd' is key here.
    cwd_str = str(CACHE_DIR.resolve())
    
    try:
        response, new_sid = llm_client.call_gemini(
            prompt=context_instruction,
            session_id=session_id,
            model=model,
            cwd=cwd_str,
            timeout=600
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
    rel_path = request.args.get('path', '')
    target_path = Path(CURRENT_ROOT_DIR) / rel_path
    
    # Security check (basic)
    try:
        target_path = target_path.resolve()
        root_path = Path(CURRENT_ROOT_DIR).resolve()
        if root_path not in target_path.parents and target_path != root_path:
             return jsonify({"error": "Access denied"}), 403
    except:
        return jsonify({"error": "Invalid path"}), 400

    if rel_path == '':
        # Root request: Return tree structure wrapper
        return jsonify({
            "name": root_path.name,
            "type": "folder",
            "path": str(root_path),
            "rel_path": "",
            "children": scan_directory(target_path)
        })
    else:
        # Subfolder request: Return list of children directly
        return jsonify(scan_directory(target_path))

@app.route('/api/file_content', methods=['GET'])
def get_file_content():
    # Expects relative path from CURRENT_ROOT_DIR
    rel_path = request.args.get('path')
    if not rel_path: return jsonify({"error": "No path"}), 400
    
    try:
        full_path = (Path(CURRENT_ROOT_DIR) / rel_path).resolve()
        print(f"DEBUG: Request content for {rel_path}")
        print(f"DEBUG: Full path resolved to {full_path}")
        
        # Simple security check
        if str(Path(CURRENT_ROOT_DIR).resolve()) not in str(full_path):
             pass # In a real app, enforce jail. Here we trust user.
        
        if full_path.exists() and full_path.is_file():
             # Check extension
             suffix = full_path.suffix.lower()
             if suffix in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']:
                 import base64
                 with open(full_path, 'rb') as f:
                     encoded = base64.b64encode(f.read()).decode('utf-8')
                     return jsonify({"content": encoded})
             else:
                 with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                     return jsonify({"content": f.read()})
        return jsonify({"error": "File not found"}), 404
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
        
        full_path = (Path(CURRENT_ROOT_DIR) / rel_path).resolve()
        
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
    if agent_process and agent_process.is_alive():
        return jsonify({"error": "Agent already running"}), 400
        
    config = request.json
    
    # Resolve workflow if it's a path string
    if isinstance(config.get('workflow'), str):
        wf_path = (Path(CURRENT_ROOT_DIR) / config['workflow']).resolve()
        if not wf_path.exists():
            return jsonify({"error": f"Workflow file not found: {config['workflow']}"}), 400
        try:
            with open(wf_path, 'r', encoding='utf-8') as f:
                config['workflow'] = json.load(f)
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
        
        full_path = (Path(CURRENT_ROOT_DIR) / rel_path).resolve()
        
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
        
        old_path = (Path(CURRENT_ROOT_DIR) / old_rel).resolve()
        new_path = (Path(CURRENT_ROOT_DIR) / new_rel).resolve()
        
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
        
        src_path = (Path(CURRENT_ROOT_DIR) / src_rel).resolve()
        dest_path = (Path(CURRENT_ROOT_DIR) / dest_rel).resolve()
        
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