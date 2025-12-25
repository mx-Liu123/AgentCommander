import os
import json
import argparse
import glob
import sys
from pathlib import Path
from flask import Flask, jsonify, request, render_template_string

# Configuration
ROOT_DIR = "."
TEMPLATE_FILE = "viz_template.html"

app = Flask(__name__)

def get_experiment_status(folder_name):
    if "_no_improved" in folder_name or "(no improved)" in folder_name:
        return "no_improved"
    elif "_improved" in folder_name or "(improved)" in folder_name:
        return "improved"
    elif "_working" in folder_name or "(working)" in folder_name:
        return "working"
    return "unknown"

def scan_directory_structure(root_dir):
    root = Path(root_dir)
    if not root.exists():
        return {"error": "Directory not found"}

    tree = {
        "root": str(root.resolve()),
        "branches": []
    }

    # Scan Branches
    branch_dirs = sorted(root.glob("Branch*"), key=lambda p: p.name)
    
    for b_dir in branch_dirs:
        if not b_dir.is_dir(): continue
        
        branch_data = {
            "name": b_dir.name,
            "experiments": []
        }
        
        # Scan Experiments inside Branch
        exp_dirs = sorted(b_dir.glob("exp*"), key=lambda p: p.name)
        
        for exp_dir in exp_dirs:
            if not exp_dir.is_dir(): continue
            
            # Read history.json for Goal and Metric
            history_path = exp_dir / "history.json"
            goal = "No goal found."
            metric = None
            
            if history_path.exists():
                try:
                    with open(history_path, 'r') as f:
                        h_data = json.load(f)
                        goal = h_data.get("goal", "No goal specified.")
                        metrics_data = h_data.get("metrics", {})
                        if isinstance(metrics_data, dict):
                            metric = metrics_data.get("score", "N/A")
                        else:
                            metric = metrics_data
                except Exception:
                    goal = "Error reading history.json"
            
            # Optimization: DO NOT list files here. Lazy load them.

            exp_data = {
                "name": exp_dir.name,
                "status": get_experiment_status(exp_dir.name),
                "goal": goal,
                "metric": metric,
                "path": str(exp_dir.relative_to(root)), # Relative path for API safety
            }
            branch_data["experiments"].append(exp_data)
            
        tree["branches"].append(branch_data)

    return tree

@app.route('/')
def index():
    try:
        # Determine template path relative to script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tpl_path = os.path.join(script_dir, TEMPLATE_FILE)
        
        with open(tpl_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return render_template_string(content)
    except Exception as e:
        return f"Error loading template '{TEMPLATE_FILE}': {e}", 500

@app.route('/api/tree')
def api_tree():
    data = scan_directory_structure(ROOT_DIR)
    return jsonify(data)

@app.route('/api/list_files')
def api_list_files():
    rel_path = request.args.get('path')
    if not rel_path:
        return jsonify({"error": "Missing path param"}), 400
        
    full_path = Path(ROOT_DIR) / rel_path
    
    # Security check
    try:
        full_path = full_path.resolve()
        root_abs = Path(ROOT_DIR).resolve()
        if root_abs not in full_path.parents and root_abs != full_path.parent:
             return jsonify({"error": "Access denied"}), 403
    except Exception as e:
         return jsonify({"error": str(e)}), 400
         
    if not full_path.exists() or not full_path.is_dir():
         return jsonify({"error": "Directory not found"}), 404
         
    files = [f.name for f in full_path.iterdir() if f.is_file()]
    files.sort()
    # Put history.json first
    if "history.json" in files:
        files.insert(0, files.pop(files.index("history.json")))
        
    return jsonify({"files": files})

@app.route('/api/file')
def api_file():
    rel_path = request.args.get('path')
    file_name = request.args.get('file')
    
    if not rel_path or not file_name:
        return jsonify({"error": "Missing path or file params"}), 400
        
    full_path = Path(ROOT_DIR) / rel_path / file_name
    
    # Security check
    try:
        full_path = full_path.resolve()
        root_abs = Path(ROOT_DIR).resolve()
        if root_abs not in full_path.parents and root_abs != full_path.parent:
             return jsonify({"error": "Access denied"}), 403
    except Exception as e:
         return jsonify({"error": str(e)}), 400

    if not full_path.exists():
        return jsonify({"error": "File not found"}), 404
        
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({"content": content})
    except UnicodeDecodeError:
        return jsonify({"content": "[Binary or Non-UTF8 content]"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Agent Progress")
    parser.add_argument('dir', type=str, help="Directory containing Branch/exp folders")
    parser.add_argument('--port', type=int, default=5000, help="Port to run server on")
    args = parser.parse_args()

    ROOT_DIR = args.dir
    if not os.path.exists(ROOT_DIR):
        print(f"Error: Directory '{ROOT_DIR}' does not exist.")
        sys.exit(1)
        
    print(f"Starting server scanning: {ROOT_DIR}")
    print(f"Open http://localhost:{args.port} in your browser")
    
    app.run(host='0.0.0.0', port=args.port, debug=False)