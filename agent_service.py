import os
import sys
import glob
import re
import shutil
import time
import threading
import multiprocessing
import subprocess
import shlex
import json
from pathlib import Path
from contextlib import contextmanager
import traceback
import copy
from concurrent.futures import ThreadPoolExecutor

# Ensure we can import from local modules
sys.path.append(os.getcwd())
try:
    from pylib import json_history, llm_client, llm_task
except ImportError:
    # If running from a subdir, try adding parent
    sys.path.append(str(Path(os.getcwd()).parent))
    from pylib import json_history, llm_client, llm_task

# Constants
EXAMPLE_WORKSPACE_DIR = "Branch_example/exp_example"

class AgentLogger:
    def __init__(self, queue):
        self.queue = queue

    def clean(self, text):
        # Remove ANSI color codes
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', str(text))

    def log(self, message):
        clean_msg = self.clean(message)
        if self.queue:
            try: self.queue.put({"type": "log", "data": clean_msg})
            except: pass
        print(message) 

    def error(self, message):
        clean_msg = self.clean(message)
        if self.queue:
            try: self.queue.put({"type": "error", "data": clean_msg})
            except: pass
        print(f"ERROR: {message}", file=sys.stderr)

class GraphExecutor:
    def __init__(self, service, context, graph_def):
        self.service = service
        self.context = context
        self.graph = graph_def
        self.logger = service.logger
        self.stop_event = service.stop_event

    def execute(self):
        # Find start node
        if not self.graph or 'nodes' not in self.graph:
            self.logger.error("Invalid graph definition.")
            return

        start_node = next((n for n in self.graph['nodes'] if n['type'] == 'start'), None)
        if not start_node:
            self.logger.error("No start node found in graph.")
            return

        current_node = start_node
        while current_node:
            if self.stop_event.is_set():
                self.logger.log("🛑 Graph execution stopped.")
                break

            try:
                node_label = current_node.get('label', current_node['type'])
                self.logger.log(f"▶️ Executing: {node_label}")
                
                # Execute Node Logic
                self._process_node(current_node)
                
                if current_node['type'] == 'end':
                    self.logger.log("🏁 Graph End Reached.")
                    break
                
                # Determine Next Node
                current_node = self._find_next_node(current_node)

            except Exception as e:
                self.logger.error(f"Node '{node_label}' failed: {e}")
                self.logger.error(traceback.format_exc())
                break

    def _process_node(self, node):
        ntype = node['type']
        cfg = node.get('config', {})
        
        if ntype == 'python_script':
            code = cfg.get('code', '')
            local_scope = {
                'context': self.context, 
                'service': self.service, 
                'logger': self.logger, 
                'json_history': json_history,
                'os': os, 'sys': sys, 'json': json, 'Path': Path
            }
            exec(code, {}, local_scope)
            
        elif ntype == 'subloop':
            sub_graph = cfg.get('sub_graph')
            if sub_graph:
                self.logger.log(f"  📦 Entering Subloop: {node.get('label')}")
                sub_executor = GraphExecutor(self.service, self.context, sub_graph)
                sub_executor.execute()
                self.logger.log(f"  📦 Subloop Finished.")
        
        elif ntype == 'parallel_loop':
            self._step_parallel(node)
                
        elif ntype == 'condition_code':
            code = cfg.get('code', '')
            local_scope = {'context': self.context, 'result': False}
            exec(code, {}, local_scope)
            self.context['last_condition_result'] = local_scope.get('result', False)
            self.logger.log(f"  ❓ Condition Result: {self.context['last_condition_result']}")
            
        elif ntype == 'llm_generate':
            self._step_llm(node)
            
        elif ntype == 'run_shell':
            self._step_shell(node)
            
        elif ntype == 'write_history':
            self._step_write_history(node)
            
        elif ntype == 'check_improvement':
            self._step_check_improvement(node)
            
        elif ntype == 'lesson':
            self._step_lesson(node)

    def _step_parallel(self, node):
        cfg = node.get('config', {})
        workers = cfg.get('workers', 2)
        sub_graph = cfg.get('sub_graph')
        modifier_code = cfg.get('context_modifier', '')
        
        if not sub_graph: return

        self.logger.log(f"  🔀 Forking into {workers} parallel workers...")

        def run_worker(worker_idx):
            # Deep copy context to ensure isolation
            # service and logger are references (safe for logging/reading, risky for state mutation if not careful)
            try:
                local_ctx = copy.deepcopy(self.context)
            except Exception as e:
                self.logger.error(f"Context copy failed: {e}. Using shallow copy.")
                local_ctx = self.context.copy()
            
            local_ctx['worker_idx'] = worker_idx
            
            # Apply modifier
            if modifier_code:
                try:
                    local_scope = {'context': local_ctx, 'worker_idx': worker_idx, 'service': self.service}
                    exec(modifier_code, {}, local_scope)
                except Exception as e:
                    self.logger.error(f"Worker {worker_idx} modifier failed: {e}")
            
            self.logger.log(f"    ▶️ Worker {worker_idx} started.")
            executor = GraphExecutor(self.service, local_ctx, sub_graph)
            executor.execute()
            self.logger.log(f"    🏁 Worker {worker_idx} finished.")
            return local_ctx

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_worker, i) for i in range(workers)]
            results = [f.result() for f in futures]
            
        self.logger.log(f"  🔀 All parallel workers joined.")

    def _find_next_node(self, current_node):
        edges = self.graph.get('edges', [])
        out_edges = [e for e in edges if e['source'] == current_node['id']]
        
        if not out_edges:
            return None
            
        if len(out_edges) == 1:
            target_id = out_edges[0]['target']
            return next((n for n in self.graph['nodes'] if n['id'] == target_id), None)
        
        # Branching logic
        cond_result = str(self.context.get('last_condition_result', '')).lower()
        
        for e in out_edges:
            edge_label = str(e.get('label', '')).lower()
            if edge_label == cond_result:
                return next((n for n in self.graph['nodes'] if n['id'] == e['target']), None)
        
        # Fallback to edge with no label (default path)
        fallback = next((e for e in out_edges if not e.get('label')), None)
        if fallback:
             return next((n for n in self.graph['nodes'] if n['id'] == fallback['target']), None)
             
        self.logger.error(f"No matching edge found for condition '{cond_result}' from node {current_node['id']}")
        return None

    def recursive_format(self, text, vars_dict, max_depth=10):
        if not text: return ""
        result = text
        for _ in range(max_depth):
            try:
                # Attempt standard formatting
                new_result = result.format(**vars_dict)
                if new_result == result: break
                result = new_result
            except (KeyError, ValueError, IndexError):
                # Fallback: Use Regex for simple {var} substitution
                # This handles JSON braces {} which confuse .format(), and missing keys
                old_result = result
                def replace_match(match):
                    key = match.group(1)
                    if key in vars_dict:
                        return str(vars_dict[key])
                    return match.group(0)
                
                result = re.sub(r"{(\w+)}", replace_match, result)
                if result == old_result: break
            except Exception as e:
                self.logger.error(f"Format error: {e}")
                break
        return result

    def _prepare_context_vars(self):
        global_vars = self.service.config.get("global_vars", {}).copy()
        
        # Load runtime updates
        try:
            runtime_path = self.service.tasks_dir / 'runtime_vars.json'
            if runtime_path.exists():
                with open(runtime_path, 'r') as f:
                    updates = json.load(f)
                    global_vars.update(updates)
        except: pass
        
        return {
            "cwd": os.getcwd(),
            **self.context,
            **global_vars
        }

    def _step_llm(self, node):
        cfg = node.get("config", {})
        user_tmpl = cfg.get("user_template", "")
        context_vars = self._prepare_context_vars()
        prompt = self.recursive_format(user_tmpl, context_vars)
        
        perm_mode = cfg.get("file_permission_mode", "open") # open, whitelist, blacklist, forbid
        target_files_str = cfg.get("target_files", "")
        no_exec_files_str = cfg.get("no_exec_files", "")
        allow_new_files = cfg.get("allow_new_files", False)
        lock_parent = cfg.get("lock_parent", False)
        timeout = cfg.get("timeout", 600)
        model = cfg.get("model", "auto-gemini-3")
        
        # System Prompt Injection (Preserved logic)
        if perm_mode == "whitelist":
            prompt += f"\n\n[SYSTEM: FILE PERMISSION]\nYou are allowed to modify ONLY the following files/folders: [{target_files_str}]."
            if allow_new_files: prompt += "\nYou are ALLOWED to create NEW files."
            else: prompt += "\nYou are NOT allowed to create new files (unless in the whitelist)."
            prompt += "\nViolating these rules will result in your changes being reverted."
        elif perm_mode == "blacklist":
            prompt += f"\n\n[SYSTEM: FILE PERMISSION]\nYou are FORBIDDEN from modifying or creating the following files/folders: [{target_files_str}]."
            if allow_new_files: prompt += "\nYou are ALLOWED to create other new files."
            else: prompt += "\nYou are NOT allowed to create any new files."
        elif perm_mode == "forbid":
            prompt += "\n\n[SYSTEM: FILE PERMISSION]\nYou are operating in STRICT READ-ONLY mode for existing files."
            if allow_new_files: prompt += "\nHowever, you are ALLOWED to create NEW files."
            else: prompt += "\nYou are NOT allowed to create or modify ANY files."
            
        # Session Management
        session_mode = cfg.get("session_mode", "new") # new / inherit
        session_id_input_var = cfg.get("session_id_input", "")
        session_id = None
        
        if session_mode == "inherit" and session_id_input_var:
            session_id = self.context.get(session_id_input_var)
            if not session_id:
                self.logger.log(f"⚠️ Inherit session var '{session_id_input_var}' is empty/missing. Fallback to AUTO_RESUME (-r).")
                session_id = "AUTO_RESUME"

        cwd_path = Path(self.context.get("current_exp_path", self.service.tasks_dir))
        if not cwd_path.exists(): cwd_path = self.service.tasks_dir
        
        self.logger.log(f"      🗣️  LLM Call ({model}, {session_mode}) in {cwd_path.name}...")
        
        # Delegate to llm_task
        try:
            target_files = [f.strip() for f in target_files_str.split(',') if f.strip()]
            no_exec_list = [f.strip() for f in no_exec_files_str.split(',') if f.strip()]
            
            response, new_session_id = llm_task.run_task(
                prompt=prompt,
                model=model,
                cwd=str(cwd_path),
                permission_mode=perm_mode,
                whitelist=target_files if perm_mode == "whitelist" else None,
                blacklist=target_files if perm_mode == "blacklist" else None,
                allow_new_files=allow_new_files,
                timeout=timeout,
                session_id=session_id,
                lock_parent=lock_parent,
                no_exec_list=no_exec_list
            )
            
            self.context["last_response"] = response
            self.context["last_prompt"] = prompt
            
            # Save Session ID
            session_id_output_var = cfg.get("session_id_output", "last_session_id")
            if session_id_output_var:
                self.context[session_id_output_var] = new_session_id
                
            # Parse response var
            response_var = cfg.get("response_output", "last_response")
            self.context[response_var] = response
            
        except Exception as e:
            self.logger.error(f"LLM Step Failed: {e}")

    def _step_shell(self, node):
        cfg = node.get("config", {})
        cmd_tmpl = cfg.get("command", "")
        timeout = cfg.get("timeout", 600)
        output_vars = cfg.get("output_vars", [])
        if isinstance(output_vars, str):
            output_vars = [x.strip() for x in output_vars.split(',') if x.strip()]
        
        context_vars = self._prepare_context_vars()
        cmd = self.recursive_format(cmd_tmpl, context_vars)
        
        cwd_path = Path(self.context.get("current_exp_path", self.service.tasks_dir))
        if not cwd_path.exists(): cwd_path = self.service.tasks_dir
        
        self.logger.log(f"      💻 Running in {cwd_path}: {cmd}")
        try:
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=str(cwd_path))
            out = res.stdout.strip()
            if res.returncode != 0: self.logger.error(f"Stderr: {res.stderr}")
            
            if output_vars and len(output_vars) > 0:
                self.context[output_vars[0]] = out
        except Exception as e: self.logger.error(f"Shell Error: {e}")

    def _step_write_history(self, node):
        cfg = node.get("config", {})
        key = cfg.get("key", "log")
        mode = cfg.get("mode", "overwrite")
        val_type = cfg.get("value_type", "string")
        val_tmpl = cfg.get("value_template", "")
        
        context_vars = self._prepare_context_vars()
        val = self.recursive_format(val_tmpl, context_vars)
        
        if val_type == "json":
            try:
                 if "```" in val:
                     match = re.search(r"```(?:json)?(.*?)```", val, re.DOTALL)
                     if match: val = match.group(1)
                 val = json.loads(val)
            except: pass
        elif val_type == "boolean":
            val = str(val).lower() == "true"
            
        path_str = self.context.get("current_exp_path")
        if not path_str:
            self.logger.error("No current_exp_path in context, cannot write history.")
            return

        path = Path(path_str)
        if path.exists():
            hist = json_history.load_history(path)
            if mode == "append":
                if key not in hist or not isinstance(hist[key], list): hist[key] = []
                hist[key].append(val)
            elif mode == "update" and isinstance(val, dict):
                if key not in hist: hist[key] = {}
                hist[key].update(val)
            else:
                hist[key] = val
            json_history.save_history(path, hist)
            self.logger.log(f"      💾 Wrote History: {key}")

    def _step_check_improvement(self, node):
        cfg = node.get("config", {})
        metric_key = cfg.get("metric_key", "score")
        direction = cfg.get("direction", "max") # max or min
        
        current = self.context.get("current_metric")
        
        # If not in context, try parsing context['analysis']
        if current is None and 'analysis' in self.context:
             try:
                 # heuristic
                 match = re.search(f'"{metric_key}":\\s*([\\d\\.]+)', str(self.context['analysis']))
                 if match: current = float(match.group(1))
             except: pass
             
        parent = self.context.get("parent_metric")
        is_improved = False
        
        if current is not None:
            try:
                current_val = float(current)
                if parent is None:
                    is_improved = True
                else:
                    parent_val = float(parent)
                    if direction == "min":
                        is_improved = current_val < parent_val
                    else:
                        is_improved = current_val > parent_val
            except: 
                pass # Conversion failed
                
        self.context["is_improved"] = is_improved
        self.logger.log(f"      📊 Check Metric ({direction}): {current} vs {parent} -> Improved: {is_improved}")

    def _step_lesson(self, node):
        cfg = node.get("config", {})
        count = int(cfg.get("lookback_count", 5))
        offset = int(cfg.get("offset", 0)) # Skip latest N experiments
        scope = cfg.get("scope", "Same Branch/Layer")
        filter_mode = cfg.get("filter", "Failures Only")
        output_var = cfg.get("output_var", "lessons")
        
        self.logger.log(f"      🎓 Generating Lessons ({scope}, {filter_mode}, offset={offset})...")
        
        lessons = []
        try:
            # Determine search path based on scope
            search_root = self.service.tasks_dir
            if "Same Branch" in scope:
                b_idx = self.context.get('branch_idx', 1)
                search_root = search_root / f"Branch{b_idx}"
            
            # Collect all history.json
            candidates = []
            for h_path in search_root.glob("**/history.json"):
                try:
                    with open(h_path, 'r') as f:
                        h = json.load(f)
                        # Filter logic
                        improved = h.get("if_improved", False)
                        if filter_mode == "Failures Only" and improved: continue
                        if filter_mode == "Successes Only" and not improved: continue
                        
                        # Get timestamp or folder name for sorting. 
                        # Using folder modification time as proxy or just folder name if structured.
                        mtime = h_path.stat().st_mtime
                        candidates.append((mtime, h_path.parent.name, h))
                except: pass
            
            # Sort by time desc
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Apply Offset and Count
            candidates = candidates[offset : offset + count]
            
            for _, name, h in candidates:
                hyp = h.get('hypothesis', 'N/A')
                des = h.get('exp_design', 'N/A')
                res = h.get('result_analysis', 'N/A')
                
                lesson_entry = f"[{name}]\n[Hypothesis] {hyp}\n[Exp Design] {des}\n[Result Analysis] {res}"
                lessons.append(lesson_entry)
                
        except Exception as e:
            self.logger.error(f"Lesson gen failed: {e}")
            lessons.append("Error generating lessons.")
            
        result_text = "\n\n".join(lessons)
        self.context[output_var] = result_text
        self.logger.log(f"      🎓 Found {len(lessons)} lessons.")



class AgentService:
    def __init__(self, tasks_dir, config, log_queue=None, stop_event=None):
        self.tasks_dir = Path(tasks_dir).resolve()
        self.config = config
        self.process = None
        
        self.stop_event = stop_event if stop_event else multiprocessing.Event()
        
        self.log_queue = log_queue
        self.logger = AgentLogger(self.log_queue)
        self.workflow_graph = config.get("workflow", {})
        self.root_file_backup = {}

    def backup_root_files(self):
        self.root_file_backup = {}
        try:
            # Backup root files
            for item in os.listdir(self.tasks_dir):
                path = self.tasks_dir / item
                if path.is_file():
                    with open(path, 'rb') as f:
                        self.root_file_backup[item] = f.read()
                        
            # Backup Branch_example recursively
            example_dir = self.tasks_dir / "Branch_example"
            if example_dir.exists():
                backup_dir = self.tasks_dir / ".backup_branch_example"
                if backup_dir.exists(): shutil.rmtree(backup_dir)
                shutil.copytree(example_dir, backup_dir)
                
        except Exception as e: self.logger.error(f"Failed to backup root files: {e}")

    def restore_root_files(self):
        # Restore root files
        for filename, content in self.root_file_backup.items():
            path = self.tasks_dir / filename
            try:
                if not path.exists() or path.read_bytes() != content:
                    self.logger.log(f"🛡️ Security: Restoring '{filename}'...")
                    with open(path, 'wb') as f:
                        f.write(content)
            except Exception as e: self.logger.error(f"Failed to restore root file {filename}: {e}")
            
        # Restore Branch_example
        example_dir = self.tasks_dir / "Branch_example"
        backup_dir = self.tasks_dir / ".backup_branch_example"
        if backup_dir.exists():
            # Check if restore needed (simple check: if original deleted or modified? 
            # Deep compare is slow. We'll just restore if it exists to be safe, or checksum?
            # For simplicity and safety, we overwrite Branch_example with backup if it differs.
            # But overwriting every time is slow.
            # Let's just restore if we detect LLM touched it? 
            # We assume LLM strictly forbidden. If we are paranoid, we restore always.
            # Let's restore always for "Branch_example".
            if example_dir.exists(): shutil.rmtree(example_dir)
            shutil.copytree(backup_dir, example_dir)
            # self.logger.log(f"🛡️ Security: Enforced 'Branch_example' integrity.") 
            # Log is too spammy if always restoring.

    # ... (validate_environment, get_max_branch_idx, setup_branch, scan_experiments, generate_next_node, setup_workspace, _generate_lessons - UNCHANGED)

    # ... (run method - UNCHANGED)

# ... (Helper for multiprocessing - UNCHANGED)



    def validate_environment(self):
        if not self.tasks_dir.exists():
            raise FileNotFoundError(f"Directory '{self.tasks_dir}' does not exist.")
        self.logger.log(f"✅ Environment verified at: {self.tasks_dir}")

    def get_max_branch_idx(self):
        branches = glob.glob(str(self.tasks_dir / "Branch*"))
        max_idx = 0
        for b in branches:
            name = os.path.basename(b)
            match = re.match(r"Branch(\d+)", name)
            if match:
                idx = int(match.group(1))
                if idx > max_idx: max_idx = idx
        return max_idx

    def setup_branch(self):
        max_idx = self.get_max_branch_idx()
        target_idx = 0
        mode = self.config.get("mode", "new")
        resume_branch_id = self.config.get("resume_branch_id", None)
        
        # New Params
        branch_name = self.config.get("branch_name", "")
        branch_hint = self.config.get("branch_hint", "")
        parent_exp = self.config.get("parent_exp", "")

        if mode == "new":
            target_idx = max_idx + 1
            branch_path = self.tasks_dir / f"Branch{target_idx}"
            branch_path.mkdir(parents=True, exist_ok=True)
            self.logger.log(f"✨ Creating new Branch{target_idx}...")
            
            # Store metadata for first exp
            self.new_branch_meta = {
                "name": branch_name,
                "hint": branch_hint,
                "parent": parent_exp
            }
        else:
            if resume_branch_id: target_idx = int(resume_branch_id)
            else: target_idx = max_idx if max_idx > 0 else 1
            
            branch_path = self.tasks_dir / f"Branch{target_idx}"
            if not branch_path.exists():
                branch_path.mkdir(parents=True, exist_ok=True)
            self.logger.log(f"🔄 Resuming Branch{target_idx}...")
            self.new_branch_meta = {}

        return branch_path, target_idx

    # --- Helper methods exposed to Python Script Nodes via 'service' ---

    def scan_experiments(self, branch_path):
        branch_path = Path(branch_path)
        exp_folders = glob.glob(str(branch_path / "exp*"))
        valid_experiments = []
        regex = re.compile(r"exp(\d+)\.(\d+)\.(\d+)$") 
        
        for folder in exp_folders:
            name = os.path.basename(folder)
            match = regex.match(name)
            if match:
                folder_path = Path(folder)
                status = "unknown"
                is_improved = False
                try:
                    h = json_history.load_history(folder_path)
                    if h.get("if_improved"): is_improved = True
                except: pass
                
                valid_experiments.append({
                    "path": folder_path,
                    "name": name,
                    "b": int(match.group(1)),
                    "l": int(match.group(2)),
                    "s": int(match.group(3)),
                    "is_improved": is_improved
                })
        
        valid_experiments.sort(key=lambda x: (x['l'], x['s']))
        
        last_improved = None
        last_attempt = None
        if valid_experiments:
            last_attempt = valid_experiments[-1]
            for exp in reversed(valid_experiments):
                if exp['is_improved']:
                    last_improved = exp
                    break
        return valid_experiments, last_improved, last_attempt

    def generate_next_node(self, branch_idx, last_improved, last_attempt):
        # Check if we have new branch metadata (first node of new branch)
        if hasattr(self, 'new_branch_meta') and self.new_branch_meta:
             parent = self.new_branch_meta.get('parent')
             if parent:
                 # Resolve parent path relative to tasks_dir
                 parent_path = self.tasks_dir / parent
                 if parent_path.exists():
                     return 1, 1, parent_path
             
        if not last_attempt: return 1, 1, None 
        last_l = last_attempt['l']
        last_s = last_attempt['s']
        if last_attempt['is_improved']: return last_l + 1, 1, last_attempt['path']
        else: return last_l, last_s + 1, last_improved['path'] if last_improved else None

    def setup_workspace(self, branch_path, next_l, next_s, parent_node_path, branch_idx):
        branch_path = Path(branch_path)
        new_folder_name = f"exp{branch_idx}.{next_l}.{next_s}"
        new_folder_path = branch_path / new_folder_name
        
        if new_folder_path.exists(): shutil.rmtree(new_folder_path)

        source_path = Path(parent_node_path) if parent_node_path else (self.tasks_dir / EXAMPLE_WORKSPACE_DIR)
        if not source_path.exists():
            source_path = self.tasks_dir / EXAMPLE_WORKSPACE_DIR

        if not source_path.exists():
             self.logger.log(f"⚠️ Source template not found. Using empty dir.")
             new_folder_path.mkdir(parents=True, exist_ok=True)
        else:
             shutil.copytree(source_path, new_folder_path)
             
        json_history.init_new_history(new_folder_path, source_path)
        
        # Inject Hint/Name if first node
        if hasattr(self, 'new_branch_meta') and self.new_branch_meta and next_l==1 and next_s==1:
            h = json_history.load_history(new_folder_path)
            meta = self.new_branch_meta
            if meta.get('hint'): h['hint'] = meta['hint']
            if meta.get('name'): h['branch_name'] = meta['name']
            if meta.get('parent'): h['parent_exp'] = meta['parent']
            json_history.save_history(new_folder_path, h)
            self.new_branch_meta = {} # Clear after use
            
        return new_folder_path

    def _generate_lessons(self, branch_path, B, L, current_S):
        branch_path = Path(branch_path)
        lessons = []
        try:
            pattern = str(branch_path / f"exp{B}.{L}.*")
            candidates = []
            for p in glob.glob(pattern):
                match = re.match(r"exp(\d+)\.(\d+)\.(\d+)$", os.path.basename(p))
                if match:
                    b, l, s = map(int, match.groups())
                    if b == B and l == L and s < current_S:
                        candidates.append((s, p))
            
            candidates.sort(key=lambda x: x[0], reverse=True)
            for _, exp_dir in candidates:
                h_path = os.path.join(exp_dir, "history.json")
                if os.path.exists(h_path):
                    try:
                        with open(h_path, 'r') as f:
                            h = json.load(f)
                            if not h.get("if_improved", False):
                                summary = f"Res: {h.get('result_analysis','N/A')}"
                                lessons.append(f"[{os.path.basename(exp_dir)}] {summary}")
                    except: pass
                if len(lessons) >= 5: break
        except Exception as e: self.logger.error(f"Error generating lessons: {e}")
            
        if lessons: return f"\n[LESSONS]\n" + "\n".join(lessons)
        return ""

    def run(self):
        try:
            self.validate_environment()
            self.backup_root_files()
            
            # Setup Branch & Context
            branch_path, branch_idx = self.setup_branch()
            context = {
                "branch_idx": branch_idx,
                "n_cycles": self.config.get("n_cycles", 1),
                "cycle": 0
            }
            
            self.logger.log("🧠 Initializing Graph Executor...")
            executor = GraphExecutor(self, context, self.workflow_graph)
            executor.execute()
            
            self.logger.log("✅ Agent Run Completed.")
            
        except Exception as e:
            self.logger.error(f"Critical Agent Error: {traceback.format_exc()}")

# Helper for multiprocessing
def agent_process_wrapper(tasks_dir, config, queue, stop_event=None):
    # Save original streams to keep terminal output alive
    orig_out = sys.__stdout__ or sys.stdout
    orig_err = sys.__stderr__ or sys.stderr

    # Setup logging via queue
    logger = None
    if queue:
        class QueueLogger:
            def log(self, msg): 
                queue.put({"type": "log", "data": msg})
                # Echo to terminal
                try: orig_out.write(f"{msg}\n"); orig_out.flush()
                except: pass

            def error(self, msg): 
                queue.put({"type": "error", "data": msg})
                try: orig_err.write(f"ERROR: {msg}\n"); orig_err.flush()
                except: pass

            def write(self, msg): 
                # Always echo to terminal to preserve formatting (newlines)
                try: orig_out.write(msg); orig_out.flush()
                except: pass

                # Only send substantive logs to Queue
                if msg.strip(): 
                    queue.put({"type": "log", "data": msg.strip()})
            def flush(self): 
                try: orig_out.flush()
                except: pass
            
        logger = QueueLogger()
        # Redirect stdout/stderr to capture all library prints
        sys.stdout = logger
        sys.stderr = logger
    else:
        # Fallback to local print if no queue
        class PrintLogger:
            def log(self, msg): print(f"[Agent] {msg}")
            def error(self, msg): print(f"[Agent Error] {msg}")
        logger = PrintLogger()

    service = AgentService(tasks_dir, config, logger, stop_event)
    service.run()