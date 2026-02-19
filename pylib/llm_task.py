import os
import shutil
import tempfile
import sys
from pathlib import Path
from . import llm_client

class LlmTaskRunner:
    def __init__(self, cwd, permission_mode="open", whitelist=None, blacklist=None, allow_new_files=True, lock_parent=False, no_exec_list=None):
        self.cwd = Path(cwd).resolve()
        self.permission_mode = permission_mode # open, whitelist, blacklist, forbid
        self.whitelist = [self._norm(p) for p in (whitelist or [])]
        self.blacklist = [self._norm(p) for p in (blacklist or [])]
        self.no_exec_list = [self._norm(p) for p in (no_exec_list or [])]
        self.allow_new_files = allow_new_files
        self.backup_dir = None
        self.lock_parent = lock_parent
        self.parent_dir = self.cwd.parent
        self._original_parent_mode = None
        self._exec_restore_map = {}

    def _norm(self, p):
        # Normalize path strings to OS separator
        return str(Path(p))

    def __enter__(self):
        # Create Snapshot
        self.backup_dir = Path(tempfile.mkdtemp(prefix="llm_snapshot_"))
        # Copy everything except heavy folders
        def ignore_patterns(path, names):
            return ['__pycache__', '.git', '.ipynb_checkpoints', 'data', 'wandb', 'node_modules']
        
        try:
            shutil.copytree(self.cwd, self.backup_dir, dirs_exist_ok=True, ignore=ignore_patterns)
        except Exception as e:
            print(f"Warning: Snapshot creation failed partially: {e}")
            
        # Lock ONLY the immediate parent directory
        if self.lock_parent:
            try:
                # Save original mode
                self._original_parent_mode = self.parent_dir.stat().st_mode
                # Remove write permission for user (u-w)
                # 0o200 is Write permission for Owner
                new_mode = self._original_parent_mode & ~0o200
                os.chmod(self.parent_dir, new_mode)
                print(f"🔒 Locked parent dir (Read-Only): {self.parent_dir.name}/", flush=True)
            except Exception as e:
                print(f"⚠️ Failed to lock parent dir: {e}", flush=True)
                self.lock_parent = False # Disable flag so we don't try to unlock later
        
        # Apply No-Exec (Remove execution permissions)
        for rel_p in self.no_exec_list:
            p = self.cwd / rel_p
            if p.exists():
                try:
                    mode = p.stat().st_mode
                    self._exec_restore_map[rel_p] = mode
                    # Remove Execute bits (0o111) -> ~0o111 mask
                    os.chmod(p, mode & ~0o111)
                    # print(f"🚫 Removed exec permission: {rel_p}", flush=True)
                except Exception as e:
                    print(f"⚠️ Failed to chmod -x {rel_p}: {e}", flush=True)
            
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore No-Exec Permissions
        for rel_p, orig_mode in self._exec_restore_map.items():
            try:
                p = self.cwd / rel_p
                if p.exists():
                    os.chmod(p, orig_mode)
            except Exception as e:
                print(f"Warning: Failed to restore mode for {rel_p}: {e}", flush=True)

        # Unlock Parent Directory FIRST
        if self.lock_parent and self._original_parent_mode:
            try:
                os.chmod(self.parent_dir, self._original_parent_mode)
                print(f"🔓 Unlocked parent dir: {self.parent_dir.name}/", flush=True)
            except Exception as e:
                print(f"❌ CRITICAL: Failed to unlock parent dir: {e}. Please manually run 'chmod u+w {self.parent_dir}'", flush=True)

        if self.backup_dir and self.backup_dir.exists():
            shutil.rmtree(self.backup_dir, ignore_errors=True)

    def is_allowed(self, rel_path_str):
        # 1. Open Mode
        if self.permission_mode == "open": return True
        # 2. Forbid Mode
        if self.permission_mode == "forbid": return False
        
        # 3. Whitelist / Blacklist
        # We need to handle directory matching too (e.g. "src" matches "src/main.py")
        is_listed = False
        p = Path(rel_path_str)
        
        check_list = self.whitelist if self.permission_mode == "whitelist" else self.blacklist
        
        for item in check_list:
            # Exact match
            if rel_path_str == item: 
                is_listed = True; break
            # Dir match (item is parent of p)
            try:
                # If item is "src", p is "src/main.py", relative_to works
                p.relative_to(item)
                is_listed = True; break
            except ValueError: pass
            
        if self.permission_mode == "whitelist": return is_listed
        if self.permission_mode == "blacklist": return not is_listed
        return False

    def enforce_permissions(self):
        """Checks changes and reverts unauthorized ones."""
        changes = []
        if not self.backup_dir or not self.backup_dir.exists():
            return changes # No snapshot, can't enforce
            
        # A. Check for Modified/New Files
        for current_file in self.cwd.rglob('*'):
            if current_file.is_dir(): continue
            if current_file.name == 'history.json': continue # Always allow history updates
            
            try:
                rel_path = current_file.relative_to(self.cwd)
            except ValueError: continue
            
            # Skip hidden and __pycache__
            if any(p.startswith('.') for p in rel_path.parts): continue
            if any(p == '__pycache__' for p in rel_path.parts): continue
            
            rel_str = str(rel_path)
            backup_file = self.backup_dir / rel_path
            
            is_new = not backup_file.exists()
            is_modified = False
            
            if not is_new:
                # Check content hash/bytes
                try:
                    with open(current_file, 'rb') as f1, open(backup_file, 'rb') as f2:
                        if f1.read() != f2.read(): is_modified = True
                except: is_modified = True # Error reading means changed/locked
            
            if is_new or is_modified:
                # Check Permission
                allowed = False
                if is_new:
                    # Logic for new files
                    if self.allow_new_files:
                        # Allowed generally, unless blacklisted
                        if self.permission_mode == "blacklist" and not self.is_allowed(rel_str):
                            allowed = False
                        else:
                            # In Whitelist mode, allow_new_files=True means "Allow creating new files NOT in whitelist"?
                            # Usually "allow_new_files" overrides whitelist restriction for NEW files.
                            allowed = True 
                    else:
                        # Blocked generally, unless whitelisted
                        if self.permission_mode == "whitelist" and self.is_allowed(rel_str):
                            allowed = True
                        else:
                            allowed = False
                else:
                    # Modified existing file
                    allowed = self.is_allowed(rel_str)
                
                if not allowed:
                    action = "Deleting" if is_new else "Reverting"
                    print(f"❌ Security: {action} unauthorized change to {rel_str}", flush=True)
                    changes.append(f"{action} {rel_str}")
                    
                    if is_new:
                        try: os.remove(current_file)
                        except: pass
                    else:
                        shutil.copy2(backup_file, current_file)

        # B. Check for Deleted Files
        for backup_file in self.backup_dir.rglob('*'):
            if backup_file.is_dir(): continue
            try:
                rel_path = backup_file.relative_to(self.backup_dir)
            except: continue
            
            if any(p.startswith('.') for p in rel_path.parts): continue
            if any(p == '__pycache__' for p in rel_path.parts): continue
            
            current_file = self.cwd / rel_path
            if not current_file.exists():
                # Deleted
                rel_str = str(rel_path)
                # Check if deletion allowed (same as modification allowed)
                if not self.is_allowed(rel_str):
                    print(f"❌ Security: Restoring deleted file {rel_str}", flush=True)
                    changes.append(f"Restored {rel_str}")
                    try:
                        current_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(backup_file, current_file)
                    except Exception as e:
                        print(f"Warning: Failed to restore {rel_str}: {e}")
                    
        return changes

def run_task(prompt, model, cwd, permission_mode="open", whitelist=None, blacklist=None, allow_new_files=True, timeout=600, session_id=None, yolo=True, lock_parent=False, no_exec_list=None):
    runner = LlmTaskRunner(cwd, permission_mode, whitelist, blacklist, allow_new_files, lock_parent=lock_parent, no_exec_list=no_exec_list)
    
    response = ""
    new_sid = session_id
    
    with runner:
        try:
            response, new_sid = llm_client.call_llm(
                prompt=prompt,
                session_id=session_id,
                model=model,
                cwd=str(runner.cwd),
                timeout=timeout,
                yolo=yolo
            )
        except Exception as e:
            # We assume llm_client logs the error
            raise e
        finally:
            # Enforce permissions even if LLM failed (partial writes)
            runner.enforce_permissions()
            
    return response, new_sid