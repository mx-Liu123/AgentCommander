import os
import time

def get_modified_files(marker_file=".cycle_marker"):
    if not os.path.exists(marker_file):
        return []
    
    marker_mtime = os.path.getmtime(marker_file)
    modified = []
    
    # Walk through current directory
    for root, dirs, files in os.walk("."):
        # Ignore hidden folders and pylib cache
        if "/." in root or "__pycache__" in root: 
            continue
            
        for file in files:
            if file.startswith("."): continue
            filepath = os.path.join(root, file)
            
            # Skip the marker itself and logs
            if filepath.endswith(marker_file): continue
            if filepath.endswith(".log"): continue
            if filepath.endswith(".txt"): continue # Skip log txts usually
            
            try:
                if os.path.getmtime(filepath) > marker_mtime:
                    # Make path relative to current dir for cleaner output
                    rel_path = os.path.relpath(filepath, ".")
                    modified.append(rel_path)
            except OSError:
                pass
                
    return modified
