import os
import shutil
from datetime import datetime
from .config import BACKUP_DIR, TARGET_FILE

def create_backup(target_file=TARGET_FILE):
    if not os.path.exists(target_file):
        print(f"⚠️ Target file {target_file} not found, skipping backup.")
        return None

    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.basename(target_file)
    backup_path = os.path.join(BACKUP_DIR, f"{filename}_{timestamp}.bak")

    try:
        shutil.copy2(target_file, backup_path)
        print(f"✅ Backup created: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return None
