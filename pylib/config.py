import os

# File Paths
TARGET_FILE = "alpha_digging/run_peak_analysis.py"
HISTORY_FILE = "improvement_history.md"
BACKUP_DIR = "backups"

# Gemini Config
GEMINI_MODEL = "auto" # Set to 'auto' as requested

# Debug Mode
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"
