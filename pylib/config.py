import os

# File Paths
TARGET_FILE = "alpha_digging/run_peak_analysis.py"
HISTORY_FILE = "improvement_history.md"
BACKUP_DIR = "backups"

# Gemini Config
GEMINI_MODEL = "auto-gemini-3" # Set to 'auto-gemini-3' as requested

# Debug Mode
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"
