# Troubleshooting

## No Console Output in UI

If the "Console Output" box in the UI is empty but the agent seems to be running:

1.  **Check Backend Terminal**: Look at the terminal where you started `run_ui.sh`. Do you see logs there?
    *   AgentCommander uses **Dual Logging**: Logs are printed to the terminal *and* sent to the UI.
2.  **Browser Refresh**: If the terminal has logs but the UI doesn't, the WebSocket connection likely dropped. Refresh the page to reconnect.
3.  **Check "Console" Visibility**: Ensure you are on the **Control** tab. The console is located at the bottom of the Control Panel.

## "Run Task" Error: Unexpected keyword argument 'no_exec_list'

This error indicates a mismatch between the running `ui_server.py` and the installed `pylib`.

*   **Fix**: Restart the UI Server (`Ctrl+C` then `bash run_ui.sh`). Python code changes in the library require a restart to be loaded.

## Metric Not Found / Infinite Score

If the agent keeps failing with "Metric is INF":

1.  **Check Evaluator Output**: Use the "White-box Debugging" method to inspect `eval_out.txt` in the experiment directory.
2.  **Check Print Format**: Ensure your `evaluator.py` prints exactly `Best metric: 0.123` (Case sensitive: lowercase 'm' in metric).
3.  **Check Leakage**: If the Anti-Cheating check fails, the score is forced to Infinity. Check if your model is modifying `y_test` in place.

## Windows WSL2 Issues

*   **Accessing UI**: If `localhost:8080` doesn't work, try the WSL IP address (run `wsl hostname -I`).
*   **Permissions**: Ensure your project folder is inside the WSL filesystem (e.g., `~/projects/`), not on the mounted Windows drive (`/mnt/c/`), for better performance and permission handling.
