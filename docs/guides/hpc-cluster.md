# HPC Cluster Deployment (PBS)

Run AgentCommander on a supercomputer head node.

## Templates
Located in `example/pbs_server_example_tasks`.
*   `run_all.pbs`: Job submission script.
*   `watch_job.sh`: Wrapper to submit and poll until completion.

## Usage
Update `eval_cmd` in `config.json` to point to `watch_job.sh` instead of running python directly.
