Installation guide: 
- pip install -r requirements.txt

SLURM jobs tracker:
- Run `python script/slurm_jobs_tracker.py` to track current SLURM jobs for your user, including both queued and running jobs.
- The tracker refreshes every 10 minutes by default and writes the latest snapshot to `exps/slurm_logs/job_tracker.json`.
- Previously tracked jobs that disappear from `squeue` are kept in the snapshot and marked as `DONE`.
- Use `python script/slurm_jobs_tracker.py --once` for a single snapshot.
