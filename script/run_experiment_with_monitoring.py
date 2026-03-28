#!/usr/bin/env python3
"""
Run training experiment with comprehensive GPU monitoring and timing.

This script wraps the training process and tracks:
- GPU utilization per device
- GPU memory usage
- Wall clock time for each phase
- Detailed timing breakdown

Usage:
    python script/run_experiment_with_monitoring.py training/config/main.yaml [--overrides]

"""
import subprocess
import threading
import time
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Try to import pynvml for NVIDIA, fall back to rocm-smi for AMD
try:
    import pynvml
    GPU_BACKEND = "nvidia"
except ImportError:
    GPU_BACKEND = "amd"


class GPUMonitor:
    """Monitor GPU usage in a background thread."""
    
    def __init__(self, interval=1.0, num_gpus=8):
        self.interval = interval
        self.num_gpus = num_gpus
        self.running = False
        self.thread = None
        self.data = defaultdict(list)
        self.timestamps = []
        
    def _get_gpu_stats_amd(self):
        """Get GPU stats using rocm-smi for AMD GPUs."""
        stats = []
        try:
            # Get utilization
            result = subprocess.run(
                ["rocm-smi", "--showuse", "--json"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                use_data = json.loads(result.stdout)
            else:
                use_data = {}
            
            # Get memory
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                mem_data = json.loads(result.stdout)
            else:
                mem_data = {}
            
            for i in range(self.num_gpus):
                card_key = f"card{i}"
                gpu_util = 0
                mem_used = 0
                mem_total = 1
                
                # Parse utilization
                if card_key in use_data:
                    gpu_use = use_data[card_key].get("GPU use (%)", "0")
                    try:
                        gpu_util = float(str(gpu_use).replace("%", ""))
                    except:
                        gpu_util = 0
                
                # Parse memory
                if card_key in mem_data:
                    mem_info = mem_data[card_key]
                    try:
                        mem_used = int(mem_info.get("VRAM Total Used Memory (B)", 0))
                        mem_total = int(mem_info.get("VRAM Total Memory (B)", 1))
                    except:
                        pass
                
                stats.append({
                    "gpu_id": i,
                    "utilization": gpu_util,
                    "memory_used_mb": mem_used / (1024 * 1024),
                    "memory_total_mb": mem_total / (1024 * 1024),
                    "memory_percent": (mem_used / mem_total * 100) if mem_total > 0 else 0
                })
                
        except Exception as e:
            print(f"Warning: Failed to get GPU stats: {e}")
            for i in range(self.num_gpus):
                stats.append({
                    "gpu_id": i,
                    "utilization": 0,
                    "memory_used_mb": 0,
                    "memory_total_mb": 0,
                    "memory_percent": 0
                })
        
        return stats
    
    def _get_gpu_stats_nvidia(self):
        """Get GPU stats using pynvml for NVIDIA GPUs."""
        stats = []
        try:
            pynvml.nvmlInit()
            for i in range(self.num_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                stats.append({
                    "gpu_id": i,
                    "utilization": util.gpu,
                    "memory_used_mb": mem.used / (1024 * 1024),
                    "memory_total_mb": mem.total / (1024 * 1024),
                    "memory_percent": (mem.used / mem.total * 100)
                })
            pynvml.nvmlShutdown()
        except Exception as e:
            print(f"Warning: Failed to get GPU stats: {e}")
            for i in range(self.num_gpus):
                stats.append({
                    "gpu_id": i,
                    "utilization": 0,
                    "memory_used_mb": 0,
                    "memory_total_mb": 0,
                    "memory_percent": 0
                })
        return stats
    
    def _get_gpu_stats(self):
        if GPU_BACKEND == "nvidia":
            return self._get_gpu_stats_nvidia()
        else:
            return self._get_gpu_stats_amd()
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.running:
            timestamp = time.time()
            stats = self._get_gpu_stats()
            
            self.timestamps.append(timestamp)
            for stat in stats:
                gpu_id = stat["gpu_id"]
                self.data[f"gpu{gpu_id}_util"].append(stat["utilization"])
                self.data[f"gpu{gpu_id}_mem_mb"].append(stat["memory_used_mb"])
                self.data[f"gpu{gpu_id}_mem_pct"].append(stat["memory_percent"])
            
            time.sleep(self.interval)
    
    def start(self):
        """Start monitoring."""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"GPU monitoring started (backend: {GPU_BACKEND}, interval: {self.interval}s)")
    
    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print("GPU monitoring stopped")
    
    def get_summary(self):
        """Get summary statistics."""
        summary = {}
        
        for key, values in self.data.items():
            if values:
                summary[key] = {
                    "mean": sum(values) / len(values),
                    "max": max(values),
                    "min": min(values),
                    "samples": len(values)
                }
        
        # Calculate aggregate stats
        total_samples = len(self.timestamps)
        if total_samples > 0:
            # Average utilization across all GPUs
            all_utils = []
            for i in range(self.num_gpus):
                all_utils.extend(self.data.get(f"gpu{i}_util", []))
            
            if all_utils:
                summary["aggregate"] = {
                    "avg_utilization": sum(all_utils) / len(all_utils),
                    "total_samples": total_samples,
                    "monitoring_duration_sec": self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0
                }
        
        return summary
    
    def save_detailed_log(self, filepath):
        """Save detailed time-series data to file."""
        with open(filepath, "w") as f:
            f.write("timestamp,")
            f.write(",".join([f"gpu{i}_util,gpu{i}_mem_mb,gpu{i}_mem_pct" for i in range(self.num_gpus)]))
            f.write("\n")
            
            for idx, ts in enumerate(self.timestamps):
                row = [str(ts)]
                for i in range(self.num_gpus):
                    util = self.data[f"gpu{i}_util"][idx] if idx < len(self.data[f"gpu{i}_util"]) else 0
                    mem_mb = self.data[f"gpu{i}_mem_mb"][idx] if idx < len(self.data[f"gpu{i}_mem_mb"]) else 0
                    mem_pct = self.data[f"gpu{i}_mem_pct"][idx] if idx < len(self.data[f"gpu{i}_mem_pct"]) else 0
                    row.extend([str(util), str(mem_mb), str(mem_pct)])
                f.write(",".join(row) + "\n")


class Timer:
    """Simple timer for tracking phase durations."""
    
    def __init__(self):
        self.phases = {}
        self.current_phase = None
        self.phase_start = None
    
    def start_phase(self, name):
        """Start timing a phase."""
        if self.current_phase:
            self.end_phase()
        self.current_phase = name
        self.phase_start = time.time()
        print(f"\n{'='*60}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting: {name}")
        print(f"{'='*60}")
    
    def end_phase(self):
        """End current phase and record duration."""
        if self.current_phase and self.phase_start:
            duration = time.time() - self.phase_start
            self.phases[self.current_phase] = duration
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed: {self.current_phase}")
            print(f"Duration: {self._format_duration(duration)}")
            self.current_phase = None
            self.phase_start = None
    
    def _format_duration(self, seconds):
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.2f} hours"
    
    def get_summary(self):
        """Get summary of all phases."""
        return {name: self._format_duration(dur) for name, dur in self.phases.items()}
    
    def get_total(self):
        """Get total time across all phases."""
        return sum(self.phases.values())


def print_gpu_summary(summary, num_gpus=8):
    """Pretty print GPU summary."""
    print("\n" + "="*80)
    print("GPU USAGE SUMMARY")
    print("="*80)
    
    print(f"\n{'GPU':<6} {'Avg Util %':<12} {'Max Util %':<12} {'Avg Mem MB':<14} {'Max Mem MB':<14}")
    print("-"*60)
    
    for i in range(num_gpus):
        util_key = f"gpu{i}_util"
        mem_key = f"gpu{i}_mem_mb"
        
        if util_key in summary:
            avg_util = summary[util_key]["mean"]
            max_util = summary[util_key]["max"]
            avg_mem = summary[mem_key]["mean"]
            max_mem = summary[mem_key]["max"]
            print(f"GPU {i:<3} {avg_util:<12.1f} {max_util:<12.1f} {avg_mem:<14.0f} {max_mem:<14.0f}")
    
    if "aggregate" in summary:
        print("-"*60)
        print(f"Overall avg utilization: {summary['aggregate']['avg_utilization']:.1f}%")
        print(f"Monitoring duration: {summary['aggregate']['monitoring_duration_sec']:.0f} seconds")


def print_timing_summary(timer):
    """Pretty print timing summary."""
    print("\n" + "="*80)
    print("TIMING SUMMARY")
    print("="*80)
    
    for phase, duration in timer.get_summary().items():
        print(f"  {phase:<40} {duration}")
    
    print("-"*60)
    total = timer.get_total()
    if total < 60:
        print(f"  {'TOTAL':<40} {total:.1f} seconds")
    elif total < 3600:
        print(f"  {'TOTAL':<40} {total/60:.1f} minutes")
    else:
        print(f"  {'TOTAL':<40} {total/3600:.2f} hours")


def run_command(cmd, description=None):
    """Run a command and stream output."""
    if description:
        print(f"\n>>> {description}")
    print(f"Running: {' '.join(cmd)}")
    
    # Ensure PYTHONPATH includes the project root
    env = os.environ.copy()
    project_root = Path(__file__).parent.parent.resolve()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}:{existing_pythonpath}" if existing_pythonpath else str(project_root)
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(project_root),  # Run from project root
        env=env
    )
    
    # Stream output in real-time
    for line in process.stdout:
        print(line, end="")
    
    process.wait()
    return process.returncode


def run_precaching(hparams_file, overrides=None):
    """Run the precaching script."""
    cmd = ["python", "script/precache_embeddings.py", str(hparams_file)]
    if overrides:
        cmd.extend(overrides)
    return run_command(cmd, "Pre-caching encoder embeddings")


def run_training(hparams_file, overrides=None):
    """Run the training script."""
    cmd = ["python", "training/train.py", str(hparams_file)]
    if overrides:
        cmd.extend(overrides)
    return run_command(cmd, "Running training pipeline")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run experiment with GPU monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple interface (recommended)
  python script/run_experiment_with_monitoring.py --encoder wavrx --probe_type Tprobe --experiment_name wavrx-test
  
  # With additional overrides
  python script/run_experiment_with_monitoring.py --encoder wavlm --experiment_name wavlm-test --overrides "num_epochs: 3"
  
  # With GPU selection
  python script/run_experiment_with_monitoring.py --encoder wavrx --gpus 0,1 --overrides "num_augmentations: 3"
  
  # Legacy: direct config file
  python script/run_experiment_with_monitoring.py training/config/main.yaml --experiment_tag=exp1
        """
    )
    # Simple interface (new)
    parser.add_argument("--encoder", "-e", help="Encoder name (e.g., wavrx, wavlm, clap)")
    parser.add_argument("--probe_type", "-p", default="Tprobe", help="Probe type (default: Tprobe)")
    parser.add_argument("--experiment_name", "-n", help="Experiment name for output folder")
    parser.add_argument("--gpus", "-g", help="Comma-separated GPU indices (e.g., 0,1,2)")
    parser.add_argument("--overrides", "-o", help="YAML overrides as string (e.g., 'num_epochs: 3, lr: 0.001')")
    
    # Legacy interface
    parser.add_argument("hparams_file", nargs="?", default="training/config/main.yaml", 
                        help="Path to the hyperparameters YAML file (default: training/config/main.yaml)")
    parser.add_argument("--experiment_tag", "-t", help="[Legacy] Experiment tag for output folder naming")
    parser.add_argument("legacy_overrides", nargs="*", help="[Legacy] Additional overrides in key=value format")
    
    args = parser.parse_args()
    
    hparams_file = args.hparams_file
    
    # Build overrides list
    overrides = []
    
    # Handle new interface
    if args.encoder:
        overrides.append(f"--model_name={args.encoder}")
    if args.experiment_name:
        overrides.append(f"--experiment_tag={args.experiment_name}")
    elif args.experiment_tag:
        overrides.append(f"--experiment_tag={args.experiment_tag}")
    if args.gpus:
        # Set CUDA_VISIBLE_DEVICES for GPU selection
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        os.environ["HIP_VISIBLE_DEVICES"] = args.gpus  # For AMD GPUs
        print(f"Using GPUs: {args.gpus}")
    
    # Parse YAML-style overrides string
    if args.overrides:
        # Parse "key1: value1, key2: value2" format
        for pair in args.overrides.split(","):
            pair = pair.strip()
            if ":" in pair:
                key, value = pair.split(":", 1)
                overrides.append(f"--{key.strip()}={value.strip()}")
    
    # Handle legacy overrides
    if args.legacy_overrides:
        for o in args.legacy_overrides:
            if "=" in o and not o.startswith("--"):
                overrides.append(f"--{o}")
            else:
                overrides.append(o)
    
    overrides = overrides if overrides else None
    
    # Generate timestamp early (needed for temp config naming)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate temporary config file with correct encoder if --encoder is specified
    temp_config = None
    if args.encoder:
        import tempfile
        import re
        
        with open(hparams_file, 'r') as f:
            config_content = f.read()
        
        print("\n" + "="*60)
        print("CONFIG MODIFICATIONS:")
        print("="*60)
        
        # Replace the encoder include line
        # Match: encoder_params: !include:encoders/ANYTHING.yaml
        old_encoder_match = re.search(r'encoder_params:\s*!include:encoders/(\w+)\.yaml', config_content)
        if old_encoder_match:
            print(f"  encoder_params: {old_encoder_match.group(1)}.yaml -> {args.encoder}.yaml")
        config_content = re.sub(
            r'(encoder_params:\s*!include:encoders/)\w+\.yaml',
            rf'\g<1>{args.encoder}.yaml',
            config_content
        )
        
        # Also update model_name
        old_model_match = re.search(r'^model_name:\s*(\w+)', config_content, re.MULTILINE)
        if old_model_match:
            print(f"  model_name: {old_model_match.group(1)} -> {args.encoder}")
        config_content = re.sub(
            r'^model_name:\s*\w+',
            f'model_name: {args.encoder}',
            config_content,
            flags=re.MULTILINE
        )
        
        # Update probe_params if probe_type specified
        if args.probe_type:
            old_probe_match = re.search(r'probe_params:\s*!include:probes/(\w+)\.yaml', config_content)
            if old_probe_match:
                print(f"  probe_params: {old_probe_match.group(1)}.yaml -> {args.probe_type}.yaml")
            config_content = re.sub(
                r'(probe_params:\s*!include:probes/)\w+\.yaml',
                rf'\g<1>{args.probe_type}.yaml',
                config_content
            )
            old_probe_name = re.search(r'^probe_name:\s*(\w+)', config_content, re.MULTILINE)
            if old_probe_name:
                print(f"  probe_name: {old_probe_name.group(1)} -> {args.probe_type}")
            config_content = re.sub(
                r'^probe_name:\s*\w+',
                f'probe_name: {args.probe_type}',
                config_content,
                flags=re.MULTILINE
            )
        
        # Update experiment_tag if experiment_name specified
        if args.experiment_name:
            old_exp_match = re.search(r'^experiment_tag:\s*(\S+)', config_content, re.MULTILINE)
            if old_exp_match:
                print(f"  experiment_tag: {old_exp_match.group(1)} -> {args.experiment_name}")
            config_content = re.sub(
                r'^experiment_tag:\s*\S+.*$',
                f'experiment_tag: {args.experiment_name}',
                config_content,
                flags=re.MULTILINE
            )
        
        print("="*60)
        
        # Create temp file in the same directory as original config (so relative includes work)
        config_dir = Path(hparams_file).parent
        temp_config = config_dir / f"_temp_{args.encoder}_{timestamp}.yaml"
        with open(temp_config, 'w') as f:
            f.write(config_content)
        
        hparams_file = str(temp_config)
        print(f"Generated config: {temp_config}")
    
    # Print overrides for debugging
    if overrides:
        print("\n" + "="*60)
        print("COMMAND-LINE OVERRIDES:")
        print("="*60)
        for o in overrides:
            print(f"  {o}")
        print("="*60)
    
    # Use experiment name/tag in log directory name
    log_tag = args.experiment_name or args.experiment_tag
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_tag:
        log_dir = Path("./exps/monitoring_logs") / f"{timestamp}_{log_tag}"
    else:
        log_dir = Path("./exps/monitoring_logs") / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"EXPERIMENT WITH GPU MONITORING")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {hparams_file}")
    if args.encoder:
        print(f"Encoder: {args.encoder}")
    if args.probe_type:
        print(f"Probe: {args.probe_type}")
    if log_tag:
        print(f"Experiment: {log_tag}")
    if args.gpus:
        print(f"GPUs: {args.gpus}")
    if overrides:
        print(f"Overrides: {overrides}")
    print(f"Log directory: {log_dir}")
    print("="*80)
    
    # Initialize monitoring
    num_gpus = len(args.gpus.split(",")) if args.gpus else 8
    timer = Timer()
    monitor = GPUMonitor(interval=2.0, num_gpus=num_gpus)
    
    try:
        # Start GPU monitoring
        monitor.start()
        
        # Phase 1: Pre-caching
        timer.start_phase("Pre-caching Embeddings")
        return_code = run_precaching(hparams_file, overrides)
        timer.end_phase()
        
        if return_code != 0:
            print(f"\nError: Pre-caching failed with code {return_code}")
            raise RuntimeError("Pre-caching failed")
        
        # Phase 2: Training (HP search + final training)
        timer.start_phase("HP Search + Training")
        return_code = run_training(hparams_file, overrides)
        timer.end_phase()
        
        if return_code != 0:
            print(f"\nWarning: Training exited with code {return_code}")
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    finally:
        # Stop monitoring
        monitor.stop()
        
        # Get summaries
        gpu_summary = monitor.get_summary()
        
        # Print summaries
        print_gpu_summary(gpu_summary)
        print_timing_summary(timer)
        
        # Save detailed logs
        monitor.save_detailed_log(log_dir / "gpu_usage.csv")
        
        # Save summary JSON
        summary = {
            "timestamp": timestamp,
            "config": hparams_file,
            "encoder": args.encoder,
            "probe_type": args.probe_type,
            "experiment_name": args.experiment_name or args.experiment_tag,
            "gpus": args.gpus,
            "overrides": overrides,
            "gpu_summary": gpu_summary,
            "timing": {
                "phases": timer.phases,
                "total_seconds": timer.get_total()
            }
        }
        
        with open(log_dir / "experiment_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nLogs saved to: {log_dir}")
        print(f"  - gpu_usage.csv (detailed time series)")
        print(f"  - experiment_summary.json (summary stats)")
        
        # Clean up temp config
        if temp_config and temp_config.exists():
            temp_config.unlink()


if __name__ == "__main__":
    main()
