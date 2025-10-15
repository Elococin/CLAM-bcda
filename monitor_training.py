#!/usr/bin/env python3
"""
Simple training progress monitor for running CLAM job
"""
import time
import re
from collections import defaultdict

def parse_training_log(log_file):
    """Parse training log to extract metrics"""
    epochs = []
    train_losses = []
    val_losses = []
    val_errors = []
    val_aucs = []
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
    except:
        return {}
    
    # Extract epoch completion info
    epoch_pattern = r'Epoch: (\d+), train_loss: ([\d.]+), train_clustering_loss:\s+([\d.]+), train_error: ([\d.]+)'
    val_pattern = r'Val Set, val_loss: ([\d.]+), val_error: ([\d.]+), auc: ([\d.]+)'
    
    epoch_matches = re.findall(epoch_pattern, content)
    val_matches = re.findall(val_pattern, content)
    
    for match in epoch_matches:
        epochs.append(int(match[0]))
        train_losses.append(float(match[1]))
    
    for match in val_matches:
        val_losses.append(float(match[0]))
        val_errors.append(float(match[1]))
        val_aucs.append(float(match[2]))
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_errors': val_errors,
        'val_aucs': val_aucs
    }

def get_gpu_stats():
    """Get recent GPU utilization stats"""
    try:
        gpu_file = '/autofs/space/crater_001/projects/bc_domain_adaptation_nicole/results/clam_sb_baseline/lodo_haiti_fullrun/gpu_utilization.log'
        with open(gpu_file, 'r') as f:
            lines = f.readlines()
        
        gpu_utils = []
        powers = []
        for line in lines[-20:]:  # Last 20 measurements
            if line.strip() and line[0].isdigit():
                parts = line.split()
                if len(parts) > 5:
                    gpu_utils.append(float(parts[4]))
                    powers.append(float(parts[1]))
        
        if gpu_utils:
            return {
                'avg_util': sum(gpu_utils) / len(gpu_utils),
                'max_util': max(gpu_utils),
                'avg_power': sum(powers) / len(powers),
                'samples': len(gpu_utils)
            }
    except:
        pass
    return {}

def print_progress_report():
    """Print comprehensive training progress report"""
    log_file = '/autofs/space/crater_001/projects/bc_domain_adaptation_nicole/logs/clam_er_lodo_optimized.sbatch-7255941.out'
    
    print("=" * 60)
    print("🚀 CLAM TRAINING PROGRESS REPORT")
    print("=" * 60)
    
    # Parse training metrics
    metrics = parse_training_log(log_file)
    
    if metrics.get('epochs'):
        print(f"📊 TRAINING METRICS:")
        print(f"   Completed Epochs: {len(metrics['epochs'])}")
        print(f"   Latest Epoch: {metrics['epochs'][-1]}")
        print(f"   Latest Train Loss: {metrics['train_losses'][-1]:.4f}")
        print()
    
    if metrics.get('val_aucs'):
        print(f"🎯 VALIDATION PERFORMANCE:")
        print(f"   Latest Val Loss: {metrics['val_losses'][-1]:.4f}")
        print(f"   Latest Val Error: {metrics['val_errors'][-1]:.4f}")
        print(f"   Latest Val AUC: {metrics['val_aucs'][-1]:.4f}")
        print()
    
    # GPU stats
    gpu_stats = get_gpu_stats()
    if gpu_stats:
        print(f"🔥 GPU UTILIZATION (last {gpu_stats['samples']} measurements):")
        print(f"   Average Utilization: {gpu_stats['avg_util']:.1f}%")
        print(f"   Peak Utilization: {gpu_stats['max_util']:.1f}%")
        print(f"   Average Power Draw: {gpu_stats['avg_power']:.1f}W")
        print()
    
    # Training trend
    if len(metrics.get('train_losses', [])) > 1:
        recent_loss = metrics['train_losses'][-1]
        prev_loss = metrics['train_losses'][-2] if len(metrics['train_losses']) > 1 else recent_loss
        trend = "📉 Decreasing" if recent_loss < prev_loss else "📈 Increasing"
        print(f"📈 LOSS TREND: {trend}")
        print()
    
    # Early stopping status
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        es_match = re.search(r'EarlyStopping counter: (\d+) out of (\d+)', content)
        if es_match:
            counter, total = es_match.groups()
            print(f"⏹️  EARLY STOPPING: {counter}/{total}")
            if int(counter) > 0:
                print(f"   Warning: Validation loss increased for {counter} epoch(s)")
            print()
    except:
        pass
    
    print("=" * 60)

if __name__ == "__main__":
    print_progress_report()
