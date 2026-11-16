"""
System Resource Monitor
Author: Shaurya Shah

A lightweight CLI tool to monitor real-time CPU, Memory, and Disk usage
with warning thresholds and formatted metrics.
"""

import os
import shutil
import sys
import time

try:
    import psutil
except ImportError:
    psutil = None


def format_bytes(bytes_num: int) -> str:
    """Format bytes to human-readable strings (MB, GB, TB)."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_num < 1024:
            return f"{bytes_num:.2f} {unit}"
        bytes_num /= 1024
    return f"{bytes_num:.2f} PB"


def get_disk_usage(path: str = "/"):
    """Fetch disk usage statistics across platforms."""
    try:
        total, used, free = shutil.disk_usage(path)
        percent = (used / total) * 100
        return {
            "total": format_bytes(total),
            "used": format_bytes(used),
            "free": format_bytes(free),
            "percent": f"{percent:.1f}%",
            "warning": percent > 85.0
        }
    except Exception as e:
        return {
            "total": "N/A",
            "used": "N/A",
            "free": "N/A",
            "percent": "N/A",
            "warning": False
        }


def display_metrics():
    """Print current system health metrics."""
    print("=" * 55)
    print("           REAL-TIME SYSTEM RESOURCE MONITOR          ")
    print("=" * 55)

    # Disk Info
    root_path = os.path.abspath(os.sep)
    disk = get_disk_usage(root_path)
    print(f"\n[Disk Usage - Root Drive]")
    print(f"  Total Space: {disk['total']}")
    print(f"  Used Space:  {disk['used']} ({disk['percent']})")
    print(f"  Free Space:  {disk['free']}")
    if disk["warning"]:
        print("  [!] WARNING: Low disk space detected!")

    # CPU & Memory (if psutil is available)
    if psutil:
        cpu_pct = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory()
        print(f"\n[CPU & Memory Usage]")
        print(f"  CPU Utilization:    {cpu_pct}%")
        print(f"  RAM Used:           {format_bytes(mem.used)} / {format_bytes(mem.total)} ({mem.percent}%)")
        print(f"  RAM Available:      {format_bytes(mem.available)}")
        if cpu_pct > 90.0:
            print("  [!] WARNING: High CPU utilization!")
        if mem.percent > 90.0:
            print("  [!] WARNING: Critical memory usage!")
    else:
        print("\n[CPU & Memory Usage]")
        print("  Tip: Install 'psutil' (pip install psutil) to display live CPU and RAM stats.")

    print("\n" + "=" * 55)


def main():
    display_metrics()


if __name__ == "__main__":
    main()
