#!/usr/bin/env python3

import math
import sys
from multiprocessing import Pipe, Process


def slice(mink: int, maxk: int) -> float:
    """
    Calculate the partial sum of the series 1/(2k+1)² from mink to maxk-1.
    
    Args:
        mink: Start index (inclusive)
        maxk: End index (exclusive)
        
    Returns:
        The computed partial sum
    """
    s: float = 0.0
    for k in range(mink, maxk):
        s += 1.0 / (2 * k + 1) / (2 * k + 1)
    return s

def worker(mink: int, maxk: int, conn) -> None:
    """Worker function to compute slice and send result via pipe"""
    try:
        result = slice(mink, maxk)
        conn.send(result)
        conn.close()
    except Exception as e:
        conn.send(f"Error: {str(e)}")
        conn.close()

def pi(n: int) -> float:
    """
    Compute an approximation of π using multi-processing.
    
    Args:
        n: Number of terms in the series to compute (divided by 10 processes)
        
    Returns:
        Approximation of π using the computed sum
    """
    processes: list[Process] = []
    parent_conns: list = []
    unit: int = n // 10
    
    for i in range(10):
        mink = unit * i
        maxk = mink + unit
        parent_conn, child_conn = Pipe()
        p = Process(target=worker, args=(mink, maxk, child_conn))
        processes.append(p)
        parent_conns.append(parent_conn)
        p.start()
    
    # Collect results
    sums: list[float] = []
    for conn in parent_conns:
        result = conn.recv()
        if isinstance(result, str) and result.startswith("Error:"):
            print(result)
            sums.append(0.0)
        else:
            sums.append(float(result))
        conn.close()
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    return math.sqrt(sum(sums) * 8)

if __name__ == "__main__":
    try:
        n: int = int(sys.argv[1]) if len(sys.argv) > 1 else 10000000
        print(f"Calculating pi with {n} iterations using 10 processes...")
        print(pi(n))
    except (ValueError, IndexError):
        print(f"Usage: {sys.argv[0]} [iterations]")
        print("  iterations: Number of terms (default: 10000000)")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)