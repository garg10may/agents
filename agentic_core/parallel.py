"""
Parallel execution utilities (thread/process pool, async)
"""
from concurrent.futures import ThreadPoolExecutor

def run_parallel(tasks):
    with ThreadPoolExecutor() as executor:
        return list(executor.map(lambda t: t(), tasks))
