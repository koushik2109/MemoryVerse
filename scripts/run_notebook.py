#!/usr/bin/env python3
"""
scripts/run_notebook.py
Executes a Jupyter Notebook (.ipynb) locally, capturing outputs, streaming stdout/stderr,
and saving the executed notebook with populated outputs and execution counts.
"""
import io
import sys
import json
import time
import traceback
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

def execute_notebook(nb_path: str):
    path = Path(nb_path).resolve()
    if not path.exists():
        print(f"Error: Notebook not found at {path}", file=sys.stderr)
        sys.exit(1)

    print(f"=== Executing Notebook: {path.name} ===", flush=True)
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    global_ns = {
        "__name__": "__main__",
        "__file__": str(path),
    }

    # Inject friendly display handler
    def custom_display(*args):
        for arg in args:
            if hasattr(arg, "to_string"):
                print(arg.to_string())
            else:
                print(str(arg))

    global_ns["display"] = custom_display

    code_cell_idx = 0
    start_total = time.time()

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue

        code_cell_idx += 1
        source = "".join(cell["source"])
        first_line = source.splitlines()[0] if source.splitlines() else f"Cell {idx}"
        print(f"\n[Cell {code_cell_idx}] {first_line}", flush=True)
        print("-" * 60, flush=True)

        cell["execution_count"] = code_cell_idx
        cell["outputs"] = []

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        t0 = time.time()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(source, global_ns)
            dt = time.time() - t0

            stdout_val = stdout_capture.getvalue()
            stderr_val = stderr_capture.getvalue()

            if stdout_val:
                print(stdout_val, end="", flush=True)
                cell["outputs"].append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": stdout_val.splitlines(keepends=True)
                })

            if stderr_val:
                print(stderr_val, end="", file=sys.stderr, flush=True)
                cell["outputs"].append({
                    "name": "stderr",
                    "output_type": "stream",
                    "text": stderr_val.splitlines(keepends=True)
                })

            print(f"(Cell {code_cell_idx} completed in {dt:.2f}s)", flush=True)

        except Exception as e:
            dt = time.time() - t0
            tb_str = traceback.format_exc()
            print(f"ERROR in Cell {code_cell_idx} after {dt:.2f}s:\n{tb_str}", file=sys.stderr, flush=True)
            cell["outputs"].append({
                "ename": type(e).__name__,
                "evalue": str(e),
                "output_type": "error",
                "traceback": tb_str.splitlines(keepends=True)
            })
            # Save partial progress
            with open(path, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=1)
            sys.exit(1)

    total_time = time.time() - start_total
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"\n{'=' * 60}", flush=True)
    print(f"SUCCESS: All {code_cell_idx} code cells executed in {total_time:.2f}s.", flush=True)
    print(f"Updated notebook saved to: {path}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_notebook.py <path_to_notebook.ipynb>")
        sys.exit(1)
    execute_notebook(sys.argv[1])
