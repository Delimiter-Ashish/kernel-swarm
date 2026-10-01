"""Benchmark harness - the judge for every kernel in the swarm.

For a candidate file that defines `kernel_fn(*inputs)` it:
  1. checks correctness against the PyTorch reference on two random seeds
  2. times the kernel, PyTorch eager, and (optionally) torch.compile
  3. returns / prints a JSON result, and appends it to results/bench.jsonl

Candidates run in a separate subprocess with a timeout, so a kernel that
crashes, hangs, or corrupts the CUDA context can't take down the swarm.

Usage:
  python -m kernel_swarm.bench --task softmax --kernel kernels/baseline/softmax.py
  python -m kernel_swarm.bench --all-baselines --compile
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = REPO_ROOT / "results" / "bench.jsonl"
BASELINE_DIR = REPO_ROOT / "kernels" / "baseline"


def _load_kernel(path):
    spec = importlib.util.spec_from_file_location("candidate_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "kernel_fn"):
        raise AttributeError(f"{path} does not define kernel_fn(*inputs)")
    return mod.kernel_fn


def _short_tb(limit=3000):
    tb = traceback.format_exc()
    return tb[-limit:]


def evaluate_inprocess(task_name, kernel_path, with_compile=False):
    """Run the full evaluation in this process. Prefer evaluate() from outside."""
    import torch
    from triton.testing import do_bench
    from kernel_swarm.tasks import TASKS

    task = TASKS[task_name]
    result = {
        "task": task_name,
        "kernel": str(kernel_path),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": None,
    }
    if not torch.cuda.is_available():
        result["status"] = "no_gpu"
        return result

    try:
        fn = _load_kernel(kernel_path)
    except Exception:
        result.update(status="load_error", error=_short_tb())
        return result

    # Correctness on two different seeds (catches kernels that "memorize" one input).
    max_err = 0.0
    for seed in (0, 1):
        torch.manual_seed(seed)
        inputs = task.make_inputs()
        try:
            out = fn(*inputs)
            torch.cuda.synchronize()
        except Exception:
            result.update(status="runtime_error", error=_short_tb())
            return result
        ref = task.reference(*inputs)
        if not isinstance(out, torch.Tensor) or out.shape != ref.shape:
            got = tuple(out.shape) if isinstance(out, torch.Tensor) else type(out).__name__
            result.update(status="incorrect", error=f"shape mismatch: got {got}, expected {tuple(ref.shape)}")
            return result
        if out.dtype != ref.dtype:
            result.update(status="incorrect", error=f"dtype mismatch: got {out.dtype}, expected {ref.dtype}")
            return result
        diff = (out.float() - ref.float()).abs()
        if not torch.isfinite(out.float()).all():
            result.update(status="incorrect", error="output contains NaN/Inf")
            return result
        max_err = max(max_err, diff.max().item())
        if not torch.allclose(out.float(), ref.float(), atol=task.atol, rtol=task.rtol):
            result.update(status="incorrect", max_abs_err=max_err,
                          error=f"allclose failed (atol={task.atol}, rtol={task.rtol}), max_abs_err={max_err:.4g}")
            return result
    result["max_abs_err"] = max_err

    # Timing (milliseconds, lower is better).
    torch.manual_seed(0)
    inputs = task.make_inputs()
    try:
        t_kernel = do_bench(lambda: fn(*inputs), warmup=25, rep=100)
    except Exception:
        result.update(status="runtime_error", error=_short_tb())
        return result
    t_eager = do_bench(lambda: task.reference(*inputs), warmup=25, rep=100)
    result.update(status="ok", ms=t_kernel, eager_ms=t_eager, speedup_vs_eager=t_eager / t_kernel)

    if with_compile:
        # torch.compile also fuses ops - this is the honest, harder baseline.
        try:
            compiled = torch.compile(task.reference)
            compiled(*inputs)
            torch.cuda.synchronize()
            t_comp = do_bench(lambda: compiled(*inputs), warmup=25, rep=100)
            result.update(compile_ms=t_comp, speedup_vs_compile=t_comp / t_kernel)
        except Exception:
            result["compile_error"] = _short_tb(800)
    return result


def evaluate(task_name, kernel_path, with_compile=False, timeout=300):
    """Evaluate in an isolated subprocess. This is what agents should call."""
    cmd = [sys.executable, "-m", "kernel_swarm.bench", "--task", task_name,
           "--kernel", str(kernel_path), "--json", "--no-save"]
    if with_compile:
        cmd.append("--compile")
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"task": task_name, "kernel": str(kernel_path), "status": "timeout"}
    for line in reversed(proc.stdout.strip().splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                break
    return {"task": task_name, "kernel": str(kernel_path), "status": "crash",
            "error": (proc.stderr or proc.stdout)[-3000:]}


def save(result):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")


def pretty(r):
    head = f"[{r.get('status', '?'):>13}] {r['task']:<14} {Path(r['kernel']).name}"
    if r.get("status") == "ok":
        line = (f"{head}  kernel {r['ms']:.4f} ms | eager {r['eager_ms']:.4f} ms "
                f"| {r['speedup_vs_eager']:.2f}x vs eager")
        if "speedup_vs_compile" in r:
            line += f" | {r['speedup_vs_compile']:.2f}x vs torch.compile"
        return line + f" | max err {r['max_abs_err']:.2e} | {r.get('gpu')}"
    return head + ("\n" + r["error"] if r.get("error") else "")


def main():
    p = argparse.ArgumentParser(description="Kernel Swarm benchmark harness")
    p.add_argument("--task")
    p.add_argument("--kernel")
    p.add_argument("--all-baselines", action="store_true", help="bench every kernels/baseline/<task>.py")
    p.add_argument("--compile", action="store_true", help="also compare against torch.compile")
    p.add_argument("--json", action="store_true", help="run in-process and print one JSON line")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()

    if args.json:  # child-process mode used by evaluate()
        print(json.dumps(evaluate_inprocess(args.task, args.kernel, args.compile)))
        return

    if args.all_baselines:
        from kernel_swarm.tasks import TASKS
        jobs = [(t, BASELINE_DIR / f"{t}.py") for t in TASKS if (BASELINE_DIR / f"{t}.py").exists()]
    elif args.task and args.kernel:
        jobs = [(args.task, Path(args.kernel))]
    else:
        p.error("give --task and --kernel, or --all-baselines")

    for task, path in jobs:
        r = evaluate(task, path, with_compile=args.compile)
        print(pretty(r))
        if not args.no_save:
            save(r)


if __name__ == "__main__":
    main()
