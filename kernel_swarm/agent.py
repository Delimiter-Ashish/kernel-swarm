"""Phase 2: a single kernel-engineer agent.

Loop:  LLM writes a Triton kernel -> anti-cheat check -> benchmark harness
       -> feedback (errors / timings) -> LLM improves it -> ...
The fastest correct kernel is saved to kernels/best/<task>.py.

Usage:
  python -m kernel_swarm.agent --task add_layernorm --iters 8
  python -m kernel_swarm.agent --task softmax --iters 6
"""
import argparse
import inspect
import json
import re
import shutil
import time
from pathlib import Path

from kernel_swarm.bench import BASELINE_DIR, REPO_ROOT, evaluate, pretty
from kernel_swarm.llm import LLM
from kernel_swarm.tasks import TASKS

RUNS_DIR = REPO_ROOT / "runs"
BEST_DIR = REPO_ROOT / "kernels" / "best"

SYSTEM_PROMPT = """You are an elite GPU performance engineer who writes Triton kernels for NVIDIA GPUs.

Hard rules for every answer:
- Reply with exactly ONE ```python code block containing a complete, self-contained module.
- The module must define `kernel_fn(*inputs)` with the same signature and output (shape, dtype, values) as the reference.
- The heavy computation must happen inside your own @triton.jit kernel(s).
- Do NOT call the PyTorch op you are replacing (no torch.nn.functional, torch.softmax, torch.layer_norm,
  torch.compile, etc.). Allocating outputs with torch.empty / empty_like is fine.
- Accumulate in fp32 for numerical stability, store in the output dtype.
- Do not hardcode values from any particular input; the kernel is tested on random inputs.
- Before the code block, write at most 3 short sentences on the optimization idea you are trying.

Performance ideas worth considering: fusing all work into one pass over memory, choosing BLOCK sizes and
num_warps for the shape, processing multiple rows per program, vectorized loads, avoiding redundant
reads, using triton.autotune over a few configs, and minimizing the number of kernel launches."""

BANNED = [
    (r"torch\.nn\.functional|\bF\.[a-z_]+\(", "calls torch.nn.functional"),
    (r"torch\.softmax|\.softmax\(", "calls torch softmax"),
    (r"torch\.(nn\.)?layer_norm|torch\.native_layer_norm", "calls torch layer_norm"),
    (r"torch\.compile", "uses torch.compile"),
    (r"torch\.ops\.", "calls torch.ops"),
    (r"torch\.logsumexp|\.logsumexp\(|log_softmax\(", "calls torch logsumexp/log_softmax"),
]


def check_source(code):
    if "@triton.jit" not in code and "triton.jit" not in code:
        return "no @triton.jit kernel found"
    if "def kernel_fn" not in code:
        return "kernel_fn is not defined"
    for pattern, reason in BANNED:
        if re.search(pattern, code):
            return f"rejected by anti-cheat check: {reason}"
    return None


def extract_code(reply):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", reply, re.DOTALL)
    if not blocks:
        return None
    return max(blocks, key=len)  # the full module is the longest block


def summarize(r):
    if r.get("status") == "ok":
        return f"ok: {r['ms']:.4f} ms, {r['speedup_vs_eager']:.2f}x vs PyTorch eager, max_abs_err {r['max_abs_err']:.2e}"
    return f"{r.get('status')}: {(r.get('error') or '')[-1500:]}"


def build_prompt(task, gpu, best, last, history):
    parts = [
        f"## Task: `{task.name}`\n{task.description}",
        f"Inputs: {json.dumps(task.shape_info)}. Target GPU: {gpu}.",
        f"Tolerance: atol={task.atol}, rtol={task.rtol}.",
        "### PyTorch reference (must match exactly)\n```python\n" + inspect.getsource(task.reference) + "```",
    ]
    if best:
        parts.append(f"### Current best kernel ({best['result']['ms']:.4f} ms, "
                     f"{best['result']['speedup_vs_eager']:.2f}x vs eager)\n```python\n{best['code']}\n```")
    if last and last is not best:
        parts.append(f"### Your previous attempt\n```python\n{last['code']}\n```\nResult: {summarize(last['result'])}")
    if history:
        lines = "\n".join(f"- attempt {h['i']}: {h['result'].get('status')}"
                          + (f", {h['result']['ms']:.4f} ms" if h['result'].get('status') == 'ok' else "")
                          + (f"  (idea: {h['idea']})" if h.get('idea') else "")
                          for h in history)
        parts.append(f"### Attempt history\n{lines}")
    if best:
        parts.append("Write a kernel that is FASTER than the current best while staying correct. "
                     "Try a meaningfully different idea if recent attempts stalled.")
    elif last:
        parts.append("Fix the problem above and return a correct kernel.")
    else:
        parts.append("Write a correct, fast first version.")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Single-agent kernel optimization loop")
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--no-baseline", action="store_true", help="don't seed with the hand-written baseline")
    args = ap.parse_args()

    task = TASKS[args.task]
    llm = LLM()
    run_dir = RUNS_DIR / args.task / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"
    print(f"Kernel agent | task={task.name} | {llm} | run dir: {run_dir.relative_to(REPO_ROOT)}\n")

    best, last, history, gpu = None, None, [], "NVIDIA GPU"

    baseline = BASELINE_DIR / f"{task.name}.py"
    if baseline.exists() and not args.no_baseline:
        r = evaluate(task.name, baseline)
        print("baseline  " + pretty(r))
        gpu = r.get("gpu") or gpu
        if r.get("status") == "ok":
            best = {"i": "baseline", "code": baseline.read_text(), "result": r, "idea": "hand-written baseline"}

    for i in range(1, args.iters + 1):
        print(f"\n--- attempt {i}/{args.iters}: asking {llm.provider} ...")
        t0 = time.time()
        try:
            reply = llm.chat(SYSTEM_PROMPT, build_prompt(task, gpu, best, last, history))
        except RuntimeError as e:
            print(f"   {e}")
            break
        idea = reply.split("```")[0].strip().replace("\n", " ")[:200]
        code = extract_code(reply)
        path = run_dir / f"attempt_{i:02d}.py"

        if code is None:
            result = {"status": "no_code", "error": "Reply had no ```python code block."}
            code = ""
        else:
            path.write_text(code)
            problem = check_source(code)
            if problem:
                result = {"status": "rejected", "error": problem}
            else:
                result = evaluate(task.name, path)
                gpu = result.get("gpu") or gpu
        result.update(task=task.name, kernel=str(path))

        entry = {"i": i, "code": code, "result": result, "idea": idea}
        history.append(entry)
        last = entry
        with open(log_path, "a") as f:
            f.write(json.dumps({"i": i, "idea": idea, "result": result, "llm_s": round(time.time() - t0, 1)}) + "\n")

        print(f"   idea: {idea}")
        print("   " + pretty(result).replace("\n", "\n   "))
        if result.get("status") == "ok" and (best is None or result["ms"] < best["result"]["ms"]):
            best = entry
            print(f"   *** new best: {result['ms']:.4f} ms ({result['speedup_vs_eager']:.2f}x vs eager)")

    print("\n================ summary ================")
    if not best or best["i"] == "baseline":
        print("No agent kernel beat the starting point this run." if best else "No correct kernel found.")
        return
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    best_path = BEST_DIR / f"{task.name}.py"
    shutil.copy(best["result"]["kernel"], best_path)
    final = evaluate(task.name, best_path, with_compile=True)
    print(pretty(final))
    (run_dir / "summary.json").write_text(json.dumps(
        {"task": task.name, "llm": repr(llm), "best_attempt": best["i"], "final": final}, indent=2))
    print(f"Best kernel saved to {best_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
