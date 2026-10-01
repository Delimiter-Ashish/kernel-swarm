"""Phase 3: the swarm.

An evolutionary, multi-agent search over Triton kernels (AlphaEvolve-style):

  Proposer agents   - each generation, several agents with different specialties
                      write children from parents chosen out of the population:
                        memory    : squeeze memory traffic (fusion, vectorization, fewer passes)
                        tuner     : launch config (BLOCK, num_warps, rows/program, autotune)
                        explorer  : a structurally different algorithm
                        crossover : merge the best ideas of two parents
  Repair agent      - gets one shot at fixing each child that fails, using the error
  Evaluator         - benchmarks every child (local GPU, or an SGE job array on SCC)
  Selector          - tournament selection + elitism keeps the population strong
  Blackboard        - every agent sees the swarm's leaderboard of ideas -> shared memory
  Analyst agent     - explains in plain words why the winning kernel is fast

Fitness = speedup vs PyTorch eager *measured on the same GPU in the same process*,
so results stay comparable even if SGE jobs land on different GPU models.

Usage:
  python -m kernel_swarm.evolve --task cross_entropy --generations 5 --children 6
  python -m kernel_swarm.evolve --task add_layernorm --eval sge      # parallel on the cluster
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kernel_swarm.agent import SYSTEM_PROMPT, check_source, extract_code, summarize
from kernel_swarm.bench import BASELINE_DIR, REPO_ROOT, evaluate, pretty
from kernel_swarm.llm import LLM
from kernel_swarm.tasks import TASKS

import inspect

RUNS_DIR = REPO_ROOT / "runs"
BEST_DIR = REPO_ROOT / "kernels" / "best"

PERSONAS = {
    "memory": "You are the MEMORY specialist. Your only goal is to cut memory traffic: read every byte once, "
              "fuse passes, use wide contiguous loads, keep intermediates in registers, avoid re-reading rows.",
    "tuner": "You are the TUNING specialist. Keep the parent's algorithm but find a better launch configuration: "
             "BLOCK sizes, num_warps, num_stages, rows per program, grid shape, or a small triton.autotune sweep "
             "keyed on the problem size.",
    "explorer": "You are the EXPLORER. Do NOT make a small tweak. Propose a structurally different algorithm "
                "(different work decomposition, different reduction strategy, persistent kernel, splitting rows "
                "across programs, single-pass vs two-pass, etc.). Bold ideas are welcome as long as they are correct.",
    "crossover": "You are the CROSSOVER specialist. You get two parent kernels. Combine the strongest idea from "
                 "each into one kernel that should beat both.",
    "fresh": "You are a kernel engineer starting from scratch. Write a correct, fast first version.",
}
ROTATION = ["memory", "tuner", "explorer", "crossover"]


def fitness(ind):
    r = ind["result"]
    return r["speedup_vs_eager"] if r.get("status") == "ok" else 0.0


def short(ind):
    r = ind["result"]
    if r.get("status") != "ok":
        return f"#{ind['id']} [{ind['persona']}] {r.get('status')}"
    bw = f", {r['pct_peak_bw']:.0f}% peak BW" if r.get("pct_peak_bw") else ""
    return f"#{ind['id']} [{ind['persona']}] {r['ms']:.4f} ms ({r['speedup_vs_eager']:.2f}x eager{bw})"


class Swarm:
    def __init__(self, task, llm, run_dir, children, pop_size, workers, eval_mode, seed):
        self.task, self.llm, self.run_dir = task, llm, run_dir
        self.children, self.pop_size, self.workers, self.eval_mode = children, pop_size, workers, eval_mode
        self.rng = random.Random(seed)
        self.individuals = []  # everything ever created (the lineage)
        self.gpu = "NVIDIA GPU"
        self.next_id = 0

    # ---------- bookkeeping ----------
    def new_individual(self, gen, persona, parents, code, idea, origin="llm"):
        ind = {"id": self.next_id, "gen": gen, "persona": persona, "parents": [p["id"] for p in parents],
               "idea": idea, "origin": origin, "code": code, "result": {"status": "pending"}}
        self.next_id += 1
        path = self.run_dir / "kernels" / f"g{gen:02d}_{ind['id']:03d}_{persona}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code or "# no code returned\n")
        ind["path"] = str(path)
        self.individuals.append(ind)
        return ind

    def alive(self):
        ok = [i for i in self.individuals if i["result"].get("status") == "ok"]
        return sorted(ok, key=fitness, reverse=True)[: self.pop_size]

    def save_lineage(self):
        slim = [{k: v for k, v in i.items() if k != "code"} for i in self.individuals]
        (self.run_dir / "lineage.json").write_text(json.dumps(slim, indent=1))

    # ---------- prompts ----------
    def task_header(self):
        t = self.task
        return "\n\n".join([
            f"## Task: `{t.name}`\n{t.description}",
            f"Inputs: {json.dumps(t.shape_info)}. Target GPU: {self.gpu}.",
            f"Tolerance: atol={t.atol}, rtol={t.rtol}.",
            "### PyTorch reference (must match exactly)\n```python\n" + inspect.getsource(t.reference) + "```",
        ])

    def blackboard(self):
        tried = [i for i in self.individuals if i["result"].get("status") != "pending"][-12:]
        if not tried:
            return ""
        lines = "\n".join(f"- {short(i)}: {i['idea'][:140]}" for i in tried)
        return f"### Swarm blackboard (recent ideas and their results; learn from them, don't repeat failures)\n{lines}"

    def child_prompt(self, persona, parents):
        parts = [self.task_header()]
        for n, p in enumerate(parents, 1):
            parts.append(f"### Parent {n}: {short(p)}\n```python\n{p['code']}\n```")
        parts.append(self.blackboard())
        if persona == "fresh":
            parts.append("Write a correct, fast kernel.")
        else:
            parts.append("Write a kernel that is FASTER than the parent(s) while staying correct.")
        return "\n\n".join(x for x in parts if x)

    def ask(self, persona, prompt):
        system = SYSTEM_PROMPT + "\n\n" + PERSONAS[persona]
        try:
            reply = self.llm.chat(system, prompt)
        except RuntimeError as e:
            return None, f"LLM error: {e}"
        idea = reply.split("```")[0].strip().replace("\n", " ")[:240]
        return extract_code(reply), idea

    # ---------- selection ----------
    def tournament(self, k=3):
        pool = self.alive()
        contenders = self.rng.sample(pool, min(k, len(pool)))
        return max(contenders, key=fitness)

    def plan_children(self):
        pool = self.alive()
        plans = []
        for c in range(self.children):
            if not pool:
                plans.append(("fresh", []))
                continue
            persona = ROTATION[c % len(ROTATION)]
            if persona == "crossover" and len(pool) < 2:
                persona = "explorer"
            if persona == "crossover":
                a = self.tournament()
                others = [p for p in pool if p["id"] != a["id"]]
                plans.append((persona, [a, self.rng.choice(others)]))
            elif c == 0:
                plans.append((persona, [pool[0]]))  # elitism: always push on the current champion
            else:
                plans.append((persona, [self.tournament()]))
        return plans

    # ---------- evaluation ----------
    def evaluate_batch(self, inds):
        todo = []
        for ind in inds:
            if not ind["code"]:
                ind["result"] = {"status": "no_code", "error": "Reply had no ```python code block."}
                continue
            problem = check_source(ind["code"])
            if problem:
                ind["result"] = {"status": "rejected", "error": problem}
                continue
            todo.append(ind)
        if self.eval_mode == "sge" and len(todo) > 1:
            results = evaluate_sge(self.task.name, [i["path"] for i in todo], self.run_dir)
        else:  # sequential on this GPU: slower, but timings are not disturbed by each other
            results = [evaluate(self.task.name, i["path"]) for i in todo]
        for ind, r in zip(todo, results):
            ind["result"] = r
            self.gpu = r.get("gpu") or self.gpu
        for ind in inds:
            print("   " + short(ind) + (f"  <- {ind['result'].get('error', '')[:160].strip()}"
                                         if ind["result"].get("status") not in ("ok",) else ""))

    def repair(self, gen, failed):
        """One repair attempt per failed child, in parallel."""
        def fix(ind):
            prompt = "\n\n".join([
                self.task_header(),
                f"### Your kernel\n```python\n{ind['code']}\n```",
                f"### It failed\n{summarize(ind['result'])}",
                "Fix the bug while keeping the optimization idea. Return the complete corrected module.",
            ])
            return self.ask(ind["persona"], prompt)
        with ThreadPoolExecutor(self.workers) as ex:
            outs = list(ex.map(fix, failed))
        fixed = [self.new_individual(gen, ind["persona"], [ind], code, f"(repair) {idea}", origin="repair")
                 for ind, (code, idea) in zip(failed, outs)]
        self.evaluate_batch(fixed)

    # ---------- main loop ----------
    def seed_population(self):
        seeds = []
        for label, path in (("baseline", BASELINE_DIR / f"{self.task.name}.py"),
                            ("previous_best", BEST_DIR / f"{self.task.name}.py")):
            if path.exists():
                seeds.append(self.new_individual(0, label, [], path.read_text(), f"seed: {label}", origin=label))
        if seeds:
            print("gen 0: evaluating seeds")
            self.evaluate_batch(seeds)

    def run_generation(self, gen):
        plans = self.plan_children()
        print(f"\n=== generation {gen}: {len(plans)} agents working "
              f"({', '.join(p for p, _ in plans)}) ===")
        with ThreadPoolExecutor(self.workers) as ex:
            outs = list(ex.map(lambda pl: self.ask(pl[0], self.child_prompt(*pl)), plans))
        kids = [self.new_individual(gen, persona, parents, code, idea)
                for (persona, parents), (code, idea) in zip(plans, outs)]
        self.evaluate_batch(kids)
        failed = [k for k in kids if k["result"].get("status") in ("incorrect", "runtime_error", "load_error",
                                                                     "timeout", "crash")]
        if failed:
            print(f"   repair agent fixing {len(failed)} failed kernel(s)...")
            self.repair(gen, failed)
        self.save_lineage()
        top = self.alive()
        if top:
            print(f"   champion: {short(top[0])}")

    def analyze(self, best):
        chain, cur = [], best
        by_id = {i["id"]: i for i in self.individuals}
        while cur:
            chain.append(cur)
            cur = by_id.get(cur["parents"][0]) if cur["parents"] else None
        story = "\n".join(f"- gen {c['gen']} {short(c)}: {c['idea'][:200]}" for c in reversed(chain))
        prompt = (self.task_header() + f"\n\n### Winning kernel\n```python\n{best['code']}\n```\n\n"
                  f"### Its lineage (oldest first)\n{story}\n\n"
                  "Write a short technical analysis in Markdown (under 300 words): what the kernel does, which "
                  "ideas along the lineage produced the speedups, and what limits further gains "
                  "(think in terms of memory bandwidth). No code blocks.")
        try:
            text = self.llm.chat("You are a concise GPU performance analyst.", prompt, max_tokens=2000)
        except RuntimeError as e:
            text = f"(analysis failed: {e})"
        (self.run_dir / "analysis.md").write_text(f"# Why the winner is fast: `{self.task.name}`\n\n"
                                                  f"## Lineage\n{story}\n\n## Analysis\n{text}\n")


# ---------- SGE job-array evaluator ----------
SGE_TEMPLATE = """#!/bin/bash -l
#$ -P {project}
#$ -N kswarm_eval
#$ -l h_rt=00:20:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -t 1-{n}
#$ -cwd
#$ -j y
#$ -o {logdir}/
module load python3/3.10.12
source kernel-swarm-venv/bin/activate
LINE=$(sed -n "${{SGE_TASK_ID}}p" {listfile})
KPATH=$(echo "$LINE" | cut -f1)
OUT=$(echo "$LINE" | cut -f2)
python -m kernel_swarm.bench --task {task} --kernel "$KPATH" --json --no-save > "$OUT.tmp" 2>/dev/null
mv "$OUT.tmp" "$OUT"
"""


def evaluate_sge(task_name, paths, run_dir):
    """Benchmark many kernels in parallel as one SGE job array (one GPU per kernel). Experimental."""
    batch = run_dir / "sge" / time.strftime("%H%M%S")
    (batch / "out").mkdir(parents=True, exist_ok=True)
    outs = [batch / "out" / f"{n}.json" for n in range(len(paths))]
    listfile = batch / "list.tsv"
    listfile.write_text("".join(f"{p}\t{o}\n" for p, o in zip(paths, outs)))
    script = batch / "eval.qsub"
    script.write_text(SGE_TEMPLATE.format(project=os.environ.get("SCC_PROJECT", "medaihack"), n=len(paths),
                                          logdir=batch, listfile=listfile, task=task_name))
    print(f"   submitting SGE job array of {len(paths)} GPU jobs (waits in queue if the cluster is busy)...")
    try:
        proc = subprocess.run(["qsub", "-sync", "y", str(script)], cwd=REPO_ROOT, capture_output=True, text=True)
    except FileNotFoundError:
        print("   qsub not available here -> falling back to local GPU")
        return [evaluate(task_name, p) for p in paths]
    if proc.returncode not in (0, 1):
        print(f"   qsub problem ({proc.returncode}): {proc.stderr.strip()[:300]} -> falling back to local GPU")
        return [evaluate(task_name, p) for p in paths]
    results = []
    for p, o in zip(paths, outs):
        r = None
        if o.exists():
            for line in reversed(o.read_text().splitlines()):
                if line.startswith("{"):
                    r = json.loads(line)
                    break
        results.append(r or {"task": task_name, "kernel": str(p), "status": "crash",
                             "error": "SGE job produced no result"})
    return results


def main():
    ap = argparse.ArgumentParser(description="Evolutionary multi-agent kernel swarm")
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--children", type=int, default=6, help="agents (children) per generation")
    ap.add_argument("--pop-size", type=int, default=8, help="survivors kept for parent selection")
    ap.add_argument("--workers", type=int, default=4, help="parallel LLM calls (lower it if you hit rate limits)")
    ap.add_argument("--eval", choices=["local", "sge"], default="local")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    task = TASKS[args.task]
    llm = LLM()
    run_dir = RUNS_DIR / args.task / ("swarm-" + time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Kernel Swarm | task={task.name} | {llm} | {args.generations} gens x {args.children} agents "
          f"| eval={args.eval}\nrun dir: {run_dir.relative_to(REPO_ROOT)}\n")

    swarm = Swarm(task, llm, run_dir, args.children, args.pop_size, args.workers, args.eval, args.seed)
    swarm.seed_population()
    t0 = time.time()
    for gen in range(1, args.generations + 1):
        swarm.run_generation(gen)
    swarm.save_lineage()

    print("\n================ leaderboard ================")
    for ind in swarm.alive()[:5]:
        print("  " + short(ind))
    top = swarm.alive()
    if not top:
        print("No correct kernel found. Check the errors above, or try more generations.")
        return
    best = top[0]
    n_ok = sum(1 for i in swarm.individuals if i["result"].get("status") == "ok")
    print(f"\n{len(swarm.individuals)} kernels created, {n_ok} correct, {time.time() - t0:.0f}s")

    if best["origin"] in ("baseline", "previous_best"):
        print("The swarm did not beat its seed this run; kernels/best is unchanged.")
    else:
        BEST_DIR.mkdir(parents=True, exist_ok=True)
        best_path = BEST_DIR / f"{task.name}.py"
        shutil.copy(best["path"], best_path)
        final = evaluate(task.name, best_path, with_compile=True)
        print("\nfinal check of the champion (with torch.compile):\n" + pretty(final))
        print(f"Champion saved to {best_path.relative_to(REPO_ROOT)}")
        (run_dir / "summary.json").write_text(json.dumps(
            {"task": task.name, "llm": repr(llm), "champion_id": best["id"], "final": final}, indent=2))
    print("analyst agent writing analysis.md ...")
    swarm.analyze(best)
    print(f"Done. Lineage + analysis in {run_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
