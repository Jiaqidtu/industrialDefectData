"""Ablation driver: the adapter on/off sweep of the design document.

Strategies
----------
``add``     start from a fully frozen trunk and switch adapters on one stage at
            a time, deepest first: () -> (4) -> (3,4) -> (2,3,4) -> (1,2,3,4).
``remove``  start from all four adapters on and drop them one at a time, so the
            accuracy drop tells you which stage matters most.
``single``  one adapter at a time, plus the two reference points.
``full``    all 16 on/off combinations (expensive, only for a final table).
``neck``    fix the best adapter set and compare panet / bifpn / fpn.
``variant`` fix the adapter set and compare Hiera tiny / small / base_plus.

Every configuration is trained from scratch with identical data splits, epochs
and learning rate (controlled-variable protocol), optionally repeated with
several seeds so that the accuracy differences can be tested for significance
rather than read off a single run.

Example::

    python -m model.ablation --strategy add --epochs 60 --seeds 0,1,2 \
        --sam2-ckpt /path/sam2.1_hiera_tiny.pt --project runs/ablation
"""

import argparse
import json
import os
import statistics
from typing import Dict, List, Sequence

from .train import build_cfg_from_args, get_parser, train


def stage_sets(strategy: str) -> List[Sequence[int]]:
    if strategy == "add":
        return [(), (4,), (3, 4), (2, 3, 4), (1, 2, 3, 4)]
    if strategy == "remove":
        return [(1, 2, 3, 4), (2, 3, 4), (1, 3, 4), (1, 2, 4), (1, 2, 3)]
    if strategy == "single":
        return [(), (1,), (2,), (3,), (4,), (1, 2, 3, 4)]
    if strategy == "full":
        import itertools
        return [tuple(c) for n in range(5)
                for c in itertools.combinations((1, 2, 3, 4), n)]
    return [(1, 2, 3, 4)]


def tag(stages: Sequence[int]) -> str:
    return "s" + ("".join(str(s) for s in stages) if stages else "none")


def run_sweep(args) -> Dict:
    seeds = [int(s) for s in str(args.seeds).split(",") if s != ""]
    results: List[Dict] = []

    if args.strategy in ("add", "remove", "single", "full"):
        configs = [{"adapter_stages": ",".join(map(str, s)) or "none",
                    "neck": args.neck, "variant": args.variant,
                    "tag": tag(s)} for s in stage_sets(args.strategy)]
    elif args.strategy == "neck":
        configs = [{"adapter_stages": args.adapter_stages, "neck": n,
                    "variant": args.variant, "tag": f"neck_{n}"}
                   for n in ("panet", "bifpn", "fpn")]
    elif args.strategy == "variant":
        configs = [{"adapter_stages": args.adapter_stages, "neck": args.neck,
                    "variant": v, "tag": f"hiera_{v}"}
                   for v in ("tiny", "small", "base_plus")]
    else:
        raise ValueError(f"unknown strategy {args.strategy!r}")

    for c in configs:
        runs = []
        for seed in seeds:
            run_args = argparse.Namespace(**vars(args))
            run_args.adapter_stages = c["adapter_stages"]
            run_args.neck = c["neck"]
            run_args.variant = c["variant"]
            run_args.seed = seed
            run_args.name = f"{c['tag']}_seed{seed}"
            print(f"\n===== {run_args.name} "
                  f"(stages={c['adapter_stages']}, neck={c['neck']}, "
                  f"hiera={c['variant']}) =====", flush=True)
            summary = train(build_cfg_from_args(run_args), run_args)
            runs.append({
                "seed": seed,
                "mAP50": summary["best"]["mAP50"],
                "mAP50-95": summary["best"]["mAP50-95"],
                "trainable_M": summary["params"]["trainable_M"],
                "total_M": summary["params"]["total_M"],
                "GFLOPs": summary["efficiency"]["GFLOPs"],
                "FPS": summary["efficiency"]["FPS"],
            })

        def agg(key):
            vals = [r[key] for r in runs]
            return {"mean": round(statistics.mean(vals), 4),
                    "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0}

        results.append({**c, "runs": runs,
                        "mAP50": agg("mAP50"), "mAP50-95": agg("mAP50-95"),
                        "trainable_M": runs[0]["trainable_M"],
                        "total_M": runs[0]["total_M"],
                        "GFLOPs": runs[0]["GFLOPs"],
                        "FPS": agg("FPS")["mean"]})

        os.makedirs(args.project, exist_ok=True)
        with open(os.path.join(args.project, "ablation.json"), "w") as fh:
            json.dump({"strategy": args.strategy, "seeds": seeds,
                       "results": results}, fh, indent=2)

    print_table(results, args.strategy)
    write_markdown(results, os.path.join(args.project, "ablation.md"), args.strategy)
    return {"strategy": args.strategy, "results": results}


def print_table(results: List[Dict], strategy: str):
    print(f"\n===== ablation summary ({strategy}) =====")
    print(f"{'config':<16}{'trainable M':>12}{'total M':>10}{'GFLOPs':>9}"
          f"{'FPS':>8}{'mAP50':>16}{'mAP50-95':>16}")
    for r in results:
        print(f"{r['tag']:<16}{r['trainable_M']:>12.3f}{r['total_M']:>10.3f}"
              f"{r['GFLOPs']:>9.2f}{r['FPS']:>8.1f}"
              f"{r['mAP50']['mean']:>10.4f}±{r['mAP50']['std']:<5.4f}"
              f"{r['mAP50-95']['mean']:>10.4f}±{r['mAP50-95']['std']:<5.4f}")


def write_markdown(results: List[Dict], path: str, strategy: str):
    """Paper-ready markdown table (accuracy / parameters / speed trade-off)."""
    lines = [f"### Adapter ablation ({strategy})", "",
             "| config | adapter stages | trainable (M) | total (M) | GFLOPs | FPS "
             "| mAP@0.5 | mAP@0.5:0.95 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        lines.append(
            f"| {r['tag']} | {r['adapter_stages']} | {r['trainable_M']:.3f} | "
            f"{r['total_M']:.3f} | {r['GFLOPs']:.2f} | {r['FPS']:.1f} | "
            f"{r['mAP50']['mean']:.4f} ± {r['mAP50']['std']:.4f} | "
            f"{r['mAP50-95']['mean']:.4f} ± {r['mAP50-95']['std']:.4f} |")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[ablation] markdown table -> {path}")


def main(argv=None):
    ap = get_parser()
    ap.add_argument("--strategy", default="add",
                    choices=["add", "remove", "single", "full", "neck", "variant"])
    ap.add_argument("--seeds", default="0", help="comma separated, e.g. 0,1,2")
    args = ap.parse_args(argv)
    if args.project == "runs":
        args.project = os.path.join("runs", f"ablation_{args.strategy}")
    return run_sweep(args)


if __name__ == "__main__":
    main()
