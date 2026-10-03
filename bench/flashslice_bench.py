"""
Eager vs fused (FlashSlice) slice/deslice in JAX: step time and peak memory.

Implementations: "eager" materializes the slice weights; "fused" is the
streamed pure-JAX path (`backend="jax"`); "triton" runs the reference's Triton
kernels through jax-triton (`backend="triton"`, G and D powers of two in
[16, 128]; included when jax-triton is importable); "transolver3" is the
Transolver-3 port, in the full-model sweep.

Meant for a GPU (written with an H100 in mind). Every case runs in its own
subprocess, so each peak-memory reading starts clean and an out-of-memory case
is reported as OOM instead of ending the sweep.

    python bench/flashslice_bench.py                     # all sweeps, fp32
    python bench/flashslice_bench.py --sweep layer       # one layer, G and N sweep
    python bench/flashslice_bench.py --sweep model       # L=8 model, + Transolver-3
    python bench/flashslice_bench.py --sweep chunk       # fused chunk_size sweep
    python bench/flashslice_bench.py --dtype bf16        # bf16 params and inputs
    python bench/flashslice_bench.py --precision highest # exact fp32 dots (no TF32)
    python bench/flashslice_bench.py --quick             # small sizes, smoke test
    python bench/flashslice_bench.py --mode both         # inference rows too
    python bench/flashslice_bench.py --dtype bf16 --triton-dot bf16

Columns: median time of a compiled step (ms); peak device memory in use
(GiB, including parameters and inputs) where the backend reports it, else
XLA's compiled temp allocation. "train" is forward + backward (`jax.grad`),
"infer" is forward only. Results are also appended as JSON lines to --out.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

# Peak-memory readings need JAX to allocate on demand rather than grabbing 75%
# of the device up front. Must be set before JAX initializes its backend.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

GiB = 2**30


# ---------------------------------------------------------------------------
# One case, run inside a fresh subprocess
# ---------------------------------------------------------------------------
def run_case(case):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from zoo._flashslice import PhysicsAttentionIrregularMesh, Transolver
    from zoo._flashslice import init_weights
    from zoo._transolver_3 import Transolver as Transolver3
    from zoo._transolver_3 import init_weights as t3_init_weights

    dtype = {"fp32": jnp.float32, "bf16": jnp.bfloat16}[case["dtype"]]
    B, N, C, H, G = 1, case["N"], case["C"], case["H"], case["G"]
    k = jr.split(jr.key(0), 4)

    if case["kind"] == "layer":
        module = PhysicsAttentionIrregularMesh(
            C, H, C // H, G,
            use_fused_slice=case["impl"] in ("fused", "triton"),
            chunk_size=case["chunk"],
            backend="triton" if case["impl"] == "triton" else "jax",
            dot=case["triton_dot"],
            key=k[0],
        )  # fmt: skip
        inputs = (jr.normal(k[1], (B, N, C)),)

        def call(m, x):
            return m(x, key=k[3], inference=True)[0]

    elif case["impl"] == "transolver3":
        module = Transolver3(
            space_dim=3, fun_dim=1, out_dim=4, num_attn_layers=case["L"],
            hidden_dim=C, num_heads=H, num_slices=G, mlp_ratio=2, key=k[0],
        )  # fmt: skip
        module = t3_init_weights(module, key=k[1])
        inputs = (jr.normal(k[2], (B, N, 4)),)

        def call(m, x):
            return m(x, key=k[3], inference=True)

    else:
        module = Transolver(
            space_dim=3, fun_dim=1, out_dim=4, num_layers=case["L"],
            hidden_dim=C, num_heads=H, num_slices=G, mlp_ratio=2,
            use_fused_slice=case["impl"] in ("fused", "triton"),
            chunk_size=case["chunk"],
            backend="triton" if case["impl"] == "triton" else "jax",
            dot=case["triton_dot"],
            key=k[0],
        )  # fmt: skip
        module = init_weights(module, key=k[1])
        inputs = (jr.normal(k[2], (B, N, 3)), jr.normal(k[3], (B, N, 1)))

        def call(m, x, fx):
            return m(x, fx, key=k[3], inference=True)

    def cast(t):
        return t.astype(dtype) if eqx.is_inexact_array(t) else t

    params, static = eqx.partition(jax.tree.map(cast, module), eqx.is_array)
    inputs = tuple(cast(x) for x in inputs)

    def forward(p, *xs):
        return call(eqx.combine(p, static), *xs)

    def loss(p, *xs):
        return jnp.mean(forward(p, *xs).astype(jnp.float32) ** 2)

    fn = jax.jit(jax.grad(loss) if case["mode"] == "train" else forward)
    with jax.default_matmul_precision(case["precision"]):
        compiled = fn.lower(params, *inputs).compile()
    temp = compiled.memory_analysis()
    temp = None if temp is None else temp.temp_size_in_bytes / GiB

    for _ in range(case["warmup"]):
        jax.block_until_ready(compiled(params, *inputs))
    times = []
    for _ in range(case["reps"]):
        t0 = time.perf_counter()
        jax.block_until_ready(compiled(params, *inputs))
        times.append(time.perf_counter() - t0)
    times.sort()

    stats = jax.devices()[0].memory_stats() or {}
    peak = stats.get("peak_bytes_in_use")
    return {
        "ms": 1e3 * times[len(times) // 2],
        "peak_gib": None if peak is None else peak / GiB,
        "temp_gib": temp,
        "device": str(jax.devices()[0].device_kind),
    }


def spawn(case, timeout):
    """Run one case in a fresh interpreter; returns its result or an error tag."""
    cmd = [sys.executable, __file__, "--case", json.dumps(case)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=os.environ
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    err = proc.stderr
    if "RESOURCE_EXHAUSTED" in err or "out of memory" in err.lower():
        return {"error": "OOM"}
    raised = [ln for ln in err.splitlines() if re.match(r"\w+(Error|Exception)\b", ln)]
    return {
        "error": (raised or err.strip().splitlines()[-1:] or ["no output"])[-1][:120]
    }


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
def memory(r):
    if "error" in r:
        return None
    return r["peak_gib"] if r["peak_gib"] is not None else r["temp_gib"]


def fmt(v, spec):
    return "—" if v is None else format(v, spec)


def cell(r):
    return r["error"] if "error" in r else f"{r['ms']:.1f}"


def compare(rows, title, base, out):
    """rows: (label, {impl: case}); prints eager vs fused (and extras)."""
    print(f"\n### {title}\n", flush=True)
    extras = []
    for _, cases in rows:
        extras += [i for i in cases if i not in ("eager", "fused", *extras)]
    head = "| case | eager ms | fused ms | speedup | eager GiB | fused GiB | saved |"
    head += "".join(f" {e} ms | {e} GiB |" for e in extras)
    print(head)
    print("|" + "---|" * (head.count("|") - 1), flush=True)
    for label, cases in rows:
        res = {}
        for impl, case in cases.items():
            res[impl] = spawn({**base, **case}, base["timeout"])
            with open(out, "a") as f:
                f.write(json.dumps({**base, **case, **res[impl]}) + "\n")
        e, fu = res["eager"], res["fused"]
        speed = e["ms"] / fu["ms"] if "error" not in e and "error" not in fu else None
        me, mf = memory(e), memory(fu)
        saved = 100 * (1 - mf / me) if me and mf else None
        line = (
            f"| {label} | {cell(e)} | {cell(fu)} | {fmt(speed, '.2f')}x "
            f"| {fmt(me, '.2f')} | {fmt(mf, '.2f')} | {fmt(saved, '.0f')}% |"
        )
        for x in extras:
            r = res.get(x, {"error": "—"})
            line += f" {cell(r)} | {fmt(memory(r), '.2f')} |"
        print(line, flush=True)


def _have_jax_triton():
    try:
        import jax_triton  # noqa: F401
    except ImportError:
        return False
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--case", help=argparse.SUPPRESS)
    p.add_argument("--sweep", choices=["all", "layer", "model", "chunk"], default="all")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument(
        "--precision",
        choices=["default", "high", "highest"],
        default="default",
        help="jax.default_matmul_precision; 'default' allows TF32 on H100",
    )
    p.add_argument("--chunk", type=int, default=4096, help="fused chunk_size")
    p.add_argument(
        "--mode",
        choices=["train", "infer", "both"],
        default="train",
        help="training steps (forward + backward), inference, or both",
    )
    p.add_argument(
        "--triton-dot",
        choices=["ieee", "tf32", "bf16v", "bf16", "tf32x3"],
        default=None,
        help="dot precision of the Triton kernels; default ieee for fp32, bf16 "
        "for bf16 inputs",
    )
    p.add_argument("--no-triton", action="store_true", help="skip the Triton rows")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--timeout", type=int, default=1800, help="seconds per case")
    p.add_argument("--quick", action="store_true", help="small sizes, smoke test")
    p.add_argument("--out", default="flashslice_bench.jsonl")
    args = p.parse_args()

    if args.case:
        print("RESULT " + json.dumps(run_case(json.loads(args.case))), flush=True)
        return

    triton_dot = args.triton_dot or ("bf16" if args.dtype == "bf16" else "ieee")
    use_triton = not args.no_triton and _have_jax_triton()
    modes = ("train", "infer") if args.mode == "both" else (args.mode,)
    base = dict(
        dtype=args.dtype, precision=args.precision, chunk=args.chunk,
        triton_dot=triton_dot,
        reps=args.reps, warmup=args.warmup, timeout=args.timeout, C=256, H=8,
    )  # fmt: skip
    if args.quick:
        base.update(reps=3, warmup=1)
        layer_ns, layer_gs, model_ns, model_gs = [8192], [32, 256], [8192], [32]
        chunk_n, chunks = 8192, [1024, 8192]
    else:
        layer_ns, layer_gs = [262144, 1048576], [32, 64, 128, 256, 512, 1024]
        model_ns, model_gs = [262144, 1048576], [32, 128]
        chunk_n, chunks = 1048576, [1024, 2048, 4096, 8192, 16384, 65536]

    print(
        f"dtype={args.dtype} precision={args.precision} chunk_size={args.chunk} "
        f"C=256 H=8 D=32 B=1; results appended to {args.out}"
    )
    print(
        f"triton: dot={triton_dot}"
        if use_triton
        else "triton: skipped (jax-triton not importable, or --no-triton)"
    )

    def impls(kind, mode, N, G, L=None, t3=False, **extra):
        case = dict(kind=kind, mode=mode, N=N, G=G, L=L, **extra)
        out = {"eager": {**case, "impl": "eager"}, "fused": {**case, "impl": "fused"}}
        if use_triton and 16 <= G <= 128 and G & (G - 1) == 0:
            out["triton"] = {**case, "impl": "triton"}
        if t3:
            out["transolver3"] = {**case, "impl": "transolver3"}
        return out

    if args.sweep in ("all", "layer"):
        rows = [
            (f"{mode}, N={N // 1024}k, G={G}", impls("layer", mode, N, G))
            for N in layer_ns
            for G in layer_gs
            for mode in modes
        ]
        compare(rows, "One attention layer", base, args.out)

    if args.sweep in ("all", "model"):
        rows = [
            (f"{mode}, N={N // 1024}k, G={G}", impls("model", mode, N, G, 8, t3=True))
            for N in model_ns
            for G in model_gs
            for mode in modes
        ]
        compare(rows, "Full model, L=8, mlp_ratio=2", base, args.out)

    if args.sweep in ("all", "chunk"):
        print(
            f"\n### Fused chunk_size, one layer, train, N={chunk_n // 1024}k, G=256\n"
        )
        print("| chunk_size | ms | GiB |\n|---|---|---|", flush=True)
        for chunk in chunks:
            case = dict(kind="layer", mode="train", N=chunk_n, G=256, L=None)
            case.update(impl="fused", chunk=chunk)
            r = spawn({**base, **case}, base["timeout"])
            with open(args.out, "a") as f:
                f.write(json.dumps({**base, **case, **r}) + "\n")
            print(f"| {chunk} | {cell(r)} | {fmt(memory(r), '.2f')} |", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    main()
