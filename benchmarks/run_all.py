"""Full TurboQuant benchmark suite runner.

Runs all benchmarks in sequence:
  1. Distortion (theoretical validation)
  2. Memory    (KV cache memory profiling)
  3. Speed     (quantize/dequantize latency + throughput)
  4. Quality   (perplexity on WikiText-2) — optional, requires model download
  5. End-to-end (16 GB MacBook Air fit check)
"""
import argparse

from benchmarks.bench_distortion import run_distortion_benchmark
from benchmarks.bench_memory import run_memory_benchmark
from benchmarks.bench_speed import run_speed_benchmark
from benchmarks.bench_e2e import run_e2e_benchmark


def main():
    parser = argparse.ArgumentParser(description="Run all TurboQuant benchmarks")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device to use (default: cpu)")
    parser.add_argument("--quick", action="store_true",
                        help="Use fewer vectors for a faster run")
    parser.add_argument("--with-quality", action="store_true",
                        help="Also run the quality (perplexity) benchmark "
                             "(requires model download)")
    parser.add_argument("--quality-model", type=str,
                        default="meta-llama/Llama-3.2-1B",
                        help="HuggingFace model ID for the quality benchmark "
                             "(default: meta-llama/Llama-3.2-1B)")
    args = parser.parse_args()

    total = 5 if args.with_quality else 4
    n_vectors = 1000 if args.quick else 10_000

    # ------------------------------------------------------------------
    # 1 / total — Distortion benchmark
    # ------------------------------------------------------------------
    print(f"\n[1/{total}] Distortion benchmark")
    run_distortion_benchmark(n_vectors=n_vectors)

    # ------------------------------------------------------------------
    # 2 / total — Memory benchmark
    # ------------------------------------------------------------------
    print(f"\n[2/{total}] Memory benchmark")
    run_memory_benchmark()

    # ------------------------------------------------------------------
    # 3 / total — Speed benchmark
    # ------------------------------------------------------------------
    print(f"\n[3/{total}] Speed benchmark  (device={args.device})")
    run_speed_benchmark(device=args.device)

    # ------------------------------------------------------------------
    # 4 / total — Quality benchmark  (optional)
    # ------------------------------------------------------------------
    if args.with_quality:
        print(f"\n[4/{total}] Quality benchmark  (model={args.quality_model})")
        from benchmarks.bench_quality import run_quality_benchmark
        run_quality_benchmark(
            model_name=args.quality_model,
            bits_list=[2, 3, 4],
            device=args.device,
        )

    # ------------------------------------------------------------------
    # final / total — End-to-end benchmark
    # ------------------------------------------------------------------
    e2e_step = 5 if args.with_quality else 4
    print(f"\n[{e2e_step}/{total}] End-to-end benchmark (16 GB MacBook Air)")
    run_e2e_benchmark()

    print("\nAll benchmarks complete.")


if __name__ == "__main__":
    main()
