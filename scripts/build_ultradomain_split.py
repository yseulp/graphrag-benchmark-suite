import argparse

from benchmark_pipeline.dataset import build_ultradomain_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen UltraDomain eval split")
    parser.add_argument("--output", required=True)
    parser.add_argument("--hf-id", default="TommyChien/UltraDomain")
    parser.add_argument("--split", default="train")
    parser.add_argument("--domains", nargs="+", default=["agriculture", "cs", "legal", "mix"])
    parser.add_argument("--samples-per-domain", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = build_ultradomain_split(
        output_path=args.output,
        hf_id=args.hf_id,
        split=args.split,
        domains=args.domains,
        samples_per_domain=args.samples_per_domain,
        seed=args.seed,
    )
    print(f"Wrote {len(rows)} rows: {args.output}")


if __name__ == "__main__":
    main()
