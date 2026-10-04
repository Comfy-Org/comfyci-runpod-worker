"""Rebuild the run index for one branch/lane from the committed summaries.

One-off backfill for runs published before the index existed, and the repair
tool if an index file ever needs regenerating. Normal runs maintain the index
incrementally from run_regression.py.

Usage:
  python build_index.py --branch master [--lane py312-torch2.11.0-cu128] [--no-meta]
Env: GITHUB_TOKEN for commit metadata lookups and (github storage) the push.
"""
from __future__ import annotations

import argparse
import sys

import index_builder
import storage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", required=True)
    ap.add_argument("--lane", default=None, help="lane id from manifest/lanes.json "
                                                 "(default: the lane matching --prefix)")
    ap.add_argument("--no-meta", action="store_true",
                    help="skip GitHub commit metadata lookups")
    ap.add_argument("--skip-publish", action="store_true", help="build locally, no push")
    storage.add_storage_args(ap)
    args = ap.parse_args()

    lanes = index_builder.load_lanes()
    if args.lane:
        if args.lane not in lanes["lanes"]:
            raise SystemExit(f"unknown lane {args.lane!r}; see manifest/lanes.json")
        args.prefix = lanes["lanes"][args.lane].get("prefix", args.prefix)
        lane = args.lane
    else:
        lane = index_builder.lane_for_prefix(lanes, args.prefix)
        if not lane:
            raise SystemExit(f"no lane in manifest/lanes.json has prefix {args.prefix!r}; pass --lane")
    layout = lanes["lanes"][lane].get("layout", 1)

    store = storage.from_args(args)
    builder = index_builder.IndexBuilder(store, args.branch, lane, layout, lanes)
    entries = builder.collect(with_meta=not args.no_meta)
    print(f"index: {len(entries)} entries collected for {args.branch}/{lane}")
    if args.skip_publish:
        builder.upsert(entries)
        return 0
    # Derived writer: regenerated on the freshest tree if the push retries
    # behind a concurrent publisher, like the per-run index update.
    store.register_derived(lambda st, b=builder, es=entries: b.upsert(es))
    store.finalize(f"index {args.branch}/{lane}: rebuilt {len(entries)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
