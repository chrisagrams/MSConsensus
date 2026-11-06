#!/usr/bin/env python3
"""Audit casanovo vocab vs. what's actually in each Lance dataset's `seq`.

Two phases per dataset:

  1. Discovery: scan every seq, extract mod-shaped tokens with a regex
     that covers all three observed conventions:
       a) inline bracketed:  C[57.0215]   (concensus style)
       b) inline signed:     M+15.995     (massive_kb style)
       c) leading mass:      -17.027ABC   (massive_kb N-terminal mod)
     Count frequencies of every distinct mod token observed.

  2. Coverage: tokenize each unique seq against the configured residue
     vocab from the dataset's yaml. Report
       n_peptides, n_covered, coverage_rate
     and the top OOV tokens (substrings the tokenizer choked on).

Designed for the existing Lance dumps where `seq` is a plain string column.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

# Inline bracket form like  X[+57.021500]  or  C[57.0215]
_INLINE_BRACKET_RE = re.compile(r"[A-Z]\[[^\]]+\]|\[[^\]]+\]-")
# Inline signed-decimal form like  M+15.995  or  C-17.027
_INLINE_SIGNED_RE = re.compile(r"[A-Z][+\-]\d+(?:\.\d+)?")
# Leading mass prefix like  -17.027ABC  or  +42.011XYZ  (no leading letter)
_LEADING_MASS_RE = re.compile(r"^[+\-]\d+(?:\.\d+)?")


def discover_mods(seq: str) -> list[str]:
    out: list[str] = []
    if not seq:
        return out
    m = _LEADING_MASS_RE.match(seq)
    if m:
        out.append(m.group(0))
    out.extend(_INLINE_BRACKET_RE.findall(seq))
    out.extend(_INLINE_SIGNED_RE.findall(seq))
    return out


def load_vocab(path: Path) -> list[str]:
    import yaml
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return list(cfg.get("residues", {}).keys())


def build_tokenizer(vocab: list[str]):
    sorted_vocab = sorted(vocab, key=len, reverse=True)

    def tokenize(s: str) -> tuple[list[str], list[str]]:
        i = 0
        toks: list[str] = []
        oovs: list[str] = []
        while i < len(s):
            matched = None
            for t in sorted_vocab:
                if s.startswith(t, i):
                    matched = t
                    break
            if matched:
                toks.append(matched)
                i += len(matched)
            else:
                # Capture the offending chunk: try to grab the whole mod
                # group (e.g. "C[57.0215]" or "M+15.995") for readability.
                if s[i] == "[":
                    end = s.find("]", i)
                    chunk = s[i:end + 1] if end != -1 else s[i:]
                    oovs.append(chunk)
                    i += len(chunk)
                else:
                    m = re.match(r"[A-Z]?[+\-]\d+(?:\.\d+)?", s[i:])
                    if m:
                        oovs.append(m.group(0))
                        i += len(m.group(0))
                    else:
                        oovs.append(s[i])
                        i += 1
        return toks, oovs

    return tokenize


def audit_lance(path: Path, vocab: list[str], sample: int | None = None,
                batch_size: int = 1_000_000):
    import lance
    tokenize = build_tokenizer(vocab)

    seen: set[str] = set()
    covered: set[str] = set()
    used_vocab: Counter = Counter()
    oov_counter: Counter = Counter()
    mod_counter: Counter = Counter()  # discovered mods (raw, dataset-side)
    n_rows = 0

    ds = lance.dataset(str(path))
    for batch in ds.scanner(columns=["seq"], batch_size=batch_size).to_batches():
        for seq in batch.column("seq").to_pylist():
            n_rows += 1
            if seq is None:
                continue
            for m in discover_mods(seq):
                mod_counter[m] += 1
            if seq in seen:
                if sample is not None and n_rows >= sample:
                    return _wrap(path, n_rows, seen, covered, used_vocab,
                                 oov_counter, mod_counter)
                continue
            seen.add(seq)
            toks, oovs = tokenize(seq)
            if not oovs:
                covered.add(seq)
                used_vocab.update(toks)
            else:
                oov_counter.update(oovs)
            if sample is not None and n_rows >= sample:
                return _wrap(path, n_rows, seen, covered, used_vocab,
                             oov_counter, mod_counter)
    return _wrap(path, n_rows, seen, covered, used_vocab, oov_counter, mod_counter)


def _wrap(path, n_rows, seen, covered, used_vocab, oov_counter, mod_counter):
    return {
        "path": str(path),
        "n_rows": n_rows,
        "n_peptides": len(seen),
        "n_covered": len(covered),
        "used_vocab": used_vocab,
        "oov_counter": oov_counter,
        "mod_counter": mod_counter,
    }


def merge(results):
    used = Counter()
    oov = Counter()
    mods = Counter()
    n_rows = 0
    n_pep = 0
    n_cov = 0
    for r in results:
        used.update(r["used_vocab"])
        oov.update(r["oov_counter"])
        mods.update(r["mod_counter"])
        n_rows += r["n_rows"]
        n_pep += r["n_peptides"]
        n_cov += r["n_covered"]
    return {
        "n_rows": n_rows,
        "n_peptides_per_split_summed": n_pep,
        "n_covered_per_split_summed": n_cov,
        "coverage_rate": n_cov / n_pep if n_pep else 0.0,
        "used_vocab": used,
        "oov_counter": oov,
        "mod_counter": mods,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=None,
                    help="Per-file row cap for smoke testing")
    args = ap.parse_args()

    DATA_ROOT = Path("data/")
    CFG = Path("casanovo/configs")

    # Each dataset audited against its own config. massive_kb inherits the
    # paper vocab (no residues block of its own), so we use the paper one.
    datasets = [
        ("concensus_spec_30M", CFG / "consensus.yaml", [
            DATA_ROOT / "concensus_spec_30M/casanovo_lance/train.lance",
            DATA_ROOT / "concensus_spec_30M/casanovo_lance/valid.lance",
        ]),
        ("massive_kb_v1_shared", CFG / "casanovo_paper.yaml", [
            DATA_ROOT / "massive_kb_v1_shared/casanovo_lance/train.lance",
            DATA_ROOT / "massive_kb_v1_shared/casanovo_lance/valid.lance",
        ]),
    ]

    summary = {}
    for name, cfg_path, paths in datasets:
        vocab = load_vocab(cfg_path)
        print(f"=== {name} (vocab: {cfg_path.name}, {len(vocab)} tokens) ===",
              file=sys.stderr)
        results = []
        for p in paths:
            t0 = time.time()
            r = audit_lance(p, vocab, sample=args.sample)
            dt = time.time() - t0
            print(f"  {p.name}: rows={r['n_rows']:,} "
                  f"unique={r['n_peptides']:,} covered={r['n_covered']:,} "
                  f"({dt:.1f}s)", file=sys.stderr)
            results.append(r)
        summary[name] = (vocab, merge(results))

    print()
    print("dataset\tn_rows\tn_unique_peptides\tn_covered\tcoverage_rate")
    for name, (_, s) in summary.items():
        print(f"{name}\t{s['n_rows']}\t{s['n_peptides_per_split_summed']}\t"
              f"{s['n_covered_per_split_summed']}\t{s['coverage_rate']:.6f}")

    print()
    print("=== effective vocab usage (count of token occurrences in covered peptides) ===")
    for name, (vocab, s) in summary.items():
        used = s["used_vocab"]
        unused = [t for t in vocab if t not in used]
        print(f"\n[{name}] {len(used)}/{len(vocab)} configured tokens used. "
              f"unused = {unused or '<none>'}")

    print()
    print("=== top OOV substrings (what the tokenizer choked on) ===")
    for name, (_, s) in summary.items():
        oov = s["oov_counter"]
        if not oov:
            print(f"[{name}] no OOV")
            continue
        print(f"\n[{name}] {len(oov):,} distinct OOV substrings; top 25:")
        for tok, n in oov.most_common(25):
            print(f"  {tok!r:<30} {n:>15,}")

    print()
    print("=== top discovered mod tokens in data (regex-based, vocab-agnostic) ===")
    for name, (_, s) in summary.items():
        mods = s["mod_counter"]
        if not mods:
            print(f"[{name}] no mods discovered")
            continue
        print(f"\n[{name}] {len(mods):,} distinct mod tokens; top 25:")
        for tok, n in mods.most_common(25):
            print(f"  {tok!r:<30} {n:>15,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
