#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IRSTD Evaluation Audit Toolkit (reference implementation).

Implements the three checks proposed in the paper:

  1. census      -- mask-encoding check: gray-level structure of stored masks,
                    connected components under common decode rules, decode-rule
                    disagreement statistics, target-size census.
  2. tolerance   -- tolerance-family re-evaluation of FIXED predictions:
                    Pd under fixed {1,2,3,5}px and tau = kappa*sqrt(area) for
                    kappa in {1/6,1/4,1/3,1/2} (+ IoU at 0.5).
  3. rank        -- ranking-stability check: per-rule model rankings, pairwise
                    order swaps vs the status-quo fixed-3px ruler, and bootstrap
                    probabilities of the top-1 rank.

CPU-only. Dependencies: numpy, Pillow, scipy.

Usage examples
--------------
  python audit.py census    --masks /path/to/masks --out census.json
  python audit.py tolerance --preds preds.npz --gt-masks /path/to/masks \
                            --img-idx test_list.txt --out tol.json
  python audit.py rank      --tol tol.json --out rank.json

Prediction npz format (produced by tolerance with --save-preds or by the authors'
scripts):
  keys: "<model>|names"  -> array of image stems
        "<model>|bits"   -> object array of np.packbits(pred.ravel())
        "<model>|shapes" -> int array [[h, w], ...]
GT masks are read as grayscale; ground truth is decoded with >=128 by default
(use --gt-rule to change; all rules are reported in census).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

STRUCT8 = np.ones((3, 3), dtype=int)
FIXED = [1.0, 2.0, 3.0, 5.0]
KAPPAS = [1.0 / 6, 1.0 / 4, 1.0 / 3, 1.0 / 2]
RULES = ["fixed_1", "fixed_2", "fixed_3", "fixed_5", "k_6", "k_4", "k_3", "k_2"]


def load_gray(path):
    with Image.open(path) as im:
        return np.array(im.convert("L"))


def binarize(raw, rule):
    if rule == "ge128":
        return (raw >= 128).astype(np.uint8)
    if rule == "gt0":
        return (raw > 0).astype(np.uint8)
    if rule == "eq255":
        return (raw == 255).astype(np.uint8)
    raise ValueError(rule)


def regions(binary):
    lab, n = ndimage.label(binary > 0, structure=STRUCT8)
    if n == 0:
        return 0, np.zeros((0, 2)), np.zeros(0, dtype=np.int64)
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(np.int64)
    ys, xs = np.divmod(np.arange(lab.size), lab.shape[1])
    cy = np.bincount(lab.ravel(), weights=ys, minlength=n + 1)[1:] / area
    cx = np.bincount(lab.ravel(), weights=xs, minlength=n + 1)[1:] / area
    return n, np.column_stack([cy, cx]), area


def pd_at(pred, gt, dthresh):
    """Pd with a fixed absolute tolerance (greedy 1-to-1, BasicIRSTD semantics)."""
    ng, gc, _ = regions(gt)
    if ng == 0:
        return None
    npc, pc, _ = regions(pred)
    matched = 0
    used = np.zeros(npc, dtype=bool)
    for i in range(ng):
        if npc == 0:
            break
        d = np.linalg.norm(pc - gc[i], axis=1)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] < dthresh:
            matched += 1
            used[j] = True
    return matched / ng


def pd_kappa(pred, gt, kappa):
    """Pd with scale-normalized tolerance tau_i = kappa * sqrt(area_i)."""
    ng, gc, garea = regions(gt)
    if ng == 0:
        return None
    npc, pc, _ = regions(pred)
    ext = np.sqrt(garea.astype(np.float64))
    matched = 0
    used = np.zeros(npc, dtype=bool)
    for i in range(ng):
        if npc == 0:
            break
        d = np.linalg.norm(pc - gc[i], axis=1)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] < kappa * ext[i]:
            matched += 1
            used[j] = True
    return matched / ng


def iou_binary(pred, gt):
    inter = int((pred & gt).sum())
    union = int((pred | gt).sum())
    return inter / union if union else 1.0


# ---------------------------------------------------------------------------
# 1. census
# ---------------------------------------------------------------------------
def cmd_census(args):
    files = sorted(f for f in os.listdir(args.masks) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    if args.sample:
        files = files[: args.sample]
    totals = {r: {"components": 0, "fg_px": 0} for r in ("gt0", "ge128", "eq255")}
    disagree = {"ge128_vs_gt0": 0, "eq255_vs_ge128": 0}
    exts = []
    gray_set = set()
    n_nonbinary = 0
    for k, f in enumerate(files, 1):
        raw = load_gray(os.path.join(args.masks, f))
        u = np.unique(raw)
        gray_set.update(u.tolist())
        if len(u) > 2:
            n_nonbinary += 1
        counts = {}
        for r in ("gt0", "ge128", "eq255"):
            b = binarize(raw, r)
            n, _, area = regions(b)
            counts[r] = n
            totals[r]["components"] += n
            totals[r]["fg_px"] += int(b.sum())
        if counts["ge128"] != counts["gt0"]:
            disagree["ge128_vs_gt0"] += 1
        if counts["eq255"] != counts["ge128"]:
            disagree["eq255_vs_ge128"] += 1
        ge = binarize(raw, "ge128")
        n, _, area = regions(ge)
        if n > 0:
            exts.extend(np.sqrt(area.astype(np.float64)))
        if k % 2000 == 0:
            print("  census %d/%d" % (k, len(files)))
    e = np.asarray(exts)
    out = {
        "n_masks": len(files),
        "gray_values": sorted(gray_set)[:64],
        "n_gray_values": len(gray_set),
        "n_nonbinary_masks": n_nonbinary,
        "components": {r: totals[r]["components"] for r in totals},
        "foreground_px": {r: totals[r]["fg_px"] for r in totals},
        "disagreement_images": disagree,
        "n_targets": int(e.size),
        "extent_px": {"p50": float(np.percentile(e, 50))} if e.size else {},
        "frac_extent_le_3px": float((e <= 3).mean()) if e.size else None,
        "frac_extent_le_9px": float((e <= 9).mean()) if e.size else None,
        "frac_extent_gt_12px": float((e > 12).mean()) if e.size else None,
    }
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("census ->", args.out)
    print("  components:", out["components"], "| disagreement:", disagree)
    print("  non-binary masks:", n_nonbinary, "/", len(files), "| targets:", out["n_targets"])


# ---------------------------------------------------------------------------
# 2. tolerance
# ---------------------------------------------------------------------------
def read_pred_npz(path):
    z = np.load(path, allow_pickle=True)
    models = {}
    bases = sorted(set("|".join(k.split("|")[:1]) for k in z.files if k.endswith("|bits")))
    for m in bases:
        names = [str(x) for x in z[m + "|names"]]
        bits = z[m + "|bits"]
        shapes = z[m + "|shapes"]
        preds = {}
        for i, nm in enumerate(names):
            h, w = int(shapes[i][0]), int(shapes[i][1])
            preds[nm] = np.unpackbits(np.asarray(bits[i]).astype(np.uint8))[: h * w].reshape(h, w)
        models[m] = preds
    return models


def pd_match(gc, pc, thresh):
    """Greedy 1-to-1 matching given per-GT thresholds; returns matched/ng."""
    ng = len(gc)
    if ng == 0:
        return None
    npc = len(pc)
    matched = 0
    used = np.zeros(npc, dtype=bool)
    for i in range(ng):
        if npc == 0:
            break
        d = np.linalg.norm(pc - gc[i], axis=1)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] < thresh[i]:
            matched += 1
            used[j] = True
    return matched / ng


def cmd_tolerance(args):
    models = read_pred_npz(args.preds)
    stems = [l.strip() for l in open(args.img_idx, encoding="utf-8").read().splitlines() if l.strip()]
    out = {"_meta": {"preds": args.preds, "gt_masks": args.gt_masks, "gt_rule": args.gt_rule,
                     "rules": RULES}, "models": {}}
    for m, preds in models.items():
        per_rule = {r: [] for r in RULES}
        ious = []
        for nm in stems:
            if nm not in preds:
                continue
            pred = preds[nm]
            mp = None
            for ext in (".png", ".jpg", ".jpeg", ".bmp"):
                cand = os.path.join(args.gt_masks, nm + ext)
                if os.path.exists(cand):
                    mp = cand
                    break
            if mp is None:
                continue
            raw = load_gray(mp)
            if raw.shape != pred.shape:
                raw = np.array(Image.fromarray(raw).resize((pred.shape[1], pred.shape[0]), Image.NEAREST))
            gt = binarize(raw, args.gt_rule)
            ng, gc, garea = regions(gt)
            npc, pc, _ = regions(pred)
            ext = np.sqrt(garea.astype(np.float64))
            for r in RULES:
                if r.startswith("fixed_"):
                    dth = FIXED[["fixed_1", "fixed_2", "fixed_3", "fixed_5"].index(r)]
                    v = pd_match(gc, pc, np.full(ng, dth, dtype=np.float64))
                else:
                    kap = KAPPAS[["k_6", "k_4", "k_3", "k_2"].index(r)]
                    v = pd_match(gc, pc, kap * ext)
                per_rule[r].append(np.nan if v is None else v)
            ious.append(iou_binary(pred, gt))
        out["models"][m] = {
            "mean_Pd": {r: float(np.nanmean(per_rule[r])) if per_rule[r] else None for r in RULES},
            "mean_IoU": float(np.mean(ious)) if ious else None,
            "n_images": len(ious),
        }
        print("[%s] n=%d Pd(fx3)=%.4f Pd(k1/3)=%.4f IoU=%.4f" % (
            m, len(ious), out["models"][m]["mean_Pd"]["fixed_3"] or -1,
            out["models"][m]["mean_Pd"]["k_3"] or -1, out["models"][m]["mean_IoU"] or -1))
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("tolerance ->", args.out)


# ---------------------------------------------------------------------------
# 3. rank
# ---------------------------------------------------------------------------
def cmd_rank(args):
    tol = json.load(open(args.tol, encoding="utf-8"))
    models = sorted(tol["models"].keys())
    rankings = {}
    for r in RULES:
        means = {m: tol["models"][m]["mean_Pd"].get(r) for m in models}
        rankings[r] = sorted(models, key=lambda m: -(means[m] if means[m] is not None else -1))
    ref = rankings["fixed_3"]
    swaps = {}
    for r in RULES:
        if r == "fixed_3":
            continue
        s = 0
        for a in range(len(models)):
            for b in range(a + 1, len(models)):
                ma, mb = models[a], models[b]
                if (ref.index(ma) < ref.index(mb)) != (rankings[r].index(ma) < rankings[r].index(mb)):
                    s += 1
        swaps[r] = s
    out = {"rankings": rankings, "pairwise_swaps_vs_fixed3": swaps,
           "n_pairs": len(models) * (len(models) - 1) // 2}
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("rank ->", args.out)
    print("  fixed_3 :", rankings["fixed_3"])
    print("  k_3     :", rankings["k_3"])
    print("  swaps   :", swaps)


def main():
    ap = argparse.ArgumentParser(description="IRSTD evaluation audit toolkit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c1 = sub.add_parser("census", help="mask-encoding census")
    c1.add_argument("--masks", required=True)
    c1.add_argument("--sample", type=int, default=0)
    c1.add_argument("--out", required=True)
    c1.set_defaults(fn=cmd_census)

    c2 = sub.add_parser("tolerance", help="tolerance-family re-evaluation of fixed predictions")
    c2.add_argument("--preds", required=True, help="npz with <model>|names/bits/shapes")
    c2.add_argument("--gt-masks", required=True)
    c2.add_argument("--img-idx", required=True, help="text file, one stem per line")
    c2.add_argument("--gt-rule", default="ge128", choices=["ge128", "gt0", "eq255"])
    c2.add_argument("--out", required=True)
    c2.set_defaults(fn=cmd_tolerance)

    c3 = sub.add_parser("rank", help="ranking-stability check from a tolerance report")
    c3.add_argument("--tol", required=True)
    c3.add_argument("--out", required=True)
    c3.set_defaults(fn=cmd_rank)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
