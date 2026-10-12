# IRSTD Evaluation Audit

A CPU-only reference implementation of the **three-check audit protocol** for
infrared small target detection (IRSTD) evaluation, accompanying the paper
*"Two Hidden Traps in Infrared Small Target Detection Evaluation: Mask Encoding
and Tolerance Scale Drift"*.

The protocol targets two unstated conventions in the community's Pd/Fa
evaluation stack:

1. **Mask decoding** — stored masks support incompatible readings (`>0`,
   `>=128`, `==255`); on JPEG-compressed or class-label masks the readings
   recover different amounts of foreground (in the worst documented cases:
   3.3x inflation, or *zero* foreground).
2. **Tolerance scale drift** — the fixed 3-pixel centroid tolerance is
   size-blind; under a scale-normalized family `tau = kappa * sqrt(area)`
   published-model leaderboards reorder.

## Install

Python >= 3.8, three dependencies:

```bash
pip install numpy Pillow scipy
```

## Usage

```bash
# 1) mask-encoding census of a dataset (e.g. SIRST-UAVB masks)
python audit.py census --masks /path/to/masks --out census.json

# 2) tolerance-family re-evaluation of fixed predictions
python audit.py tolerance --preds preds.npz --gt-masks /path/to/masks \
                          --img-idx test_list.txt --out tol.json

# 3) ranking stability across models
python audit.py rank --tol tol.json --out rank.json
```

`--img-idx` is a text file with one image stem per line (no extension).
Ground truth is decoded with `>=128` by default; change with `--gt-rule`.

## Prediction format (`preds.npz`)

For each model, three parallel arrays:

```
"<model>|names"   -> array of image stems (strings)
"<model>|bits"    -> object array of np.packbits(pred.ravel().astype(uint8))
"<model>|shapes"  -> int array of [h, w] per image
```

Predictions are binary masks at the **original image resolution** (models that
infer at a reduced resolution are upsampled back to native size with
nearest-neighbor before packing).

## Reproducing the paper's numbers

The paper's supplementary package contains all raw metric records and the
driver scripts that produced every table and figure (six-dataset census and
the seven-dataset extension, controlled decode-rule experiments, published-model
leaderboard re-evaluation, official-release cross-checks).

## Notes

- The toolkit performs **evaluation-side manipulations only**; it never trains
  or modifies models.
- Matching semantics follow the community `PD_FA` protocol (8-connected
  components, greedy 1-to-1 centroid matching, matched prediction removed).
- `census` reports all three decode rules because stored-mask conventions
  differ across codebases; datasets storing class-label maps or JPEG-compressed
  masks will disagree strongly between rules.

## Citation

If you use this toolkit, please cite:

```bibtex
@article{twohiddentraps,
  title  = {Two Hidden Traps in Infrared Small Target Detection Evaluation:
            Mask Encoding and Tolerance Scale Drift},
  author = {To be completed},
  year   = {2026},
  note   = {Manuscript under review; citation updated upon publication}
}
```

## License

MIT — see [LICENSE](LICENSE).
