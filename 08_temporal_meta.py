#!/usr/bin/env python3
"""
08_temporal_meta.py
========================================================================
07_temporal_structure.py が各言語に吐いた {lang}_temporal_interaction.json
を束ねて、時間構造変数ごとに DerSimonian-Laird ランダム効果メタ解析を行う。

raw(時間構造×freq) と residualized(freq直交化した時間構造×freq) の両方を
プールし、「frequency と独立な時間構造が stabilization を予測するか」を
8サンプル横断で判定する。

使い方:
  python 08_temporal_meta.py --output_dir ./output/v3/ \
     --languages English English-UK Japanese Korean Mandarin Russian Spanish Indonesian
========================================================================
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.stats import binomtest


def dl_meta(betas, ses):
    betas, ses = np.asarray(betas, float), np.asarray(ses, float)
    ok = ~np.isnan(betas) & ~np.isnan(ses) & (ses > 0)
    betas, ses = betas[ok], ses[ok]
    if len(betas) < 2:
        return None
    w = 1 / ses**2
    fixed = (w * betas).sum() / w.sum()
    Q = (w * (betas - fixed)**2).sum()
    k = len(betas)
    C = w.sum() - (w**2).sum() / w.sum()
    tau2 = max(0, (Q - (k - 1)) / C) if C > 0 else 0
    wr = 1 / (ses**2 + tau2)
    pooled = (wr * betas).sum() / wr.sum()
    se_p = np.sqrt(1 / wr.sum())
    zval = pooled / se_p
    pval = 2 * stats.norm.sf(abs(zval))
    I2 = max(0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0
    signs = int((betas > 0).sum())
    bp = binomtest(signs, len(betas)).pvalue
    return dict(pooled=pooled, ci_lo=pooled - 1.96 * se_p, ci_hi=pooled + 1.96 * se_p,
                z=zval, p=pval, I2=I2, k=k, signs=signs, binom_p=bp)


def se_from_beta_p(beta, p):
    if p is None or np.isnan(p) or p <= 0:
        return np.nan
    zc = stats.norm.isf(p / 2)
    return abs(beta) / zc if zc > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="./output/v3/")
    ap.add_argument("--languages", nargs="+", required=True)
    args = ap.parse_args()
    od = Path(args.output_dir)

    TVARS = ["span_bins", "dispersion_gini", "dispersion_evenness",
             "persistence", "burstiness"]

    # collect
    data = {}
    for lang in args.languages:
        f = od / f"{lang}_temporal_interaction.json"
        if not f.exists():
            print(f"[skip] {f} なし")
            continue
        data[lang] = json.load(open(f))["interaction"]

    print("=" * 78)
    print("TEMPORAL STRUCTURE × FREQUENCY — cross-linguistic meta-analysis")
    print("peak production (stabilization) を outcome とする interaction の pooled β")
    print("=" * 78)

    for kind, key_b, key_p in [("RAW (temporal×freq)", "raw_beta", "raw_p"),
                               ("RESIDUALIZED (freq⊥temporal×freq)", "resid_beta", "resid_p")]:
        print(f"\n### {kind}")
        print(f"{'temporal var':<22}{'pooled β':>10}{'95% CI':>20}{'z':>7}{'p':>9}{'I²':>7}{'sign':>7}")
        for tv in TVARS:
            betas, ses = [], []
            for lang, res in data.items():
                if tv in res:
                    b = res[tv][key_b]
                    p = res[tv][key_p]
                    betas.append(b)
                    ses.append(se_from_beta_p(b, p))
            m = dl_meta(betas, ses)
            if m is None:
                print(f"{tv:<22}  (不足)")
                continue
            ci = f"[{m['ci_lo']:+.3f},{m['ci_hi']:+.3f}]"
            star = "***" if m['p'] < .001 else "**" if m['p'] < .01 else "*" if m['p'] < .05 else ""
            print(f"{tv:<22}{m['pooled']:>+10.3f}{ci:>20}{m['z']:>7.2f}"
                  f"{m['p']:>9.4f}{m['I2']:>6.1f}%{m['signs']:>4}/{m['k']} {star}")

    print("\n読み方:")
    print("  RESIDUALIZED で pooled β が有意(p<.05, 符号も揃う)な時間構造変数があれば、")
    print("  それは frequency と直交した『第三の量』として stabilization を駆動している。")
    print("  → 『愛してる事故』回避成功。frequency でも単純 repetition でもない、")
    print("     時間構造(span/dispersion/persistence/burstiness)が RSM の本体。")


if __name__ == "__main__":
    main()
