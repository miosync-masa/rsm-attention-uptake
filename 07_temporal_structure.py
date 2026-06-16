#!/usr/bin/env python3
"""
07_temporal_structure.py
========================================================================
"愛してる事故" 回避エンジン — frequency / repetition / temporal-structure の分離

問題:
  total frequency は時間を畳んだスカラー。「3ヶ月で100回(burst)」と
  「12ヶ月で100回(sustained)」を区別できない。host-diversity 由来の
  S_repetition は frequency と r≈0.88 で被る(= frequency の服を着た repetition)。

このスクリプトがやること:
  caregiver の各 cue を「対象児月齢ビン」で時間分解し、frequency とは
  概念的に独立な時間構造変数を作る:

    SPAN        : cue が caregiver 発話に現れた月齢の広がり (last_bin - first_bin)
    DISPERSION  : ビン間出現の均等さ。burst(一点集中)=低, sustained(均等)=高
                  → 1 - Gini(ビン別カウント) と entropy ベースの2種を出す
    PERSISTENCE : cue が出現したビン数 / 観測対象ビン総数 (presence ratio)
    BURSTINESS  : (σ_iti - μ_iti)/(σ_iti + μ_iti)  出現間隔の Fano 的指標
                  +1=burst, 0=ポアソン的, -1=規則的(=持続)

  これらを frequency に対して residualize し、「frequency と直交した
  時間構造」だけを取り出して、child の peak production(=stabilization)を
  予測できるか H4 形式の interaction で検定する。

  もし residualized temporal structure × frequency が peak を予測すれば:
    → frequency でも単純 repetition でもない「第三の量(時間構造)」が
      productive stabilization を駆動している = RSM の核心の深掘り実証。

入力:
  {language}_tokens_tagged.csv   (02_extract_cues_v2.py の出力)
  必須カラム: cue_subtype, is_cue_token, speaker_role, age_months
             (無い場合の代替カラム名も自動探索する)

出力:
  {language}_temporal_structure.csv   cueごとの span/dispersion/persistence/burstiness + freq
  コンソールに interaction 検定結果

使い方:
  python 07_temporal_structure.py \
      --tagged_csv ./output/v2/Japanese_tokens_tagged.csv \
      --uptake_csv ./results/Japanese_uptake.csv \
      --language Japanese \
      --output_dir ./output/v3/ \
      --bin_width 3

  # 全言語ループ:
  for lang in English English-UK Japanese Korean Mandarin Russian Spanish Indonesian; do
    python 07_temporal_structure.py \
      --tagged_csv ./output/v2/${lang}_tokens_tagged.csv \
      --uptake_csv ./results/${lang}_uptake.csv \
      --language ${lang} --output_dir ./output/v3/ --bin_width 3
  done
========================================================================
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# カラム名の柔軟解決(パイプラインのバージョン差を吸収)
# ----------------------------------------------------------------------
def find_col(df, candidates, required=True, what=""):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        sys.exit(f"[ERROR] 必要なカラムが見つかりません({what}): {candidates}\n"
                 f"        実在カラム: {list(df.columns)}")
    return None


# ----------------------------------------------------------------------
# 時間構造メトリクス
# ----------------------------------------------------------------------
def gini(x):
    """Gini係数 (0=完全均等, 1=完全集中)。ビン別カウントの偏りに使う。"""
    x = np.asarray(x, float)
    x = x[x >= 0]
    if x.sum() == 0 or len(x) == 0:
        return np.nan
    xs = np.sort(x)
    n = len(xs)
    cum = np.cumsum(xs)
    # Gini = (2*Σ i*x_i)/(n*Σx) - (n+1)/n
    return (2.0 * np.sum((np.arange(1, n + 1)) * xs) / (n * cum[-1])) - (n + 1.0) / n


def shannon_evenness(counts):
    """Shannon evenness J = H / log(k)。1=完全均等(sustained), 0=一点集中(burst)。"""
    c = np.asarray(counts, float)
    c = c[c > 0]
    k = len(c)
    if k <= 1:
        return 0.0
    p = c / c.sum()
    H = -np.sum(p * np.log(p))
    return float(H / np.log(k))


def burstiness(active_bins):
    """
    出現ビン列(ソート済み)の間隔(ITI)から Burstiness 係数を計算。
    B = (σ - μ)/(σ + μ).  +1=bursty, 0=Poisson, -1=regular(sustained).
    ビンが2個未満なら NaN。
    """
    b = np.sort(np.asarray(active_bins, float))
    if len(b) < 3:
        return np.nan
    iti = np.diff(b)
    mu, sd = iti.mean(), iti.std(ddof=0)
    if (sd + mu) == 0:
        return np.nan
    return float((sd - mu) / (sd + mu))


def compute_temporal_metrics(care_cue_df, bin_width, global_bins):
    """
    caregiver の1 cue分の DataFrame(age_bin 列付き)から時間構造指標を返す。
    global_bins: 全 caregiver 発話が観測された binの集合(presence ratio の分母)。
    """
    by_bin = care_cue_df.groupby("age_bin").size()
    active = sorted(by_bin.index.tolist())
    total = int(by_bin.sum())

    if len(active) == 0:
        return dict(total_count=0, n_active_bins=0, span_bins=np.nan,
                    dispersion_gini=np.nan, dispersion_evenness=np.nan,
                    persistence=np.nan, burstiness=np.nan)

    span = (max(active) - min(active)) / bin_width + 1  # ビン数で表現
    # span窓内の全ビンに対するカウントベクトル(欠測=0)を作って分散指標を計算
    lo, hi = min(active), max(active)
    full_range = list(range(lo, hi + bin_width, bin_width))
    counts_vec = [int(by_bin.get(b, 0)) for b in full_range]

    g = gini(counts_vec)
    disp_gini = (1.0 - g) if not np.isnan(g) else np.nan  # 1=sustained, 0=burst
    evenness = shannon_evenness(counts_vec)               # 1=sustained, 0=burst
    persistence = len(active) / max(1, len(global_bins))   # 観測窓に占める出現ビン割合
    burst = burstiness(active)

    return dict(
        total_count=total,
        n_active_bins=len(active),
        span_bins=float(span),
        dispersion_gini=float(disp_gini) if not np.isnan(disp_gini) else np.nan,
        dispersion_evenness=float(evenness),
        persistence=float(persistence),
        burstiness=float(burst) if not np.isnan(burst) else np.nan,
    )


# ----------------------------------------------------------------------
# 統計: residualize と interaction 検定
# ----------------------------------------------------------------------
def z(x):
    x = np.asarray(x, float)
    s = x.std(ddof=0)
    return (x - x.mean()) / s if s > 0 else x * 0.0


def residualize(target, covariate):
    """target から covariate(線形)を回帰除去した残差を返す(両者 z 化)。"""
    t, c = z(target), z(covariate)
    b = np.polyfit(c, t, 1)
    return t - (b[0] * c + b[1])


def interaction_test(predictor, logfreq, outcome):
    """
    outcome ~ logfreq + predictor + predictor*logfreq (全て z 化)。
    interaction の標準化β, p, ΔR²(interaction) を返す。
    """
    from scipy import stats
    zp, zf, zy = z(predictor), z(logfreq), z(outcome)
    inter = z(zp * zf)
    # 分散ゼロ(定数列)や完全共線のガード
    if zp.std() == 0 or zf.std() == 0 or zy.std() == 0:
        return np.nan, np.nan, np.nan
    X = np.column_stack([np.ones_like(zf), zf, zp, inter])
    beta, *_ = np.linalg.lstsq(X, zy, rcond=None)
    resid = zy - X @ beta
    n, k = X.shape
    dof = n - k
    if dof <= 0:
        return np.nan, np.nan, np.nan
    mse = (resid @ resid) / dof
    XtX = X.T @ X
    # 特異/悪条件ならNaNを返す(pinvでも可だがSEが不安定なので除外)
    if np.linalg.cond(XtX) > 1e12:
        return np.nan, np.nan, np.nan
    se = np.sqrt(np.diag(mse * np.linalg.inv(XtX)))
    t = beta[3] / se[3]
    p = 2 * stats.t.sf(abs(t), dof)
    ss = ((zy - zy.mean()) ** 2).sum()
    r2_full = 1 - (resid @ resid) / ss
    X3 = X[:, :3]
    b3, *_ = np.linalg.lstsq(X3, zy, rcond=None)
    r3 = zy - X3 @ b3
    r2_3 = 1 - (r3 @ r3) / ss
    return beta[3], p, r2_full - r2_3


# ----------------------------------------------------------------------
# メイン
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tagged_csv", required=True)
    ap.add_argument("--uptake_csv", required=True,
                    help="{lang}_uptake.csv (peak_rate_per_1k, log_caregiver_count を含む)")
    ap.add_argument("--language", required=True)
    ap.add_argument("--output_dir", default="./output/v3/")
    ap.add_argument("--bin_width", type=int, default=3)
    ap.add_argument("--min_count", type=int, default=5,
                    help="この総数未満の cue は除外(時間構造が不安定なため)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 読み込み ---
    df = pd.read_csv(args.tagged_csv, low_memory=False)
    col_cue = find_col(df, ["cue_subtype"], what="cue_subtype")
    col_role = find_col(df, ["speaker_role", "role"], what="speaker_role")
    col_age = find_col(df, ["age_months", "age", "target_child_age_months"],
                       what="age_months")
    col_iscue = find_col(df, ["is_cue_token", "is_cue"], required=False)
    col_file = find_col(df, ["file", "filename", "transcript", "source_file"],
                        required=False, what="file")

    df["age_months"] = pd.to_numeric(df[col_age], errors="coerce")

    # --- 月齢の継承(CHIDESの仕様: age は対象児にしか付かない) ---
    # 同一録音ファイル = 対象児がある月齢のときのセッション。
    # ファイル内の child 行の age を、そのファイルの全行(caregiver含む)へ broadcast。
    if col_file is not None:
        is_child = df[col_role].astype(str).str.lower().str.contains(
            "child|chi|target", na=False)
        # file -> 代表児月齢(そのファイルの child age の中央値)
        file_age = (df[is_child].dropna(subset=["age_months"])
                    .groupby(col_file)["age_months"].median())
        inherited = df[col_file].map(file_age)
        # 元々ageがある行はそのまま、無い行はファイル継承で埋める
        df["age_months"] = df["age_months"].fillna(inherited)
        n_inherited = int(inherited.notna().sum())
        print(f"[{args.language}] 月齢をファイル経由で継承: {n_inherited} 行に付与 "
              f"({file_age.notna().sum()} ファイル, "
              f"{file_age.min():.0f}–{file_age.max():.0f}mo)")
    else:
        print(f"[{args.language}] ⚠ file カラムが無く月齢継承不可。"
              f"caregiver age が NaN なら span/dispersion は測れません。")

    # cue トークン(bool)・caregiver・月齢あり に絞る
    if col_iscue is not None:
        df = df[df[col_iscue] == True]  # noqa: E712  bool 列なので直接比較
    care = df[df[col_role].astype(str).str.lower().str.contains(
        "care|mot|fat|adult|parent", na=False)].copy()
    care = care[care["age_months"].notna()].copy()
    care["age_bin"] = (care["age_months"] // args.bin_width * args.bin_width).astype(int)

    if len(care) == 0:
        sys.exit(f"[ERROR] caregiver の cue トークンが0件。role値: "
                 f"{df[col_role].unique()[:10]}  "
                 f"(月齢継承後も caregiver age が全て NaN の可能性)")

    global_bins = sorted(care["age_bin"].unique().tolist())
    print(f"[{args.language}] caregiver cue tokens: {len(care)}, "
          f"観測ビン数: {len(global_bins)} ({min(global_bins)}–{max(global_bins)}mo)")

    # --- cue ごとに時間構造を計算 ---
    rows = []
    for cue, g in care.groupby(col_cue):
        if len(g) < args.min_count:
            continue
        m = compute_temporal_metrics(g, args.bin_width, global_bins)
        m["cue_subtype"] = cue
        m["log_freq"] = np.log1p(m["total_count"])
        rows.append(m)

    tdf = pd.DataFrame(rows)
    if len(tdf) == 0:
        sys.exit("[ERROR] min_count を満たす cue がありません。")

    out_csv = out_dir / f"{args.language}_temporal_structure.csv"
    tdf.to_csv(out_csv, index=False)
    print(f"  → 保存: {out_csv}  ({len(tdf)} cues)")

    # --- frequency との相関(各時間構造が freq からどれだけ独立か) ---
    print("\n  時間構造 vs log_freq の相関 (低いほど frequency と独立):")
    for col in ["span_bins", "dispersion_gini", "dispersion_evenness",
                "persistence", "burstiness"]:
        v = tdf[[col, "log_freq"]].dropna()
        if len(v) > 3 and v[col].std() > 0 and v["log_freq"].std() > 0:
            r = np.corrcoef(v[col], v["log_freq"])[0, 1]
            print(f"    {col:<22} r={r:+.3f}  (独立分散 {(1-r**2)*100:4.1f}%)")
        else:
            print(f"    {col:<22} r=  n/a  (分散不足/欠測)")

    # --- uptake と結合して interaction 検定 ---
    up = pd.read_csv(args.uptake_csv)
    col_peak = find_col(up, ["peak_rate_per_1k", "peak_rate"], what="peak")
    col_lf = find_col(up, ["log_caregiver_count"], required=False)
    merged = tdf.merge(up, on="cue_subtype", how="inner")
    if col_lf is None:
        merged["log_caregiver_count"] = merged["log_freq"]
        col_lf = "log_caregiver_count"
    merged = merged.dropna(subset=[col_peak, col_lf])

    print(f"\n  === interaction 検定 (outcome = {col_peak}, n={len(merged)}) ===")
    print(f"  {'temporal var':<24}{'raw β_int':>10}{'p':>8}   {'resid β_int':>12}{'p':>8}")
    results = {}
    for col in ["span_bins", "dispersion_gini", "dispersion_evenness",
                "persistence", "burstiness"]:
        sub = merged[[col, col_lf, col_peak]].dropna()
        if len(sub) < 6:
            print(f"  {col:<24}  (n<6, skip)")
            continue
        # raw: その時間構造 × freq
        b_raw, p_raw, _ = interaction_test(sub[col].values, sub[col_lf].values,
                                           sub[col_peak].values)
        # residualized: freq を抜いた時間構造 × freq
        resid = residualize(sub[col].values, sub[col_lf].values)
        b_res, p_res, _ = interaction_test(resid, sub[col_lf].values,
                                           sub[col_peak].values)
        star_raw = "*" if p_raw < .05 else " "
        star_res = "*" if p_res < .05 else " "
        print(f"  {col:<24}{b_raw:>+10.3f}{p_raw:>8.3f}{star_raw} "
              f"{b_res:>+12.3f}{p_res:>8.3f}{star_res}")
        results[col] = dict(raw_beta=b_raw, raw_p=p_raw,
                            resid_beta=b_res, resid_p=p_res, n=len(sub))

    # サマリ JSON
    import json
    json.dump({"language": args.language, "n_cues": len(tdf),
               "interaction": results},
              open(out_dir / f"{args.language}_temporal_interaction.json", "w"),
              indent=2, default=float)
    print(f"\n  → interaction結果: {out_dir / f'{args.language}_temporal_interaction.json'}")
    print(f"\n  読み方: 'resid β_int' が有意(*)なら、frequency と直交した時間構造が")
    print(f"          stabilization を駆動 = frequency/repetition と別物の第三の量。")


if __name__ == "__main__":
    main()
