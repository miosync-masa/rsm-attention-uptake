#!/usr/bin/env python3
"""
09_temporal_diagnostics.py
========================================================================
時間構造 interaction の (A)本物 / (B)鏡像アーティファクト / (C)録音密度交絡
を切り分ける診断。特に burstiness(+0.693, 8/8)が「TVの流行語=burst dosing が
即production」仮説の実体なのかを検証する。

診断A — 独立性: span/persistence/burstiness を同時投入し、どれが独立に効くか
診断B — 鏡像チェック: 3変数の相互相関(高相関なら同一構造の別表現)
診断C — 録音密度交絡: burstiness vs (観測ビン数/ファイル数/総トークン)の相関、
         および録音密度を統制しても burstiness×freq が peak を予測するか

入力: {lang}_temporal_structure.csv (07が出力)、{lang}_uptake.csv
使い方:
  python 09_temporal_diagnostics.py --output_dir ./output/v3/ --results_dir ./results/ \
     --languages English English-UK Japanese Korean Mandarin Russian Spanish Indonesian
========================================================================
"""
import argparse, json, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import binomtest

TVARS = ["span_bins", "dispersion_gini", "dispersion_evenness", "persistence", "burstiness"]

def z(x):
    x=np.asarray(x,float); s=x.std(ddof=0); return (x-x.mean())/s if s>0 else x*0

def dl_meta(betas, ses):
    betas, ses = np.asarray(betas,float), np.asarray(ses,float)
    ok = ~np.isnan(betas) & ~np.isnan(ses) & (ses>0)
    betas, ses = betas[ok], ses[ok]
    if len(betas)<2: return None
    w=1/ses**2; fixed=(w*betas).sum()/w.sum(); Q=(w*(betas-fixed)**2).sum(); k=len(betas)
    C=w.sum()-(w**2).sum()/w.sum(); tau2=max(0,(Q-(k-1))/C) if C>0 else 0
    wr=1/(ses**2+tau2); pooled=(wr*betas).sum()/wr.sum(); se=np.sqrt(1/wr.sum())
    zv=pooled/se; I2=max(0,(Q-(k-1))/Q)*100 if Q>0 else 0
    return dict(pooled=pooled,z=zv,p=2*stats.norm.sf(abs(zv)),I2=I2,
                signs=int((betas>0).sum()),k=len(betas),binom_p=binomtest(int((betas>0).sum()),len(betas)).pvalue)

def se_bp(b,p):
    if p is None or np.isnan(p) or p<=0: return np.nan
    zc=stats.norm.isf(p/2); return abs(b)/zc if zc>0 else np.nan

def ols_p(X, y):
    """returns betas, pvals (X already has intercept col)."""
    beta,*_=np.linalg.lstsq(X,y,rcond=None); resid=y-X@beta; n,k=X.shape; dof=n-k
    if dof<=0 or np.linalg.cond(X.T@X)>1e12: return None,None
    mse=(resid@resid)/dof; se=np.sqrt(np.diag(mse*np.linalg.inv(X.T@X)))
    t=beta/se; p=2*stats.t.sf(np.abs(t),dof); return beta,p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="./output/v3/")
    ap.add_argument("--results_dir", default="./results/")
    ap.add_argument("--languages", nargs="+", required=True)
    args=ap.parse_args()
    od=Path(args.output_dir); rd=Path(args.results_dir)

    # ---------- 診断B: 3変数の相互相関(鏡像チェック) ----------
    print("="*78)
    print("診断B — span/persistence/burstiness 相互相関 (|r|高=同一構造の別表現)")
    print("="*78)
    corr_acc={("span_bins","persistence"):[], ("span_bins","burstiness"):[],
              ("persistence","burstiness"):[]}
    for lang in args.languages:
        f=od/f"{lang}_temporal_structure.csv"
        if not f.exists(): continue
        t=pd.read_csv(f)
        for (a,b) in corr_acc:
            v=t[[a,b]].dropna()
            if len(v)>3 and v[a].std()>0 and v[b].std()>0:
                corr_acc[(a,b)].append(np.corrcoef(v[a],v[b])[0,1])
    for (a,b),rs in corr_acc.items():
        if rs:
            print(f"  {a:<14} ~ {b:<12} mean r = {np.mean(rs):+.3f}  (range {min(rs):+.2f}..{max(rs):+.2f}, {len(rs)} langs)")
    print("  → |mean r|>0.7 なら鏡像疑い。burstiness が他と低相関なら独立な量。")

    # ---------- 診断C: burstiness vs 録音密度 ----------
    print("\n"+"="*78)
    print("診断C — burstiness vs 録音密度指標 (高相関=録音アーティファクト疑い)")
    print("="*78)
    dens_r={"n_active_bins":[], "total_count":[], "span_bins":[]}
    for lang in args.languages:
        f=od/f"{lang}_temporal_structure.csv"
        if not f.exists(): continue
        t=pd.read_csv(f)
        for col in dens_r:
            if col in t.columns and "burstiness" in t.columns:
                v=t[[col,"burstiness"]].dropna()
                if len(v)>3 and v[col].std()>0 and v["burstiness"].std()>0:
                    dens_r[col].append(np.corrcoef(v[col],v["burstiness"])[0,1])
    for col,rs in dens_r.items():
        if rs:
            print(f"  burstiness ~ {col:<16} mean r = {np.mean(rs):+.3f}  ({len(rs)} langs)")
    print("  → burstiness が n_active_bins と強い負相関なのは定義上当然(active bin 少→burst)。")
    print("    重要なのは診断Cの統制後検定(下)で burstiness×freq が生き残るか。")

    # ---------- 診断A+C: 録音密度統制 + 3変数同時投入 ----------
    # peak ~ freq + bursti + bursti*freq + n_active_bins(統制)   をメタ
    print("\n"+"="*78)
    print("診断A+C — burstiness×freq の interaction、録音密度(n_active_bins)統制後")
    print("="*78)
    betas_ctrl, ses_ctrl = [], []
    # 同時投入: peak ~ freq + span + persist + bursti + 各*freq → burstiが独立に効くか
    betas_joint, ses_joint = [], []
    for lang in args.languages:
        tf=od/f"{lang}_temporal_structure.csv"; uf=rd/f"{lang}_uptake.csv"
        if not (tf.exists() and uf.exists()): continue
        t=pd.read_csv(tf); u=pd.read_csv(uf)
        m=t.merge(u,on="cue_subtype",how="inner")
        peak_col="peak_rate_per_1k"
        lf_col="log_caregiver_count" if "log_caregiver_count" in m.columns else "log_freq"
        need=["burstiness",lf_col,peak_col,"n_active_bins"]
        sub=m.dropna(subset=[c for c in need if c in m.columns])
        if len(sub)<8: continue
        zf=z(sub[lf_col]); zb=z(sub[lf_col]); zy=z(sub[peak_col])
        zburst=z(sub["burstiness"]); zdens=z(sub["n_active_bins"])
        # 録音密度統制モデル: y ~ 1 + freq + burst + burst*freq + dens
        X=np.column_stack([np.ones_like(zf), zf, zburst, z(zburst*zf), zdens])
        beta,p=ols_p(X,zy)
        if beta is not None:
            betas_ctrl.append(beta[3]); ses_ctrl.append(se_bp(beta[3],p[3]))
        # 3変数同時(span/persist/burst)×freq
        if all(c in sub.columns for c in ["span_bins","persistence"]):
            zsp=z(sub["span_bins"]); zpe=z(sub["persistence"])
            Xj=np.column_stack([np.ones_like(zf), zf,
                                zsp, z(zsp*zf),
                                zpe, z(zpe*zf),
                                zburst, z(zburst*zf)])
            bj,pj=ols_p(Xj,zy)
            if bj is not None:
                betas_joint.append(bj[7]); ses_joint.append(se_bp(bj[7],pj[7]))  # burst interaction idx
    mc=dl_meta(betas_ctrl,ses_ctrl)
    if mc:
        print(f"  [録音密度統制] burstiness×freq pooled β={mc['pooled']:+.3f} "
              f"z={mc['z']:.2f} p={mc['p']:.4f} I²={mc['I2']:.1f}% sign={mc['signs']}/{mc['k']}")
        print(f"     → 統制後も有意で符号維持なら、録音アーティファクト(C)ではない。")
    mj=dl_meta(betas_joint,ses_joint)
    if mj:
        print(f"  [3変数同時投入] burstiness×freq pooled β={mj['pooled']:+.3f} "
              f"z={mj['z']:.2f} p={mj['p']:.4f} I²={mj['I2']:.1f}% sign={mj['signs']}/{mj['k']}")
        print(f"     → span/persistence と同時でも burstiness が残れば、独立な担い手(A)。")

    print("\n"+"="*78)
    print("判定ガイド:")
    print("  (A)本物    : 診断Cの統制後 & 診断Aの同時投入で burstiness×freq が有意維持")
    print("  (B)鏡像    : 診断Bで3変数が相互に|r|>0.7、かつ同時投入で1つしか残らない")
    print("  (C)交絡    : 録音密度統制で burstiness×freq が消える")
    print("="*78)

if __name__=="__main__":
    main()
