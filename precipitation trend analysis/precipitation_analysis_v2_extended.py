# ============================================================================
# Precipitation Trend Analysis -- EXTENDED VERSION (v2)
# Same analysis as v1 (validated), now with:
#   * more cities (extended aridity gradient: Bergen..Ouarzazate)
#   * fallback naming: ANY folder containing a *.dat.txt is analysed even if
#     not in CITY_INFO -- its folder name is used as the city name
#   * city_metadata.csv written so you can verify how each file was mapped
#
# IMPORTANT -- scientific integrity rule:
#   The added-city list must be chosen BEFORE looking at their trends.
#   Run this once with all files present and report whatever comes out.
#
# INPUTS: same KNMI files as v1, new cities in folders (e.g. wet/bergen/,
#         dry/agadir/). Download from KNMI Climate Explorer, era5_tp field,
#         monthly, 1950-2026, whole-degree grid boxes as in v1.
#
# Colab: !pip install pymannkendall -q  |  main("/content")
# ============================================================================

import numpy as np
import pandas as pd
import calendar
import glob
import os
from scipy import stats
import pymannkendall as mk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# v1 cities + new additions (folder name -> (city, climate type))
CITY_INFO = {
    # --- wet end (extended) ---
    "glasglow":      ("Glasgow",  "Wet oceanic"),
    "dubin":         ("Dublin",   "Wet oceanic"),
    "bergen":        ("Bergen",   "Very wet oceanic"),
    "cork":          ("Cork",     "Wet oceanic"),
    "bilbao":        ("Bilbao",   "Wet Atlantic"),
    # --- middle of the gradient ---
    "lisbon":        ("Lisbon",   "Atlantic/Med transition"),
    "bordeaux":      ("Bordeaux", "Atlantic temperate"),
    "marseille":     ("Marseille","Mediterranean"),
    "belgrade":      ("Belgrade", "Continental"),
    "rome":          ("Rome",     "Semi-wet Mediterranean"),
    "palermo":       ("Palermo",  "Mediterranean"),
    "madrid":        ("Madrid",   "Semi-arid"),
    "seville":       ("Seville",  "Hot semi-arid"),
    "atehns,greece": ("Athens",   "Mediterranean dry"),
    # --- dry end (extended) ---
    "alicante":      ("Alicante", "Very dry"),
    "almeria":       ("Almeria",  "Near-desert"),
    "murcia":        ("Murcia",   "Near-desert"),
    "tunis":         ("Tunis",    "Semi-arid"),
    "agadir":        ("Agadir",   "Semi-arid coastal"),
    "gabes":         ("Gabes",    "Arid coastal"),
    "ouarzazate":    ("Ouarzazate", "Arid desert"),
}

SPLIT_YEAR = 1987   # early = 1950-1987, late = 1988-2025


def parse_knmi_file(path):
    month_cols = [f"m{m}" for m in range(1, 13)]
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 13:
                continue
            try:
                year = int(parts[0])
                vals = [float(p) for p in parts[1:13]]
            except ValueError:
                continue
            rows.append([year] + vals)
    df = pd.DataFrame(rows, columns=["year"] + month_cols)
    df[month_cols] = df[month_cols].replace(-999.9, np.nan)
    return df.set_index("year").sort_index()


def monthly_mm_to_annual(df_mmd):
    monthly = df_mmd.copy()
    for m in range(1, 13):
        days = monthly.index.to_series().apply(lambda y: calendar.monthrange(y, m)[1])
        monthly[f"m{m}"] = monthly[f"m{m}"] * days
    complete = monthly.notna().all(axis=1)
    annual = monthly[complete].sum(axis=1)
    return monthly, annual


def city_stats(years, annual):
    n = len(annual)
    slope, intercept, r, p_ols, se = stats.linregress(years, annual)
    res = mk.original_test(annual)
    mk_p, mk_tau, mk_z, sen_slope = res.p, res.Tau, res.z, res.slope
    early = annual[years <= SPLIT_YEAR]
    late  = annual[years >  SPLIT_YEAR]
    early_mean, late_mean = early.mean(), late.mean()
    pct_split = (late_mean - early_mean) / early_mean * 100.0
    t_stat, p_ttest = stats.ttest_ind(late, early, equal_var=False)
    r1 = np.corrcoef(annual[:-1], annual[1:])[0, 1] if n > 2 else np.nan
    return {
        "years":            n,
        "baseline_mm_yr":   annual.mean(),
        "early_mean":       early_mean,
        "late_mean":        late_mean,
        "pct_change_split": pct_split,
        "sen_slope_mm_yr":  sen_slope,
        "sen_pct_per_dec":  sen_slope * 10.0 / annual.mean() * 100.0,
        "mk_p":             mk_p,
        "mk_tau":           mk_tau,
        "mk_z":             mk_z,
        "ols_slope":        slope,
        "ols_p":            p_ols,
        "ttest_p":          p_ttest,
        "lag1_r":           r1,
    }


def bh_fdr(pvals, alpha=0.05):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    sig = p[order] <= (np.arange(1, n + 1) / n) * alpha
    if not sig.any():
        return np.zeros(n, bool)
    k = np.where(sig)[0].max()
    out = np.zeros(n, bool); out[order[: k + 1]] = True
    return out


def main(data_root="."):
    files = sorted(glob.glob(os.path.join(data_root, "**", "*.dat.txt"), recursive=True))
    if not files:
        raise SystemExit(f"No *.dat.txt files found under '{data_root}'.")

    results, annual_all, monthly_all, meta = [], {}, [], []

    for path in files:
        folder = os.path.basename(os.path.dirname(path)).lower().strip()
        if folder in CITY_INFO:
            city, climate = CITY_INFO[folder]
        else:
            # fallback: unknown folder -> use its name as the city name
            city, climate = folder.replace("_", " ").title(), "unclassified"
            print(f"NOTE: no metadata for folder '{folder}' -> city named '{city}'")
        meta.append({"folder": folder, "city": city, "climate": climate})

        df_mmd = parse_knmi_file(path)
        dropped = [y for y in df_mmd.index if not df_mmd.loc[y].notna().all()]
        monthly, annual = monthly_mm_to_annual(df_mmd)

        s = city_stats(annual.index.to_numpy(), annual.to_numpy())
        s.update({"city": city, "climate": climate, "dropped_years": dropped})
        results.append(s)

        annual_all[city] = annual
        m = monthly.reset_index().melt(id_vars="year", var_name="month", value_name="mm")
        m["city"] = city
        monthly_all.append(m)
        print(f"  {city:<12} {climate:<22} parsed {s['years']} yrs | "
              f"dropped {dropped} | mean {s['baseline_mm_yr']:.0f} mm/yr")

    pd.DataFrame(meta).to_csv("city_metadata.csv", index=False)

    res = pd.DataFrame(results)
    res = res[["city", "climate", "years", "baseline_mm_yr", "early_mean", "late_mean",
               "pct_change_split", "sen_slope_mm_yr", "sen_pct_per_dec",
               "mk_p", "mk_tau", "mk_z", "ols_slope", "ols_p", "ttest_p", "lag1_r"]]
    res["mk_sig_p05"] = res["mk_p"] < 0.05
    res["fdr_sig"] = bh_fdr(res["mk_p"].to_numpy())

    res.to_csv("results_era5_trends.csv", index=False, float_format="%.4g")

    rounded = res.copy()
    for c in ["baseline_mm_yr", "early_mean", "late_mean", "sen_slope_mm_yr"]:
        rounded[c] = rounded[c].round(0)
    for c in ["pct_change_split", "sen_pct_per_dec", "mk_tau", "mk_z", "lag1_r", "ols_slope"]:
        rounded[c] = rounded[c].round(2)
    for c in ["mk_p", "ols_p", "ttest_p"]:
        rounded[c] = rounded[c].round(4)
    rounded.to_csv("results_era5_trends_rounded.csv", index=False)

    pd.DataFrame(annual_all).to_csv("annual_precip_mm.csv")
    pd.concat(monthly_all).to_csv("monthly_precip_mm.csv", index=False)

    grad = res[["city", "baseline_mm_yr", "pct_change_split"]].copy()
    g = stats.linregress(grad["baseline_mm_yr"], grad["pct_change_split"])
    grad["fit"] = g.intercept + g.slope * grad["baseline_mm_yr"]
    grad.to_csv("aridity_gradient.csv", index=False, float_format="%.4g")

    show = res.copy()
    for c in ["baseline_mm_yr", "early_mean", "late_mean", "sen_slope_mm_yr"]:
        show[c] = show[c].round(0)
    for c in ["pct_change_split", "sen_pct_per_dec", "mk_tau", "mk_z", "lag1_r", "ols_slope"]:
        show[c] = show[c].round(2)
    for c in ["mk_p", "ols_p", "ttest_p"]:
        show[c] = show[c].round(4)
    show["mk_sig_p05"] = show["mk_sig_p05"].map({True: "YES", False: "no"})
    show["fdr_sig"] = show["fdr_sig"].map({True: "YES", False: "no"})
    pd.set_option("display.width", 220)

    print("\n" + "=" * 104)
    print(f"CITY TREND RESULTS -- extended set (n = {len(res)} sites, 1950-2025)")
    print("=" * 104)
    print(show.to_string(index=False))

    print("\n" + "=" * 104)
    print(f"ARIDITY GRADIENT  (% change vs baseline rainfall, n = {len(grad)})")
    print("=" * 104)
    print(f"  slope = {g.slope:+.4f} % per mm/yr baseline     intercept = {g.intercept:+.2f}")
    print(f"  R^2   = {g.rvalue**2:.3f}    p = {g.pvalue:.4g}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    wet_cities = [c for c in res["city"] if res.loc[res["city"]==c, "baseline_mm_yr"].iloc[0] > 1100]
    dry_cities = [c for c in res["city"] if res.loc[res["city"]==c, "baseline_mm_yr"].iloc[0] < 450]
    for ax, cities, title in [(axes[0], wet_cities[:4], "Wettest sites"),
                              (axes[1], dry_cities[:4], "Driest sites")]:
        for c in cities:
            a = annual_all[c]
            years = a.index.to_numpy()
            ax.plot(years, a, lw=0.8, alpha=0.6, label=c)
        ax.set_title(title); ax.set_xlabel("Year"); ax.set_ylabel("mm")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("Annual precipitation 1950-2025 (extended set)")
    fig.tight_layout(); fig.savefig("fig_time_series.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.scatter(grad["baseline_mm_yr"], grad["pct_change_split"], s=50, zorder=3)
    for _, r in grad.iterrows():
        ax2.annotate(r["city"], (r["baseline_mm_yr"], r["pct_change_split"]),
                     xytext=(3, 3), textcoords="offset points", fontsize=7)
    xs = np.linspace(grad["baseline_mm_yr"].min() - 30, grad["baseline_mm_yr"].max() + 30, 50)
    ax2.plot(xs, g.intercept + g.slope * xs, "r-",
             label=f"R$^2$={g.rvalue**2:.2f}, p={g.pvalue:.3g}, n={len(grad)}")
    ax2.axhline(0, color="k", lw=0.6, ls=":")
    ax2.set_xlabel("Baseline annual precipitation (mm/yr)")
    ax2.set_ylabel("Precipitation change, late vs early period (%)")
    ax2.set_title("Aridity gradient (extended)")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig2.tight_layout(); fig2.savefig("fig_aridity_gradient.png", dpi=150)
    print("Figures saved: fig_time_series.png, fig_aridity_gradient.png")
    print("CSVs: results_era5_trends.csv, annual_precip_mm.csv, aridity_gradient.csv, city_metadata.csv")
    return res, grad, g


if __name__ == "__main__":
    res, grad, g = main()
