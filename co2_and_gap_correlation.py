# ============================================================================
# Part 3 -- CO2 and the Wet-Dry Gap (mechanism support, NOT proof)
#
# Loads Mauna Loa annual-mean CO2 (NOAA GML) and correlates it with the
# wet-dry gap index. Honest statistics:
#   * raw correlation        -> co-evolution (both trending -> expect high r)
#   * DETRENDED correlation  -> the honest test: do they move together
#                               beyond their shared upward drift?
#   * per-city CO2 correlations (report ALL cities, not just the good ones)
# Plus the combined twin-axis figure for the final poster.
#
# Download the CO2 file once (in Colab):
#   !wget -q https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_mlo.txt
#   (or upload it manually; file starts at 1959)
#
# INPUTS: annual_precip_mm.csv, results_era5_trends.csv (from Part 1)
# ============================================================================

import numpy as np
import pandas as pd
from scipy import stats
import pymannkendall as mk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WET = ["Glasgow", "Dublin"]
DRY = ["Alicante", "Almeria", "Athens", "Madrid", "Murcia",
       "Palermo", "Rome", "Seville", "Tunis"]

# ---------------------------------------------------------------------------
# 1. Load Mauna Loa annual-mean CO2 (tolerant parser: skip #, take year+mean)
# ---------------------------------------------------------------------------
def load_co2(path="co2_annmean_mlo.txt"):
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            try:
                rows.append((int(parts[0]), float(parts[1])))   # year, annual mean ppm
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows, columns=["year", "co2_ppm"])\
             .drop_duplicates("year").set_index("year").sort_index()


def detrend(x, t):
    g = stats.linregress(t, x)
    return x - (g.intercept + g.slope * t)


# ---------------------------------------------------------------------------
# 2. Gap index (same definition as Part 2)
# ---------------------------------------------------------------------------
df = pd.read_csv("annual_precip_mm.csv", index_col=0)
gap = (df[WET].mean(axis=1) - df[DRY].mean(axis=1)).rename("gap")

co2 = load_co2()
print(f"CO2 loaded: {len(co2)} years, {co2.index.min()}-{co2.index.max()}, "
      f"{co2['co2_ppm'].iloc[0]:.1f} -> {co2['co2_ppm'].iloc[-1]:.1f} ppm\n")

m = pd.concat([gap, co2["co2_ppm"]], axis=1).dropna()      # 1959-2025
t = m.index.to_numpy()

# ---------------------------------------------------------------------------
# 3. Correlations (raw AND detrended)
# ---------------------------------------------------------------------------
r_raw, p_raw = stats.pearsonr(m["gap"], m["co2_ppm"])
r_sp,  p_sp  = stats.spearmanr(m["gap"], m["co2_ppm"])
r_det, p_det = stats.pearsonr(detrend(m["gap"], t), detrend(m["co2_ppm"], t))

print("=" * 60)
print("GAP vs CO2  (n = %d, %d-%d)" % (len(m), m.index.min(), m.index.max()))
print("=" * 60)
print(f"  raw Pearson      r = {r_raw:+.3f}   p = {p_raw:.3g}   (expected high: both trend)")
print(f"  raw Spearman     r = {r_sp:+.3f}   p = {p_sp:.3g}")
print(f"  DETRENDED Pearson r = {r_det:+.3f}   p = {p_det:.3g}   <- the honest test")
print("  Interpretation: detrended p < 0.05 means the gap and CO2 move together\n"
      "  BEYOND their shared upward drift. If not significant: co-evolution only.\n"
      "  Either way: 'consistent with' the CO2-driven mechanism, never 'proves'.\n")

# Is CO2 itself trending? (sanity check for the narrative)
mk_co2 = mk.original_test(co2["co2_ppm"].to_numpy())
print(f"CO2 trend: MK p = {mk_co2.p:.2e}, Sen slope = {mk_co2.slope:.3f} ppm/yr "
      f"(clearly trending, as expected)\n")

# ---------------------------------------------------------------------------
# 4. Per-city CO2 correlations (report ALL -- the honest way)
# ---------------------------------------------------------------------------
print("Per-city correlation with CO2 (raw Pearson; beware trend-trend artefact):")
res = pd.read_csv("results_era5_trends.csv")
for c in res["city"]:
    mm = pd.concat([df[c].rename("p"), co2["co2_ppm"]], axis=1).dropna()
    r, p = stats.pearsonr(mm["p"], mm["co2_ppm"])
    print(f"  {c:<12} r = {r:+.2f}   p = {p:.3f}")

# ---------------------------------------------------------------------------
# 5. Combined figure: gap + CO2 on twin axes (the poster centerpiece)
# ---------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(m.index, m["gap"], color="tab:blue", lw=1.2, label="wet-dry gap (mm/yr)")
ax1.set_xlabel("Year"); ax1.set_ylabel("Gap (mm/yr)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(m.index, m["co2_ppm"], color="tab:red", lw=1.4, label="Mauna Loa CO2 (ppm)")
ax2.set_ylabel("CO2 (ppm)", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

fig.suptitle("Wet-dry precipitation gap and atmospheric CO2, 1959-2025\n"
             f"raw r={r_raw:+.2f} (p={p_raw:.2g}) | detrended r={r_det:+.2f} (p={p_det:.2g})")
fig.tight_layout(); fig.savefig("fig_gap_and_co2.png", dpi=150)
print("\nSaved: fig_gap_and_co2.png")
