# Climate Change Signals: Arctic Sea Ice and Global Precipitation Trends
This project consists of two parts Arctic Sea-Ice Decline and Atmospheric CO₂ and Precipitation Trend Analysis Across a European Aridity Gradient

# Arctic Sea-Ice Decline and Atmospheric CO₂

## Project overview

This project studies the decline of Arctic sea ice in September. It also examines the relationship between atmospheric CO₂ concentration and September sea-ice extent.

The project uses two analysis routes:

1. **Time trend:** How has September Arctic sea ice changed over time?
2. **CO₂ association:** How is atmospheric CO₂ related to September sea-ice extent?

An **ice-free September** is defined here as a sea-ice extent below **1 million km²**.

## Research question

How fast is September Arctic sea ice declining, and what do simple statistical models suggest about the timing of a future ice-free September?

## Data

The analysis uses annual data from **1979 to 2025**:

- September Arctic sea-ice extent from the **National Snow and Ice Data Center (NSIDC)**
- Annual mean atmospheric CO₂ concentration from **NOAA Global Monitoring Laboratory**

The main data files are:

- `N_09_extent_v4.0.csv`
- `co2_annmean_mlo.csv`

## Methods

### Route 1: Time trend

The first Python script uses:

- Linear regression to measure the yearly decline in sea ice
- A quadratic polynomial model to estimate when sea ice may cross the threshold
- An exponential model as a second threshold estimate

### Route 2: CO₂ association

The second Python script uses:

- Pearson correlation
- Ordinary least squares (OLS) regression
- Bayesian linear regression
- Posterior predictive simulation to estimate the probability of very low sea ice through 2100

## Main findings

- September Arctic sea ice declined by about **0.076 million km² per year**.
- The linear time-trend model has an R² of about **0.88**, showing a clear downward trend.
- The quadratic model crosses the ice-free threshold around **2064**.
- The exponential model crosses the threshold later, around **2139**.
- Higher CO₂ concentration is strongly associated with lower September sea-ice extent.
- An increase of **10 ppm CO₂** is associated with a decrease of about **0.39 million km²** in sea-ice extent.
- In the Bayesian forecast, the probability of sea ice falling below 1 million km² first becomes greater than 50% around **2076** and reaches about **98.4% by 2100**.

These results come from simple statistical models. They are useful for showing patterns, but they are **not physical climate-model forecasts**.

## Project files

| File | Description |
| --- | --- |
| `sea_ice_analysis_summer.py` | Analyses the time trend and fits the polynomial and exponential models. |
| `sea_ice_co2_analysis.py` | Studies the CO₂–sea-ice relationship using OLS and Bayesian regression. |
| `N_09_extent_v4.0.csv` | September Arctic sea-ice data. |
| `co2_annmean_mlo.csv` | Annual mean atmospheric CO₂ data. |

## How to run the project

### 1. Install Python

Python 3.9 or a newer version is recommended.

### 2. Install the required packages

```bash
pip install numpy pandas matplotlib statsmodels scipy
```

### 3. Put the files together

Place the Python scripts and both CSV data files in the same folder.

### 4. Run the scripts

```bash
python sea_ice_analysis_summer.py
python sea_ice_co2_analysis.py
```

The scripts print the statistical results and display the figures used in the project poster.

## Limitations

- The models use past statistical patterns to make long-term estimates.
- Different models give different threshold years.
- The analysis shows association, but it does not prove that CO₂ is the only cause of sea-ice decline.
- Real Arctic sea ice is affected by many physical and environmental processes.
# Precipitation Trend Analysis Across a European and North Africa

## Project overview

This project tests the hypothesis that climate change intensifies the hydrological
cycle so that **wet regions get wetter and dry regions get drier**. It analyses
76 years (1950–2025) of monthly precipitation from the ERA5 reanalysis for
**29 European and North-African cities** spanning a designed aridity gradient,
from very wet Atlantic oceanic sites (e.g. Bergen, ~2250 mm/yr) to near-desert
sites (e.g. Ouarzazate, ~120 mm/yr).

Two complementary analyses are used:

1. **Per-city trends:** How has each city's annual precipitation changed over time?
2. **Aridity gradient:** Does the magnitude of change scale with a city's baseline
   (starting) wetness or dryness?

## Research question

Does precipitation change over 1950–2025 scale with baseline aridity — i.e. do
wet regions gain rainfall while dry regions lose it — and is a gradient-based,
multi-station analysis more sensitive than testing single stations alone in
noisy arid climates?

## Data

The analysis uses annual precipitation totals from **1950 to 2025** (76 full
years; the incomplete 2026 row is dropped):

- **Source:** ERA5 reanalysis (ECMWF), obtained via the KNMI Climate Explorer
  (climexp.knmi.nl), `era5_tp` (total precipitation) field, monthly means in mm/day.
- **Spatial resolution:** 1°×1° grid boxes (~110 km) centred on each city.
- **Study sites:** 29 cities — 9 wet (Atlantic oceanic, Köppen *Cfb*) and
  20 dry (Mediterranean *Csa* and arid/semi-arid *B*), chosen a priori from the
  Köppen–Geiger classification and climatological baseline rainfall.

The data files live in `data_precipitation/`, organised as:

```
data_precipitation/
├── wet/   # wet cities: bergen, bilboa, brest, cork, dubin, glasglow,
│          #               plymouth, stavanger, valencia
└── dry/   # dry cities: agadir, alicante, almeria, atehns, belgrade, bordeux,
           #               gabes, istanbul, lisbon, ljubljana, madrid, marseille,
           #               murcia, ouarzazate, palermo, rome, seville, stockholm,
           #               tunis, warsaw
```

Each city folder contains one KNMI text file: `year + 12 monthly values` per row
(mm/day), with `#` comment lines and `-999.9` marking missing values.

## Methods

### Route 1: Per-city trend statistics
For each city's 76-year annual series:

- **Mann–Kendall test** — non-parametric, rank-based significance test for a
  monotonic trend (the headline test).
- **Sen's slope** — robust median-based estimate of the trend magnitude (mm/yr).
- **OLS linear regression** — complementary slope and p-value.
- **Split-period comparison** — mean rainfall 1950–1987 vs 1988–2025 with a
  Welch t-test; the percent change is the headline number.
- **Benjamini–Hochberg FDR correction** — guards against false positives from
  running many significance tests at once.

### Route 2: Aridity gradient (primary scientific test)
Each city becomes one dot: baseline rainfall (mm/yr) on the x-axis, percent
change (late vs early period) on the y-axis. A linear regression across all
29 sites asks whether wetter-starting cities gained while drier ones lost. This
test never groups cities, so it is independent of the wet/dry classification.

### Route 3: Wet–dry gap and CO₂ association
- **Wet–dry gap** — each year, mean rainfall of the wet cities minus mean of the
  dry cities; the gap is smoothed with a 10-year Butterworth low-pass filter to
  expose the multi-decadal trend, and tested with Mann–Kendall.
- **CO₂ correlation** — the gap is correlated with Mauna Loa annual-mean CO₂
  (NOAA GML), reporting both the raw Pearson correlation and the **detrended**
  correlation (the honest test of whether the two move together beyond their
  shared upward drift).

## Main findings

- **Wet cities are significantly wetter.** Glasgow (+9.9%) and Dublin (+9.1%)
  show significant wetting (Mann–Kendall p ≈ 0.0002, robust to FDR correction).
- **Dry cities are directionally drier.** Most dry cities show declining rainfall
  (Rome −10.8%, Almería −9.5%, Palermo −6.6%, Seville −6.3%, Tunis −5.5%,
  Athens −4.4%), though no single dry site reaches p < 0.05 on its own — dry
  climates are too variable for a single 76-year station record to prove a trend.
- **The aridity gradient is positive** — change scales with starting wetness —
  and the wet–dry gap widened over the record (~560 → ~700 mm/yr early vs late).
- **Methodological result:** pooling sites (the gradient and the gap) reveals a
  coherent wet-get-wetter / dry-get-drier pattern that single-station tests miss
  in noisy arid climates.
- **CO₂ co-evolution:** the gap tracks Mauna Loa CO₂; the correlation is
  consistent with a CO₂-driven mechanism but does not prove causation.

These results come from simple statistical models on reanalysis data. They are
useful for showing regional patterns, but they are not physical climate-model
forecasts.

## Project files

| File | Description |
| --- | --- |
| `precipitation trend analysis/precipitation_analysis_v2_extended.py` | Parses the KNMI files, runs per-city trends (Mann–Kendall, Sen's slope, OLS, split-period t-test, FDR), fits the aridity gradient, saves CSVs and figures. |
| `precipitation trend analysis/co2_and_gap_correlation.py` | Builds the wet–dry gap index, correlates it with Mauna Loa CO₂ (raw + detrended), reports per-city CO₂ correlations, saves the combined gap–CO₂ figure. |
| `data_precipitation/wet/` | ERA5 precipitation text files for the 9 wet (Atlantic oceanic) cities. |
| `data_precipitation/dry/` | ERA5 precipitation text files for the 20 dry (Mediterranean/arid) cities. |

Outputs produced by the scripts: `results_era5_trends.csv` (per-city results),
`aridity_gradient.csv`, `annual_precip_mm.csv`, `city_metadata.csv`, and the
figures `fig_time_series.png`, `fig_aridity_gradient.png`, `fig_gap_and_co2.png`.

## How to run the project

### 1. Install Python
Python 3.9 or a newer version is recommended.

### 2. Install the required packages

```bash
pip install numpy pandas matplotlib scipy pymannkendall
```

### 3. Put the files together
Place the two scripts and the `data_precipitation/` folder (with its `wet/` and
`dry/` subfolders) in the same folder.

### 4. Run the scripts

```bash
python precipitation_analysis_v2_extended.py
python co2_and_gap_correlation.py
```

(Also works in Google Colab: run the script contents in a cell after
`!pip install pymannkendall -q`, with the data uploaded.)

The scripts print the statistical results and display the figures used in the
project poster.

## Limitations

- ERA5 is reanalysis, not rain-gauge measurements — extremes may be smoothed,
  and the pre-1979 record relies more heavily on the model.
- 76 years is short for arid climates with very large interannual variability,
  which is why individual dry sites rarely reach significance on their own.
- Neighbouring 1° grid boxes are not fully independent samples.
- The analysis shows association (e.g. with CO₂), not causation.
- The wet/dry split is a display simplification; the continuous aridity gradient
  is the primary test and never groups cities.

## Authors

**Xingnuo Li**(24202272) - Arctic Sea-Ice Decline and Atmospheric CO₂

**Nandini Hazarika**(25201202) - Precipitation Trend Analysis Across a European and North Africa
