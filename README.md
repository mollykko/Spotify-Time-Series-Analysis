# Spotify Pop Music Loudness: Time Series Analysis & Forecasting

Forecasting how the loudness of pop music has changed over five decades, using moving averages and autoregressive (AR) models on 32,000+ Spotify tracks.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![pandas](https://img.shields.io/badge/pandas-data%20wrangling-150458) ![statsmodels](https://img.shields.io/badge/statsmodels-AutoReg-4B8BBE) ![Colab](https://img.shields.io/badge/Google%20Colab-notebook-F9AB00)

---

## Overview

Music producers and labels make investment decisions based on where listener preferences are heading. This project takes one measurable audio feature, **loudness**, and asks:

> *How has the loudness of pop songs evolved over time, and can its past behavior forecast where it is going?*

The notebook walks through the full workflow: pulling data programmatically from Kaggle, validating data quality, engineering time features, building and comparing forecasting models, and checking those models against standard statistical assumptions.

## Key Results

| Model | What it does | Result |
|---|---|---|
| **Moving Average (MA)** | Forecasts each value as the average of the previous *n* observations | Tested windows from 3 to 30. MAD fell from **1.89 dB** (MA-3) to a low of **1.75 dB** (MA-24), with gains flattening after a window of ~12 |
| **Autoregressive AR(2)** | Predicts each year's mean loudness from the two preceding years | Both lag terms statistically significant (p = 0.002 and p = 0.008). Innovation SD **1.20 dB**. Model is **stationary** (all roots outside the unit circle) |

**Takeaways**

- **Loudness has momentum.** Roughly 40% of last year's mean loudness and 35% of the year before carry forward into the current year (AR coefficients 0.40 and 0.35).
- **The series is stationary.** The AR(2) roots (moduli 1.21 and 2.37) both lie outside the unit circle, so the historical relationships are stable enough to support forecasting without differencing.
- **Longer windows smooth noise, up to a point.** Moving-average accuracy improves quickly as the window grows, then plateaus, which shows the trade-off between responsiveness and stability.
- **Forecast uncertainty can drive decisions.** The notebook also shows how a forecast's mean and standard deviation turn into probabilities (e.g., stockout risk) and safety levels using the normal distribution.

## Dataset

- **Source:** [30,000 Spotify Songs](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs) on Kaggle (originally collected via the `spotifyr` R package and the Spotify Web API)
- **Size:** 32,833 tracks × 23 columns
- **Genres:** EDM (6,043), rap (5,746), pop (5,507), R&B (5,431), Latin (5,155), rock (4,951)
- **Features used:** `track_album_release_date`, `playlist_genre`, `loudness` (dB), plus audio features such as danceability, energy, tempo and valence
- **Data quality:** 0 duplicate rows; 5 missing values each in `track_name`, `track_artist` and `track_album_name` (no effect on the time series analysis)

## Methodology

### 1. Data sourcing
Data is downloaded programmatically with `kagglehub`, so the notebook is reproducible without manual file handling.

### 2. Data validation & exploration
Shape, data types, duplicates, missing values, summary statistics and genre distribution.

### 3. Feature engineering
Release dates come in mixed formats (`YYYY`, `YYYY-MM-DD`). They are parsed with `pd.to_datetime(..., format='mixed', errors='coerce')`, then `year`, `month` and `year_month` features are extracted.

### 4. Moving-average forecasting
- Lagged moving average (`.rolling(n).mean().shift(1)`), so each forecast uses only past data and avoids look-ahead leakage
- `min_periods = window` to ensure every forecast is based on a full window
- Accuracy measured with **Mean Absolute Deviation (MAD)**:

$$\text{MAD} = \frac{1}{N}\sum_{t=1}^{N} |Y_t - \hat{Y}_t|$$

### 5. Autoregressive modeling
- Aggregated pop-song loudness to a **yearly mean** series with a proper `DatetimeIndex`
- Used the **Partial Autocorrelation Function (PACF)** to choose the model order, selecting *p = 2*
- Fit an **AR(2)** model with `statsmodels.tsa.ar_model.AutoReg`:

$$Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \varepsilon_t$$

- Interpreted coefficients, significance, information criteria (AIC/BIC/HQIC) and characteristic roots
- Compared **SD of innovations** (the theoretical noise floor) with **RMSE** (achieved accuracy) to judge model specification

### 6. Forecasting & assumptions
- Out-of-sample forecast of mean loudness for **2021–2028**
- Reviewed AR assumptions: stationarity, linearity, white-noise residuals, homoscedasticity and normality

## Tech Stack

| Purpose | Tools |
|---|---|
| Data access | `kagglehub` |
| Data wrangling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Time series modeling | `statsmodels` (`AutoReg`, `plot_pacf`) |
| Probability | `scipy.stats` |
| Environment | Google Colab / Jupyter |

## How to Run

**Option A: Google Colab (recommended)**
1. Upload `Spotify_Time_Series_Analysis.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Select **Runtime → Run all**. The dataset downloads automatically.

**Option B: Run locally**
```bash
git clone <your-repo-url>
cd <repo-folder>
pip install kagglehub pandas numpy matplotlib seaborn statsmodels scipy jupyter
jupyter notebook Spotify_Time_Series_Analysis.ipynb
```
> Running locally may require a Kaggle API token (`~/.kaggle/kaggle.json`). See [Kaggle's API docs](https://www.kaggle.com/docs/api). You can also download `spotify_songs.csv` directly from the dataset page.

## Project Structure

```
.
├── Spotify_Time_Series_Analysis.ipynb   # Main analysis notebook
└── README.md
```

## Skills Demonstrated

- End-to-end analytics workflow, from raw data to business-relevant forecast
- Time series concepts: moving averages, autocorrelation, PACF, AR models, stationarity
- Model evaluation with MAD, RMSE, information criteria and residual diagnostics
- Avoiding data leakage in forecasting with lagged features
- Turning forecast uncertainty into decision metrics such as risk probabilities and safety levels
- Clear technical communication: every step is documented with explanations and formulas

## Future Improvements

- Compare against ARIMA, exponential smoothing (Holt-Winters) and Prophet
- Use a train/test split or rolling-origin cross-validation for true out-of-sample evaluation
- Add residual diagnostics (ACF of residuals, Ljung-Box test) and prediction intervals on forecasts
- Extend the analysis to other genres and audio features (energy, danceability, tempo)
- Explore the relationship between loudness trends and `track_popularity`

## Author

**[Your Name]**
[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username) · your.email@example.com

---

*Dataset credit: Spotify Web API via the `spotifyr` package (Charlie Thompson, Josiah Parry, Donal Phipps, Tom Wolff), published on Kaggle by joebeachcapital.*
