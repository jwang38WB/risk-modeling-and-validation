"""
Economic Capital for Credit Risk — Four Model Approaches
=========================================================================

Hypothetical 10-obligor portfolio run through four different credit-portfolio models,
each with a genuinely different mechanism for turning "obligor risk" into
"portfolio Economic Capital":

  1. KMV / Merton      — structural model: infer Distance-to-Default and an
                          Expected Default Frequency from asset value vs. debt,
                          then roll up to portfolio capital via the Vasicek
                          single-factor (ASRF) formula.
  2. CreditMetrics      — mark-to-market model: use a rating-transition matrix
                          to build the full distribution of value (not just
                          default) for each obligor one year out, then
                          aggregate via an asset-correlation matrix.
  3. CreditRisk+        — actuarial model: treat defaults as a Poisson process
                          over exposure bands; build the *exact* aggregate
                          loss distribution with the Panjer recursion.
  4. CreditPortfolioView — macroeconomic model: condition each obligor's PD on
                          a systematic macro index across economic scenarios.

Confidence level: 99.9%. Horizon: 1 year.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUT_DIR, exist_ok=True)
pd.options.display.float_format = "{:,.4f}".format

# =============================================================================
# 1. SHARED PORTFOLIO & ASSUMPTIONS
# =============================================================================

CONFIDENCE = 0.999      # Economic Capital confidence level
RISK_FREE_RATE = 0.04   # r
HORIZON = 1              # T, years
LGD = 0.45                # flat LGD assumption (Basel FIRB senior-unsecured proxy)

# Rating -> 1yr PD lookup (illustrative, approximates long-run S&P average
# one-year global corporate default rates).
RATING_PD = {
    "AAA": 0.0001, "AA": 0.0002, "A": 0.0005, "BBB": 0.0015,
    "BB": 0.0085, "B": 0.0400, "CCC": 0.1500,
}

PORTFOLIO = pd.DataFrame([
    (1,  "Alpha Manufacturing Corp", "Manufacturing", "BBB", 40),
    (2,  "Beta Retail Group",        "Retail",         "BB", 25),
    (3,  "Gamma Tech Solutions",     "Technology",     "A",  35),
    (4,  "Delta Energy Partners",    "Energy",         "BB", 50),
    (5,  "Epsilon Financial Svcs",   "Financials",     "BBB",30),
    (6,  "Zeta Consumer Goods",      "Retail",         "B",  15),
    (7,  "Eta Industrial Inc",       "Manufacturing",  "BB", 20),
    (8,  "Theta Software Corp",      "Technology",     "BBB",45),
    (9,  "Iota Utilities Co",        "Energy",         "A",  60),
    (10, "Kappa Holdings",           "Financials",     "B",  20),
], columns=["id", "obligor", "sector", "rating", "ead_mm"])

PORTFOLIO["pd_1yr"] = PORTFOLIO["rating"].map(RATING_PD)
PORTFOLIO["el_mm"] = PORTFOLIO["ead_mm"] * PORTFOLIO["pd_1yr"] * LGD

TOTAL_EAD = PORTFOLIO["ead_mm"].sum()


def basel_asset_correlation(pd_: np.ndarray) -> np.ndarray:
    """
    Basel corporate risk-weight asset-correlation formula:
        rho(PD) = 0.12*(1-e^-50PD)/(1-e^-50) + 0.24*(1-(1-e^-50PD)/(1-e^-50))
    Derived from the same Merton/KMV single-factor asset framework used below.
    """
    w = (1 - np.exp(-50 * pd_)) / (1 - np.exp(-50))
    return 0.12 * w + 0.24 * (1 - w)


PORTFOLIO["asset_corr"] = basel_asset_correlation(PORTFOLIO["pd_1yr"].to_numpy())


def print_portfolio():
    print("=" * 100)
    print("SHARED PORTFOLIO (10 obligors)")
    print("=" * 100)
    show = PORTFOLIO.copy()
    print(show.to_string(index=False, formatters={
        "ead_mm": "{:,.0f}".format, "pd_1yr": "{:.2%}".format,
        "el_mm": "{:,.2f}".format, "asset_corr": "{:.1%}".format}))
    print(f"\nTotal EAD: ${TOTAL_EAD:,.0f}mm   LGD: {LGD:.0%}   Confidence: {CONFIDENCE:.1%}")
    print("=" * 100 + "\n")
    PORTFOLIO.to_csv(os.path.join(OUT_DIR, "ec4_portfolio.csv"), index=False)


# =============================================================================
# 2. MODEL 1 — KMV / MERTON STRUCTURAL MODEL
# =============================================================================
# Rating -> hypothetical structural assumptions (asset volatility, leverage).
# Illustrative mapping used only to build a Merton/KMV worked example -- real
# KMV backs these out of observed equity value/volatility.
KMV_ASSUMPTIONS = {
    "AAA": dict(asset_vol=0.15, leverage=0.20),
    "AA":  dict(asset_vol=0.18, leverage=0.25),
    "A":   dict(asset_vol=0.20, leverage=0.30),
    "BBB": dict(asset_vol=0.25, leverage=0.40),
    "BB":  dict(asset_vol=0.32, leverage=0.55),
    "B":   dict(asset_vol=0.40, leverage=0.70),
    "CCC": dict(asset_vol=0.55, leverage=0.85),
}


def run_kmv(df: pd.DataFrame, confidence: float = CONFIDENCE) -> dict:
    """
    Step 1 (per obligor): Merton distance-to-default and EDF.
        D = debt face value (approximated by EAD)
        V = D / leverage                                (asset value)
        DD = [ln(V/D) + (r - 0.5*sigmaV^2)*T] / (sigmaV*sqrt(T))
        EDF = N(-DD)

    Step 2 (portfolio): Vasicek/ASRF single-factor capital, using each
    obligor's own KMV-implied EDF as its PD, and the Basel asset-correlation
    formula (evaluated at the obligor's rating-based PD, matching the Excel
    workbook) as rho.
        PD(alpha) = N[(N^-1(EDF) + sqrt(rho)*N^-1(alpha)) / sqrt(1-rho)]
        K = LGD * (PD(alpha) - EDF)
        EC_i = EAD_i * K
    """
    out = df.copy()
    vol = out["rating"].map(lambda r: KMV_ASSUMPTIONS[r]["asset_vol"])
    lev = out["rating"].map(lambda r: KMV_ASSUMPTIONS[r]["leverage"])
    D = out["ead_mm"]
    V = D / lev
    dd = (np.log(V / D) + (RISK_FREE_RATE - 0.5 * vol ** 2) * HORIZON) / (vol * np.sqrt(HORIZON))
    edf = norm.cdf(-dd)

    out["asset_vol"] = vol
    out["leverage"] = lev
    out["distance_to_default"] = dd
    out["edf"] = edf

    rho = out["asset_corr"].to_numpy()
    edf_arr = np.clip(edf, 1e-9, 0.9999)
    stressed = norm.cdf((norm.ppf(edf_arr) + np.sqrt(rho) * norm.ppf(confidence)) / np.sqrt(1 - rho))
    out["stressed_pd"] = stressed
    out["capital_rate"] = LGD * (stressed - edf_arr)
    out["ec_mm"] = out["ead_mm"] * out["capital_rate"]

    return dict(detail=out, ec_total=out["ec_mm"].sum(), el_total=out["el_mm"].sum())


def print_kmv(res: dict):
    print("=" * 100)
    print("MODEL 1 — KMV / MERTON STRUCTURAL MODEL")
    print("=" * 100)
    print("DD = [ln(V/D) + (r - 0.5*sigma^2)*T] / (sigma*sqrt(T))     EDF = N(-DD)")
    print("PD(alpha) = N[(N^-1(EDF) + sqrt(rho)*N^-1(alpha)) / sqrt(1-rho)]     EC = EAD * LGD * (PD(alpha)-EDF)\n")
    show = res["detail"][["obligor", "rating", "ead_mm", "asset_vol", "leverage",
                            "distance_to_default", "edf", "pd_1yr", "ec_mm"]].copy()
    show.columns = ["Obligor", "Rating", "EAD", "AssetVol", "Leverage", "DD", "EDF", "RatingPD", "EC($mm)"]
    print(show.to_string(index=False, formatters={
        "EAD": "{:,.0f}".format, "AssetVol": "{:.0%}".format, "Leverage": "{:.0%}".format,
        "DD": "{:.2f}".format, "EDF": "{:.4%}".format, "RatingPD": "{:.2%}".format,
        "EC($mm)": "{:,.2f}".format}))
    print(f"\nKMV Portfolio Economic Capital (99.9%): ${res['ec_total']:,.2f}mm")
    print("=" * 100 + "\n")


# =============================================================================
# 3. MODEL 2 — CREDITMETRICS (RATINGS MIGRATION / MARK-TO-MARKET)
# =============================================================================
# Illustrative 1-year rating transition matrix (%) -- approximates S&P's
# historical average, as widely reproduced in the credit-risk literature
# (e.g., CreditMetrics Technical Document, 1997).
RATINGS_8 = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
TRANSITION_MATRIX = pd.DataFrame([
    [90.81, 8.33, 0.68, 0.06, 0.08, 0.02, 0.01, 0.01],
    [0.70, 90.65, 7.79, 0.64, 0.06, 0.14, 0.02, 0.00],
    [0.09, 2.27, 91.05, 5.52, 0.74, 0.26, 0.01, 0.06],
    [0.02, 0.33, 5.95, 86.93, 5.30, 1.17, 0.12, 0.18],
    [0.03, 0.14, 0.67, 7.73, 80.53, 8.84, 1.00, 1.06],
    [0.00, 0.11, 0.24, 0.43, 6.48, 83.46, 4.07, 5.20],
    [0.22, 0.00, 0.22, 1.30, 2.38, 11.24, 64.86, 19.78],
], index=RATINGS_8[:-1], columns=RATINGS_8) / 100.0

# Illustrative valuation factor (% of EAD) by year-end rating state.
VALUE_FACTORS = pd.Series(
    {"AAA": 1.005, "AA": 1.004, "A": 1.002, "BBB": 1.000,
     "BB": 0.95, "B": 0.85, "CCC": 0.65, "D": 1 - LGD}, name="value_factor")

SAME_SECTOR_RHO = 0.30
DIFF_SECTOR_RHO = 0.10


def run_creditmetrics(df: pd.DataFrame, confidence: float = CONFIDENCE) -> dict:
    """
    Step 1 (per obligor): full forward-value distribution one year out.
        E[Value]  = sum_state p(state) * ValueFactor(state) * EAD
        sigma(V)  = sqrt( sum_state p(state) * (ValueFactor(state)*EAD - E[Value])^2 )
    Captures BOTH migration risk (small, likely moves) and default risk
    (large, unlikely loss) in a single sigma per obligor.

    Step 2 (portfolio): variance-covariance aggregation via an asset-
    correlation matrix (same-sector obligors correlated at 30%, different-
    sector at 10% -- a simplified proxy for CreditMetrics' equity-return
    correlations).
        Var[portfolio] = sum_i sum_j rho_ij * sigma_i * sigma_j
        EC = z_alpha * sqrt(Var[portfolio])        (normal approximation)
    """
    out = df.copy()
    e_value, sigma = [], []
    for _, row in out.iterrows():
        probs = TRANSITION_MATRIX.loc[row["rating"]]
        values = VALUE_FACTORS.reindex(probs.index) * row["ead_mm"]
        ev = (probs * values).sum()
        var = (probs * (values - ev) ** 2).sum()
        e_value.append(ev)
        sigma.append(np.sqrt(var))
    out["e_value_mm"] = e_value
    out["sigma_mm"] = sigma
    out["el_cm_mm"] = out["ead_mm"] - out["e_value_mm"]

    n = len(out)
    sectors = out["sector"].to_numpy()
    corr = np.where(sectors[:, None] == sectors[None, :], SAME_SECTOR_RHO, DIFF_SECTOR_RHO)
    np.fill_diagonal(corr, 1.0)
    sigma_vec = out["sigma_mm"].to_numpy()
    portfolio_var = sigma_vec @ corr @ sigma_vec
    portfolio_sigma = np.sqrt(portfolio_var)
    z = norm.ppf(confidence)
    ec_total = z * portfolio_sigma

    return dict(detail=out, corr_matrix=corr, portfolio_sigma=portfolio_sigma,
                ec_total=ec_total, el_total=out["el_cm_mm"].sum())


def print_creditmetrics(res: dict):
    print("=" * 100)
    print("MODEL 2 — CREDITMETRICS (RATINGS MIGRATION / MARK-TO-MARKET)")
    print("=" * 100)
    print("E[Value] = sum p(state)*ValueFactor(state)*EAD     sigma(V) = sqrt(sum p(state)*(Value-E[Value])^2)")
    print("Portfolio Var = sum_i sum_j rho_ij*sigma_i*sigma_j     EC = z_99.9% * sqrt(Portfolio Var)\n")
    show = res["detail"][["obligor", "sector", "rating", "ead_mm", "e_value_mm", "sigma_mm", "el_cm_mm"]].copy()
    show.columns = ["Obligor", "Sector", "Rating", "EAD", "E[Value]", "Sigma", "EL($mm)"]
    print(show.to_string(index=False, formatters={
        "EAD": "{:,.0f}".format, "E[Value]": "{:,.2f}".format,
        "Sigma": "{:,.2f}".format, "EL($mm)": "{:,.2f}".format}))
    print(f"\nPortfolio sigma(Value): ${res['portfolio_sigma']:,.2f}mm")
    print(f"CreditMetrics Portfolio Economic Capital (99.9%): ${res['ec_total']:,.2f}mm")
    print("Note: this is a normal approximation (z*sigma) to the 99.9th-percentile loss for")
    print("transparency of formulas; CreditMetrics' own methodology simulates correlated asset")
    print("returns via Monte Carlo to capture the fat left tail more accurately.")
    print("=" * 100 + "\n")


# =============================================================================
# 4. MODEL 3 — CREDITRISK+ (ACTUARIAL / POISSON, PANJER RECURSION)
# =============================================================================
LOSS_UNIT_MM = 2.0   # L: size of one discrete "loss unit" band
N_BANDS = 25          # max severity band, in loss units
N_LOSS_UNITS = 60      # how far to build the recursion (n = 0..60)


def run_creditrisk_plus(df: pd.DataFrame, confidence: float = CONFIDENCE,
                          loss_unit: float = LOSS_UNIT_MM) -> dict:
    """
    Step 1: band each obligor's loss-given-default into whole loss units.
        v_i = max(1, round(EAD_i * LGD / L))
    Step 2: for each band b, lambda_b = sum of PD_i over obligors with v_i = b
            (Poisson approximation to each obligor's Bernoulli default).
    Step 3: Panjer recursion for the compound-Poisson aggregate loss pmf:
        f(0) = exp(-lambda_total)
        f(n) = (1/n) * sum_b [ lambda_b * b * f(n-b) ],  n = 1, 2, 3, ...
    This is the EXACT probability mass function of total portfolio loss (in
    loss units) -- no simulation, no correlation parameter needed, because
    concentration risk is captured directly by the severity bands.
    """
    out = df.copy()
    out["loss_units"] = np.maximum(1, np.round(out["ead_mm"] * LGD / loss_unit).astype(int))

    lam_by_band = np.zeros(N_BANDS + 1)  # index 0 unused
    for b in range(1, N_BANDS + 1):
        lam_by_band[b] = out.loc[out["loss_units"] == b, "pd_1yr"].sum()
    lam_total = lam_by_band.sum()

    f = np.zeros(N_LOSS_UNITS + 1)
    f[0] = np.exp(-lam_total)
    for n in range(1, N_LOSS_UNITS + 1):
        s = 0.0
        for b in range(1, min(n, N_BANDS) + 1):
            s += lam_by_band[b] * b * f[n - b]
        f[n] = s / n

    cdf = np.cumsum(f)
    var_units = int(np.searchsorted(cdf, confidence))
    var_mm = var_units * loss_unit
    el_mm = (lam_by_band * np.arange(N_BANDS + 1)).sum() * loss_unit
    ec_mm = var_mm - el_mm

    return dict(detail=out, lam_by_band=lam_by_band, pmf=f, cdf=cdf,
                var_units=var_units, var_mm=var_mm, el_total=el_mm, ec_total=ec_mm)


def print_creditrisk_plus(res: dict):
    print("=" * 100)
    print("MODEL 3 — CREDITRISK+ (ACTUARIAL / POISSON, PANJER RECURSION)")
    print("=" * 100)
    print("f(0) = exp(-lambda)     f(n) = (1/n) * sum_b [lambda_b * b * f(n-b)]    (Panjer recursion)\n")
    show = res["detail"][["obligor", "ead_mm", "pd_1yr", "loss_units"]].copy()
    show.columns = ["Obligor", "EAD", "PD", "LossUnits(v_i)"]
    print(show.to_string(index=False, formatters={
        "EAD": "{:,.0f}".format, "PD": "{:.2%}".format}))
    print(f"\nLoss unit L = ${LOSS_UNIT_MM:,.0f}mm     Total expected defaults (lambda): "
          f"{res['lam_by_band'].sum():.4f}")
    print(f"99.9% VaR: {res['var_units']} loss units = ${res['var_mm']:,.2f}mm")
    print(f"Expected Loss: ${res['el_total']:,.2f}mm")
    print(f"CreditRisk+ Portfolio Economic Capital (99.9%) = VaR - EL: ${res['ec_total']:,.2f}mm")
    print("=" * 100 + "\n")


# =============================================================================
# 5. MODEL 4 — CREDITPORTFOLIOVIEW (MACROECONOMIC / CONDITIONAL PD)
# =============================================================================
BETA0, BETA1, BETA2 = -2.2, -8.0, 6.0   # illustrative logit regression coefficients

SCENARIOS = pd.DataFrame([
    ("Severe recession", 0.05, -0.030, 0.020),
    ("Recession",         0.15, -0.010, 0.010),
    ("Baseline",          0.50,  0.020, 0.000),
    ("Expansion",         0.20,  0.035, -0.005),
    ("Strong expansion",  0.10,  0.050, -0.012),
], columns=["scenario", "probability", "gdp_growth", "delta_unemployment"])
assert abs(SCENARIOS["probability"].sum() - 1.0) < 1e-9


def run_creditportfolioview(df: pd.DataFrame, confidence: float = CONFIDENCE) -> dict:
    """
    Step 1: systematic macro index and logit link, per scenario.
        Y_t = beta0 + beta1*GDPgrowth_t + beta2*(-DeltaUnemployment_t)
        p_t = 1 / (1 + e^-Y_t)                       (logit default index)
        M_t = p_t / p_baseline                        (PD multiplier)
        PD_i,t = PD_i * M_t                            (conditional obligor PD)

    Step 2: conditional portfolio loss per scenario.
        EL_t  = sum_i EAD_i*LGD*PD_i,t
        Var_t = sum_i (EAD_i*LGD)^2 * PD_i,t*(1-PD_i,t)   (idiosyncratic risk
                within the scenario; the scenario itself carries systematic risk)

    Step 3: combine scenarios via the law of total variance.
        E[L]   = sum_t pi_t * EL_t
        Var[L] = sum_t pi_t*(Var_t + EL_t^2) - E[L]^2
        EC     = z_alpha * sqrt(Var[L])                 (normal approximation)
    """
    scen = SCENARIOS.copy()
    scen["Y"] = BETA0 + BETA1 * scen["gdp_growth"] + BETA2 * (-scen["delta_unemployment"])
    scen["p"] = 1 / (1 + np.exp(-scen["Y"]))
    baseline_p = scen.loc[scen["scenario"] == "Baseline", "p"].iloc[0]
    scen["pd_multiplier"] = scen["p"] / baseline_p

    ead_lgd = (df["ead_mm"] * LGD).to_numpy()
    pd_base = df["pd_1yr"].to_numpy()

    el_list, var_list = [], []
    for m in scen["pd_multiplier"]:
        pd_cond = pd_base * m
        el_list.append((ead_lgd * pd_cond).sum())
        var_list.append((ead_lgd ** 2 * pd_cond * (1 - pd_cond)).sum())
    scen["el_mm"] = el_list
    scen["var_mm2"] = var_list

    e_l = (scen["probability"] * scen["el_mm"]).sum()
    var_l = (scen["probability"] * (scen["var_mm2"] + scen["el_mm"] ** 2)).sum() - e_l ** 2
    sigma_l = np.sqrt(var_l)
    z = norm.ppf(confidence)
    ec_total = z * sigma_l

    return dict(scenarios=scen, e_l=e_l, sigma_l=sigma_l, ec_total=ec_total, el_total=e_l)


def print_creditportfolioview(res: dict):
    print("=" * 100)
    print("MODEL 4 — CREDITPORTFOLIOVIEW (MACROECONOMIC / CONDITIONAL PD)")
    print("=" * 100)
    print("Y_t = b0 + b1*GDPgrowth + b2*(-dUnemployment)     p_t = 1/(1+e^-Y_t)     M_t = p_t/p_baseline")
    print("EL_t = sum EAD*LGD*PD_t     Var_t = sum (EAD*LGD)^2*PD_t*(1-PD_t)")
    print("E[L] = sum pi_t*EL_t     Var[L] = sum pi_t*(Var_t+EL_t^2) - E[L]^2     EC = z*sqrt(Var[L])\n")
    show = res["scenarios"][["scenario", "probability", "gdp_growth", "delta_unemployment",
                              "pd_multiplier", "el_mm"]].copy()
    show.columns = ["Scenario", "Prob", "GDPgrowth", "dUnemployment", "PD_multiplier", "EL($mm)"]
    print(show.to_string(index=False, formatters={
        "Prob": "{:.0%}".format, "GDPgrowth": "{:.1%}".format, "dUnemployment": "{:.1%}".format,
        "PD_multiplier": "{:.2f}x".format, "EL($mm)": "{:,.2f}".format}))
    print(f"\nPortfolio E[L]: ${res['e_l']:,.2f}mm     Portfolio sigma[L]: ${res['sigma_l']:,.2f}mm")
    print(f"CreditPortfolioView Economic Capital (99.9%): ${res['ec_total']:,.2f}mm")
    print("=" * 100 + "\n")


# =============================================================================
# 6. SUMMARY & CHARTS
# =============================================================================
def build_summary(kmv, cm, crp, cpv) -> pd.DataFrame:
    rows = [
        ("KMV / Merton",         "Structural: infers PD from asset value vs. debt",              kmv["el_total"], kmv["ec_total"]),
        ("CreditMetrics",        "Mark-to-market: full ratings-migration distribution",            cm["el_total"],  cm["ec_total"]),
        ("CreditRisk+",          "Actuarial: Poisson default counts, Panjer recursion",             crp["el_total"], crp["ec_total"]),
        ("CreditPortfolioView",  "Macro-conditional: PD scaled by economic scenario",               cpv["el_total"], cpv["ec_total"]),
    ]
    df = pd.DataFrame(rows, columns=["model", "mechanism", "el_mm", "ec_mm"])
    df["ec_pct_of_ead"] = df["ec_mm"] / TOTAL_EAD
    return df


def print_summary(summary: pd.DataFrame):
    print("=" * 100)
    print("ECONOMIC CAPITAL — MODEL COMPARISON")
    print("=" * 100)
    print(summary.to_string(index=False, formatters={
        "el_mm": "{:,.2f}".format, "ec_mm": "{:,.2f}".format, "ec_pct_of_ead": "{:.2%}".format}))
    print(f"\nTotal portfolio EAD: ${TOTAL_EAD:,.0f}mm")
    print("-" * 100)
    print("Why the numbers differ: KMV and CreditMetrics both price in default correlation")
    print("through an asset-value factor (rho), so they are the closest conceptually, but")
    print("CreditMetrics also charges capital for pure downgrade risk, which KMV's EDF-only")
    print("view does not. CreditRisk+ has no explicit correlation and instead concentrates")
    print("loss on a few large severity bands. CreditPortfolioView starts from the same PDs")
    print("as the others but stresses them by macro scenario, so its capital reflects the")
    print("assumed economic-cycle sensitivity (b1, b2) as much as the portfolio itself.")
    print("=" * 100 + "\n")
    summary.to_csv(os.path.join(OUT_DIR, "ec4_summary.csv"), index=False)


def make_summary_chart(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1F4E78", "#4472C4", "#ED7D31", "#70AD47"]
    bars = ax.bar(summary["model"], summary["ec_mm"], color=colors)
    for b, v in zip(bars, summary["ec_mm"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"${v:,.1f}mm", ha="center", fontsize=9)
    ax.set_ylabel("Economic Capital ($mm)")
    ax.set_title("Economic Capital by Model (99.9% confidence, same $340mm portfolio)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ec4_summary_chart.png"), dpi=150)
    plt.close(fig)


def make_creditrisk_plus_chart(res: dict):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(res["pmf"])) * LOSS_UNIT_MM
    ax.bar(x, res["pmf"], width=LOSS_UNIT_MM * 0.9, color="#1F4E78")
    ax.axvline(res["var_mm"], color="darkred", linestyle="--",
               label=f"99.9% VaR (${res['var_mm']:,.0f}mm)")
    ax.set_xlabel("Portfolio loss ($mm)")
    ax.set_ylabel("Probability")
    ax.set_title("CreditRisk+ exact aggregate loss distribution (Panjer recursion)")
    ax.set_xlim(0, res["var_mm"] * 1.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ec4_creditrisk_plus_loss_distribution.png"), dpi=150)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print_portfolio()

    kmv = run_kmv(PORTFOLIO)
    print_kmv(kmv)

    cm = run_creditmetrics(PORTFOLIO)
    print_creditmetrics(cm)

    crp = run_creditrisk_plus(PORTFOLIO)
    print_creditrisk_plus(crp)

    cpv = run_creditportfolioview(PORTFOLIO)
    print_creditportfolioview(cpv)

    summary = build_summary(kmv, cm, crp, cpv)
    print_summary(summary)

    make_summary_chart(summary)
    make_creditrisk_plus_chart(crp)
    print(f"Charts and CSVs written to: {OUT_DIR}")
