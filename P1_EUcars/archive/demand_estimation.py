"""Level-price demand: OLS, IV logit and demand-only BLP.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
import pyblp
from scipy.special import roots_hermitenorm


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output_simple_demand"
OUT.mkdir(exist_ok=True)

pyblp.options.verbose = False


# ------------------------------------------------------------------
# 1. Load data and define prices, characteristics and market shares
# ------------------------------------------------------------------

d = pd.read_csv(ROOT / "data" / "cars.csv")

d = (
    d.sort_values(["ma", "ye", "co"])
    .reset_index(drop=True)
)

d["market_ids"] = (
    d["ma"].astype(str) + "_" + d["ye"].astype(str)
)

d["firm_ids"] = d["frm"]
d["clustering_ids"] = d["co"]

# Price relative to annual GDP per capita.
d["prices"] = d["princ"]

# One potential annual purchase opportunity per household.
d["market_size"] = d["pop"] / 2.5
d["shares"] = d["qu"] / d["market_size"]

d["outside_share"] = (
    1 - d.groupby("market_ids")["shares"].transform("sum")
)

d["weight"] = d["we"] / 1000
d["fuel"] = d["li"]
d["power"] = d["hp"] / d["weight"] / 100
d["size"] = d["le"] * d["wi"] / 10000

controls = ["weight", "fuel", "home", "power", "size"]

required = controls + ["prices", "shares", "outside_share"]

assert d[required].notna().all().all()
assert (d[["prices", "shares", "outside_share"]] > 0).all().all()

print(f"Observations: {len(d):,}")
print("Markets:", d["market_ids"].nunique())
print(
    "Mean included-car share:",
    d.groupby("market_ids")["shares"].sum().mean(),
)


# ------------------------------------------------------------------
# 2. Invert homogeneous-logit demand
#
# delta = x'beta + brand FE + market FE - alpha*price + xi
# ------------------------------------------------------------------

d["delta"] = (
    np.log(d["shares"])
    - np.log(d["outside_share"])
)

print("\nFirst five inverted utilities:")
print(d["delta"].head().to_numpy())


# ------------------------------------------------------------------
# 3. Construct four rival-only quadratic differentiation instruments
#
# For each characteristic:
# Z_j = sum over rival products k of (x_k - x_j)^2
# ------------------------------------------------------------------

distances = pyblp.build_differentiation_instruments(
    pyblp.Formulation("0 + weight + fuel + power + size"),
    d,
    version="quadratic",
)

# PyBLP returns four own-firm columns, then four rival columns.
Z = distances[:, 4:]

# Match the scaling used in the earlier estimates.
Z = Z / Z.std(axis=0)

instrument_names = [
    "rival_weight",
    "rival_fuel",
    "rival_power",
    "rival_size",
]

d[instrument_names] = Z

# PyBLP recognises excluded instruments by these column names.
for k in range(4):
    d[f"demand_instruments{k}"] = Z[:, k]


# ------------------------------------------------------------------
# 4. OLS: one formula, with fixed effects included directly
# ------------------------------------------------------------------

ols = smf.ols(
    formula=(
        "delta ~ prices "
        "+ weight + fuel + home + power + size "
        "+ C(market_ids) + C(brd)"
    ),
    data=d,
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups": d["co"],
        # Match the earlier uncorrected clustered covariance.
        "use_correction": False,
    },
)


# ------------------------------------------------------------------
# 5. IV: one formula; linearmodels performs both stages internally
# ------------------------------------------------------------------

iv = IV2SLS.from_formula(
    formula=(
        "delta ~ 1 "
        "+ weight + fuel + home + power + size "
        "+ C(market_ids) + C(brd) "
        "+ [prices ~ rival_weight + rival_fuel "
        "+ rival_power + rival_size]"
    ),
    data=d,
).fit(
    cov_type="clustered",
    clusters=d["co"],
    debiased=False,
)

first_stage = iv.first_stage.diagnostics

print("\nIV first-stage diagnostics:")
print(first_stage.to_string())

print(
    "Clustered first-stage Wald / 4:",
    first_stage.iloc[0]["f.stat"] / 4,
)

print("\nPrice-exogeneity test, conditional on valid instruments:")
print(iv.wooldridge_regression)

# These diagnostics do not establish instrument exclusion.


# ------------------------------------------------------------------
# 6. Quadrature for the BLP consumer integral
# ------------------------------------------------------------------

# Gaussian quadrature for a standard normal distribution.
normal_nodes, normal_weights = roots_hermitenorm(40)

# Normalise the weights to integrate against the normal density.
normal_weights = normal_weights / np.sqrt(2 * np.pi)

market_ids = d["market_ids"].drop_duplicates().to_numpy()

agents = pd.DataFrame({
    "market_ids": np.repeat(market_ids, len(normal_nodes)),
    "nodes0": np.tile(normal_nodes, len(market_ids)),
    "weights": np.tile(normal_weights, len(market_ids)),
    "one": 1.0,
})

print("Quadrature points per market:", len(normal_nodes))
print("Weights sum to:", normal_weights.sum())
print("Approximated E[nu]:", normal_weights @ normal_nodes)
print("Approximated E[nu²]:", normal_weights @ normal_nodes**2)


# ------------------------------------------------------------------
# 7. Demand-only BLP
#
# Utility:
# x'beta + brand FE + market FE + xi
# - exp(mu + sigma*nu_i)*price + epsilon
#
# PyBLP handles share inversion, FE absorption and two-step GMM.
# ------------------------------------------------------------------

x1 = pyblp.Formulation(
    "0 + weight + fuel + home + power + size",
    absorb="C(market_ids) + C(brd)",
)

# Positive lognormal sensitivity multiplied by negative price.
x2 = pyblp.Formulation("0 + I(-prices)")

problem = pyblp.Problem(
    product_formulations=(x1, x2),
    product_data=d,
    agent_formulation=pyblp.Formulation("0 + one"),
    agent_data=agents,
    rc_types=["log"],
)

blp = problem.solve(
    sigma=np.array([[0.25]]),
    pi=np.array([[
        np.log(max(-iv.params["prices"], 2.0))
    ]]),
    sigma_bounds=(0, 1.5),
    pi_bounds=(-3, 5),
    method="2s",
    W_type="clustered",
    se_type="clustered",
    optimization=pyblp.Optimization(
        "l-bfgs-b",
        {
            "gtol": 1e-7,
            "ftol": 1e-12,
            "maxiter": 220,
            "maxls": 40,
        },
    ),
    iteration=pyblp.Iteration(
        "squarem",
        {
            "atol": 1e-12,
            "max_evaluations": 4000,
        },
    ),
    error_behavior="raise",
)

mu = blp.pi.item()
sigma = blp.sigma.item()
mean_alpha = np.exp(mu + sigma**2 / 2)

share_error = np.max(np.abs(
    blp.compute_shares().ravel()
    - d["shares"].to_numpy()
))

print("\nBLP converged:", blp.converged)
print("BLP maximum share error:", share_error)
print("BLP projected gradient:", blp.projected_gradient_norm)

assert blp.converged
assert share_error < 1e-9
assert blp.projected_gradient_norm < 1e-4
assert not blp.clipped_shares.any()

# Reject solutions hitting the artificial upper/location bounds.
assert -2.999 < mu < 4.999
assert sigma < 1.499


# ------------------------------------------------------------------
# 8. Display compact coefficient tables
# ------------------------------------------------------------------

terms = controls + ["prices"]

coefficients = pd.DataFrame({
    "OLS": ols.params.reindex(terms),
    "OLS_SE": ols.bse.reindex(terms),
    "IV": iv.params.reindex(terms),
    "IV_SE": iv.std_errors.reindex(terms),
    "BLP": np.r_[blp.beta.ravel(), np.nan],
    "BLP_SE": np.r_[blp.beta_se.ravel(), np.nan],
}, index=terms)

price_distribution = pd.DataFrame({
    "estimate": [mu, sigma, mean_alpha],
    "SE": [
        blp.pi_se.item(),
        blp.sigma_se.item(),
        np.nan,
    ],
}, index=["mu", "sigma", "mean_alpha"])

print("\nDemand coefficients:")
print(coefficients.round(6).to_string())

print("\nBLP price-sensitivity distribution:")
print(price_distribution.round(6).to_string())

print("\nMean positive price sensitivity:")
print("OLS:", -ols.params["prices"])
print("IV: ", -iv.params["prices"])
print("BLP:", mean_alpha)


# # ------------------------------------------------------------------
# # 9. Compare with the earlier rival-only estimates
# #
# # These reference checks apply to the supplied cars.csv and this exact
# # specification. They intentionally fail if the specification changes.
# # ------------------------------------------------------------------

# reference_coefficients = np.array([
#     # OLS       OLS_SE     IV        IV_SE     BLP       BLP_SE
#     [-0.024795, 0.313234,  1.504519, 0.583521, 2.143330, 1.279083],
#     [-0.025556, 0.022160, -0.027740, 0.024629, -0.039343, 0.025713],
#     [ 1.221877, 0.048788,  1.210898, 0.048098, 1.225778, 0.047615],
#     [-0.618647, 0.276553,  0.564240, 0.520751, 1.046982, 1.070972],
#     [ 0.089845, 0.061539,  0.103722, 0.065178, 0.169724, 0.122406],
#     [-1.108489, 0.178731, -2.822323, 0.645268, np.nan, np.nan],
# ])

# np.testing.assert_allclose(
#     coefficients.to_numpy(),
#     reference_coefficients,
#     rtol=0,
#     atol=5e-4,
#     equal_nan=True,
# )

# reference_distribution = np.array([
#     [1.511499, 0.905141],
#     [0.259824, 0.303222],
#     [4.689156, np.nan],
# ])

# np.testing.assert_allclose(
#     price_distribution.to_numpy(),
#     reference_distribution,
#     rtol=0,
#     atol=5e-4,
#     equal_nan=True,
# )

# np.testing.assert_allclose(
#     [
#         first_stage.iloc[0]["partial.rsquared"],
#         first_stage.iloc[0]["f.stat"],
#     ],
#     [0.116123270958, 91.6304197068],
#     rtol=0,
#     atol=1e-4,
# )

# print("\nPASS: estimates match the earlier rival-only results.")


# # ------------------------------------------------------------------
# # 10. Save results for inspection and the later merger calculation
# # ------------------------------------------------------------------

# coefficients.to_csv(
#     OUT / "coefficients.csv",
#     index_label="parameter",
# )

# price_distribution.to_csv(
#     OUT / "blp_price_distribution.csv",
#     index_label="parameter",
# )

# first_stage.to_csv(OUT / "first_stage.csv")

# d.to_csv(OUT / "prepared.csv", index=False)

# with (OUT / "blp_results.pkl").open("wb") as file:
#     pickle.dump(blp, file)

# print("Saved results to:", OUT)