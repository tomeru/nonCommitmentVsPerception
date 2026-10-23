"""
Null-result analyses for Li, Hammond & Ullman, "Non-commitment in mental imagery
is distinct from perceptual inattention"

- All Bayes factors are computed from first principles (scipy only):
  * correlation BF: exact likelihood of r given rho (Jeffreys 1961; Ly et al. 2016),
    stretched-beta prior of width kappa (kappa=1 is JASP's default, uniform on rho).
  * paired / one-sample t BF: JZS prior, Cauchy scale r (Rouder et al., 2009).
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats, special, integrate

rng = np.random.default_rng(2026)

# ----------------------------------------------------------------------------
# Bayes factor helpers
# ----------------------------------------------------------------------------

def _corr_lik(rho, r, n):
    """Exact (up to a constant) likelihood of sample r given population rho."""
    return ((1 - rho**2) ** ((n - 1) / 2) * (1 - rho * r) ** (-(n - 1.5))
            * special.hyp2f1(0.5, 0.5, n - 0.5, (1 + rho * r) / 2))


def _sbeta_pdf(rho, kappa):
    a = 1 / kappa
    return stats.beta.pdf((rho + 1) / 2, a, a) / 2


def bf_corr(r, n, kappa=1.0, side="two", prior=None):
    """BF10 for rho != 0 (side='two'), rho > 0 ('plus'). Custom prior pdf optional."""
    L0 = _corr_lik(0.0, r, n)
    if prior is None:
        pdf = lambda x: _sbeta_pdf(x, kappa)
        lo, hi, norm = (-1, 1, 1.0) if side == "two" else (0, 1, 0.5)
    else:
        pdf, lo, hi = prior
        norm = integrate.quad(pdf, lo, hi)[0]
    num = integrate.quad(lambda x: _corr_lik(x, r, n) * pdf(x), lo, hi, limit=200)[0] / norm
    return num / L0


def bf_jzs_t(t, n, scale=np.sqrt(2) / 2):
    """JZS BF10 for a one-sample / paired t test (Rouder et al., 2009)."""
    v = n - 1
    def f(g):
        return ((1 + n * g * scale**2) ** -0.5
                * (1 + t**2 / ((1 + n * g * scale**2) * v)) ** (-(v + 1) / 2)
                * (2 * np.pi) ** -0.5 * g ** -1.5 * np.exp(-1 / (2 * g)))
    num = integrate.quad(f, 0, np.inf, limit=200)[0]
    den = (1 + t**2 / v) ** (-(v + 1) / 2)
    return num / den


def fisher_ci(r, n, level=0.90):
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    q = stats.norm.ppf(1 - (1 - level) / 2)
    return np.tanh(z - q * se), np.tanh(z + q * se)


def tost_r(r, n, bound):
    """Two one-sided tests for -bound < rho < bound (Fisher z). Returns max p."""
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    p_lo = 1 - stats.norm.cdf((z - np.arctanh(-bound)) / se)
    p_hi = stats.norm.cdf((z - np.arctanh(bound)) / se)
    return max(p_lo, p_hi)


def min_effect_test(r, n, rho0):
    """One-sided test of H0: rho >= rho0 vs H1: rho < rho0 (inferiority)."""
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return stats.norm.cdf((z - np.arctanh(rho0)) / se)


def mdes_r(n, power, alpha=0.05, sides=2):
    """Smallest |rho| detectable with given power (Fisher z approximation)."""
    za = stats.norm.ppf(1 - alpha / sides)
    zb = stats.norm.ppf(power)
    return np.tanh((za + zb) / np.sqrt(n - 3))


def tost_bound_r(n, power, alpha=0.05):
    """Smallest equivalence bound attainable with given power when true rho = 0."""
    za = stats.norm.ppf(1 - alpha)
    zb = stats.norm.ppf(1 - (1 - power) / 2)
    return np.tanh((za + zb) / np.sqrt(n - 3))


def fmt_bf(b):
    return f"{b:.2f}" if 0.01 <= b < 100 else f"{b:.2e}"


# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
f1, f2, f3 = sys.argv[1:4]
e1, e2, e3 = pd.read_csv(f1), pd.read_csv(f2), pd.read_csv(f3)

SCENES = ["Ball", "Supermarket", "Zoo", "Candle", "Shoe"]
BLOCKS = {
    "Ball": ["Q1", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10"],
    "Supermarket": ["Q11", "Q12", "Q13", "Q14", "Q15", "Q16", "Q17", "Q18.1", "Q19"],
    "Zoo": ["Q20", "Q21", "Q22", "Q23", "Q24", "Q25", "Q26", "Q27", "Q28"],
    "Candle": ["Q29", "Q31", "Q32", "Q33", "Q34", "Q35", "Q36", "Q37", "Q30"],
    "Shoe": ["Q38", "Q40", "Q41", "Q42", "Q43", "Q44", "Q45", "Q46", "Q39"],
}

out = []
def P(s=""):
    print(s)
    out.append(s)

# ============================================================================
# PART A. Perception-imagination correlation (Exp 1 & 2)
# ============================================================================
P("=" * 78)
P("PART A. Item-level correlation: perception non-noticing vs imagery non-commitment")
P("=" * 78)

# Reported item-level results. r recovered from R^2; sign from fitted slopes in
# Figs 3 and 5 (Candle negative). n = 45 properties overall, 9 per scene.
reported = {
    "Exp1": {"All": (0.043, +1, 45, 0.17), "Ball": (0.00079, +1, 9, 0.94),
             "Supermarket": (0.48, +1, 9, 0.038), "Zoo": (0.24, +1, 9, 0.18),
             "Candle": (0.20, -1, 9, 0.23), "Shoe": (0.40, +1, 9, 0.068)},
    "Exp2": {"All": (0.044, +1, 45, 0.17), "Ball": (0.0082, +1, 9, 0.82),
             "Supermarket": (0.51, +1, 9, 0.03), "Zoo": (0.23, +1, 9, 0.19),
             "Candle": (0.21, -1, 9, 0.22), "Shoe": (0.27, +1, 9, 0.15)},
}

P("\nA0. Sanity check: recovered r reproduces the reported p values")
for ex, rows in reported.items():
    for sc, (R2, sgn, n, p_rep) in rows.items():
        r = sgn * np.sqrt(R2)
        t = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
        p = 2 * stats.t.sf(abs(t), n - 2)
        P(f"  {ex} {sc:12s} n={n:2d} r={r:+.3f}  p(recomputed)={p:.3f}  p(reported)={p_rep}")

# ---- A1. Noise ceiling: what Hypothesis 1 actually predicts ------------------
P("\nA1. Reliability of the perception measure (raw data) -> H1-predicted correlation")

def item_props(df):
    props, ns = [], []
    for sc in SCENES:
        sub = df[df[BLOCKS[sc][0]].notna()]
        for q in BLOCKS[sc]:
            props.append((sub[q] == "No").mean())
            ns.append(sub[q].notna().sum())
    return np.array(props), np.array(ns)

p1, n1 = item_props(e1)
p2, n2 = item_props(e2)
P(f"  Exp1 '% not noticed' across 45 items: mean={100*p1.mean():.1f}%, SD={100*p1.std(ddof=1):.1f}, range {100*p1.min():.0f}-{100*p1.max():.0f}%")
P(f"  Exp2 '% not noticed' across 45 items: mean={100*p2.mean():.1f}%, SD={100*p2.std(ddof=1):.1f}, range {100*p2.min():.0f}-{100*p2.max():.0f}%")
r12 = np.corrcoef(p1, p2)[0, 1]
lo12, hi12 = fisher_ci(r12, 45, 0.95)
P(f"  Exp1 vs Exp2 perception item profile (independent samples, different instructions):")
P(f"    r = {r12:.3f}  [95% CI {lo12:.3f}, {hi12:.3f}],  BF10 = {fmt_bf(bf_corr(r12, 45))}")


def split_half(df, n_iter=2000):
    rs = []
    groups = {sc: df[df[BLOCKS[sc][0]].notna()][BLOCKS[sc]].eq("No").to_numpy() for sc in SCENES}
    for _ in range(n_iter):
        a, b = [], []
        for sc, m in groups.items():
            idx = rng.permutation(len(m))
            h = len(m) // 2
            a.extend(m[idx[:h]].mean(0)); b.extend(m[idx[h:]].mean(0))
        r = np.corrcoef(a, b)[0, 1]
        rs.append(2 * r / (1 + r))  # Spearman-Brown
    return np.mean(rs), np.percentile(rs, [2.5, 97.5])

sh1, ci1 = split_half(e1)
sh2, ci2 = split_half(e2)
P(f"  Split-half reliability (Spearman-Brown, 2000 random splits):")
P(f"    Exp1 = {sh1:.3f} [{ci1[0]:.3f}, {ci1[1]:.3f}]   Exp2 = {sh2:.3f} [{ci2[0]:.3f}, {ci2[1]:.3f}]")

# Binomial-sampling reliability for a measure with this spread and n per item
def binom_rel(p, n):
    err = np.mean(p * (1 - p) / n)
    return 1 - err / p.var(ddof=1)
P(f"  Binomial-sampling reliability check: Exp1 = {binom_rel(p1, n1):.3f}, Exp2 = {binom_rel(p2, n2):.3f}")

rel_perc = sh1
P("\n  If Hypothesis 1 held (imagery omissions are the same latent profile as perceptual")
P("  omissions), the expected observed r = rho_true * sqrt(rel_perception * rel_imagery).")
P("  Imagery reliability is unknown (raw Bigelow data not available here); the imagery")
P("  measure has a similar design (binary report, ~200 per scene), so a range is shown:")
ceilings = {}
for rel_im in [0.95, 0.90, 0.80, 0.70, 0.60]:
    c = np.sqrt(rel_perc * rel_im)
    ceilings[rel_im] = c
    P(f"    rel_imagery = {rel_im:.2f}  ->  H1-predicted r (rho_true = 1) = {c:.2f};  if rho_true = .7: {0.7*c:.2f}")
P(f"  Empirical benchmark: changing the *perception* task (adding the text prime) still")
P(f"  left r = {r12:.2f} between the two perception profiles.")

# ---- A2. Bayes factors --------------------------------------------------------
P("\nA2. Bayes factors (H0: rho = 0). BF01 > 1 favors the null.")
P("    two-sided: stretched-beta prior; one-sided BF0+: H1 restricted to rho > 0 (H1 is directional)")
hdr = f"  {'':5s}{'Scene':12s}{'n':>3s} {'r':>7s} | {'BF01 k=1':>9s} {'BF01 k=.5':>9s} {'BF01 k=1/3':>10s} | {'BF0+ k=1':>9s} {'BF0+ k=1/3':>10s}"
P(hdr)
for ex, rows in reported.items():
    for sc, (R2, sgn, n, _) in rows.items():
        r = sgn * np.sqrt(R2)
        b = [1 / bf_corr(r, n, k) for k in (1, 0.5, 1 / 3)]
        bp = [1 / bf_corr(r, n, k, side="plus") for k in (1, 1 / 3)]
        P(f"  {ex:5s}{sc:12s}{n:3d} {r:+7.3f} | {fmt_bf(b[0]):>9s} {fmt_bf(b[1]):>9s} {fmt_bf(b[2]):>10s} | {fmt_bf(bp[0]):>9s} {fmt_bf(bp[1]):>10s}")

P("\n  Robustness of overall BF01 (Exp1, r = .207, n = 45) to prior width kappa:")
r_all = np.sqrt(0.043)
P("    " + "  ".join(f"k={k:.2f}: {1/bf_corr(r_all, 45, k):.2f}" for k in [0.1, 0.2, 0.33, 0.5, 0.75, 1.0, 1.5, 2.0]))

P("\n  Theory-informed test: H_tight (Hypothesis 1) vs H0 and vs the observed data.")
P("  H_tight: rho ~ Uniform(0.5, 0.9), i.e. a strong positive correlation, below the ceiling.")
for ex in ["Exp1", "Exp2"]:
    R2 = reported[ex]["All"][0]; r = np.sqrt(R2)
    bf_tight = bf_corr(r, 45, prior=(lambda x: 1.0, 0.5, 0.9))
    bf_small = bf_corr(r, 45, prior=(lambda x: 1.0, 0.0, 0.3))
    P(f"    {ex}: BF(H0 : H_tight) = {fmt_bf(1/bf_tight)};  BF(H_small[0,.3] : H_tight) = {fmt_bf(bf_small/bf_tight)};  BF(H_small : H0) = {fmt_bf(bf_small)}")

# ---- A3. Frequentist equivalence / minimum-effect tests ---------------------
P("\nA3. Equivalence (TOST) and minimum-effect (inferiority) tests, Fisher z")
for ex in ["Exp1", "Exp2"]:
    R2 = reported[ex]["All"][0]; r = np.sqrt(R2)
    lo90, hi90 = fisher_ci(r, 45, 0.90)
    lo95, hi95 = fisher_ci(r, 45, 0.95)
    P(f"  {ex}: r = {r:.3f}, 95% CI [{lo95:.3f}, {hi95:.3f}], 90% CI [{lo90:.3f}, {hi90:.3f}]")
    for bnd in [0.1, 0.2, 0.3, 0.4, 0.5]:
        P(f"     TOST |rho| < {bnd:.1f}: p = {tost_r(r, 45, bnd):.4f}  {'EQUIVALENT' if tost_r(r,45,bnd)<.05 else 'not established'}")
    P(f"     -> smallest bound at which equivalence holds (alpha=.05): |rho| < {max(abs(lo90), hi90):.3f}")
    for rho0 in [0.5, 0.6, 0.7, 0.8]:
        P(f"     Inferiority test H0: rho >= {rho0:.1f}:  p = {min_effect_test(r, 45, rho0):.2e}")

P("\n  Per-scene (n = 9) 90% CIs, to show what single scenes can and cannot support:")
for ex, rows in reported.items():
    for sc, (R2, sgn, n, _) in rows.items():
        if sc == "All":
            continue
        r = sgn * np.sqrt(R2)
        lo, hi = fisher_ci(r, n, 0.90)
        P(f"    {ex} {sc:12s} r={r:+.2f}  90% CI [{lo:+.2f}, {hi:+.2f}]")

# ---- A4. Sensitivity power ---------------------------------------------------
P("\nA4. Sensitivity power analysis (unit of analysis = 45 properties, not 3,479 people)")
for pw in [0.80, 0.90, 0.95]:
    P(f"  power {pw:.2f}: min detectable |rho| two-sided = {mdes_r(45, pw):.3f}, one-sided = {mdes_r(45, pw, sides=1):.3f};"
      f"  smallest TOST bound attainable if rho = 0: {tost_bound_r(45, pw):.3f}")
for rho0 in [0.5, 0.6, 0.7]:
    za = stats.norm.ppf(0.95); se = 1 / np.sqrt(42)
    pw = stats.norm.cdf(np.arctanh(rho0) / se - za)
    pw2 = stats.norm.cdf((np.arctanh(rho0) - np.arctanh(0.2)) / se - za)
    P(f"  Power of the inferiority test vs rho0 = {rho0:.1f}: {pw:.3f} if true rho = 0; {pw2:.3f} if true rho = .2")
P("  Items needed for 80%/90% power to detect rho = .30 (two-sided): "
  + ", ".join(str(int(np.ceil(((stats.norm.ppf(.975) + stats.norm.ppf(p)) / np.arctanh(0.3)) ** 2 + 3))) for p in [0.8, 0.9]))
P("  Items needed for TOST |rho| < .30 at 80%/90% power (if rho = 0): "
  + ", ".join(str(int(np.ceil(((stats.norm.ppf(.95) + stats.norm.ppf(1 - (1 - p) / 2)) / np.arctanh(0.3)) ** 2 + 3))) for p in [0.8, 0.9]))

# ============================================================================
# PART B. Experiment 3 null / equivalence claims
# ============================================================================
P("\n" + "=" * 78)
P("PART B. Experiment 3")
P("=" * 78)
CAT = {1: "Spatial", 2: "Spatial", 3: "Spatial", 4: "Entity", 5: "Identity",
       6: "Identity", 7: "Surface", 8: "Surface", 9: "Surface"}
rows = []
for _, r in e3.iterrows():
    s = r.scene
    for k in range(1, 10):
        rows.append(dict(pid=r.participant, scene=s, cond=r.condition, item=k, cat=CAT[k],
                         yes=int(pd.notna(r[f"{s}Q_0_GROUP_{k}"])), invalid=r.invalid))
L = pd.DataFrame(rows)
L["ms"] = L.cond.str.replace("ms", "").astype(int)
P(f"  N = {L.pid.nunique()} participants; flagged 'invalid' = {int(e3.invalid.fillna(0).sum())} (kept, matching the paper's N = 1,465)")
P("  % committed by category (check vs Fig 8):")
P(L.groupby(["scene", "cat"]).yes.mean().unstack().mul(100).round(1).to_string())

pp = L.groupby(["pid", "scene", "ms", "cat"]).yes.mean().unstack().reset_index()
pp["logt"] = np.log2(pp.ms)
pp["surf_minus_core"] = pp.Surface - (pp.Identity + pp.Spatial) / 2

def partial_r_time(df, col):
    """Correlation of log2(time) with outcome after removing scene means (pooled)."""
    y = df[col] - df.groupby("scene")[col].transform("mean")
    x = df.logt - df.groupby("scene").logt.transform("mean")
    r = np.corrcoef(x, y)[0, 1]
    n = len(df) - 2  # 2 df spent on scene means (approximation)
    return r, n

P("\nB1. Effect of imagination time (250-2000 ms, log2 scale) on commitment")
P("    r = within-scene correlation of log2(time) with participant-level % committed")
P(f"    {'outcome':22s}{'scene':7s}{'n':>6s}{'r':>8s}{'95% CI':>18s}{'p':>8s}{'BF01':>8s}{'BF0+':>8s}{'TOST|r|<.1 p':>14s}")
for col in ["Surface", "surf_minus_core", "Identity", "Spatial", "Entity"]:
    for sc in ["All", "Drop", "Push", "Pull"]:
        df = pp if sc == "All" else pp[pp.scene == sc]
        if sc == "All":
            r, n = partial_r_time(df, col)
        else:
            r, n = np.corrcoef(df.logt, df[col])[0, 1], len(df)
        lo, hi = fisher_ci(r, n, 0.95)
        t = r * np.sqrt(n - 2) / np.sqrt(1 - r**2); p = 2 * stats.t.sf(abs(t), n - 2)
        P(f"    {col:22s}{sc:7s}{n:6d}{r:+8.3f}   [{lo:+.3f}, {hi:+.3f}]{p:8.3f}{1/bf_corr(r, n):8.2f}{1/bf_corr(r, n, side='plus'):8.2f}{tost_r(r, n, 0.1):14.4f}")

# Omnibus: one-way ANOVA across 4 time conditions on surface - core, BIC-approx BF
P("\n  Omnibus 4-level timing effect (scene as covariate), BIC-approximated BF01 (Wagenmakers, 2007):")
for col in ["Surface", "surf_minus_core", "Entity"]:
    y = pp[col].to_numpy()
    Xs = pd.get_dummies(pp.scene, drop_first=True).to_numpy(float)
    Xt = pd.get_dummies(pp.ms, drop_first=True).to_numpy(float)
    X0 = np.column_stack([np.ones(len(y)), Xs]); X1 = np.column_stack([X0, Xt])
    def bic(X):
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        rss = np.sum((y - X @ b) ** 2); n = len(y)
        return n * np.log(rss / n) + X.shape[1] * np.log(n), rss
    b0, rss0 = bic(X0); b1, rss1 = bic(X1)
    F = ((rss0 - rss1) / 3) / (rss1 / (len(y) - X1.shape[1]))
    pF = stats.f.sf(F, 3, len(y) - X1.shape[1])
    P(f"    {col:16s} F(3,{len(y)-X1.shape[1]}) = {F:.2f}, p = {pF:.3f}, BF01 = {np.exp((b1 - b0) / 2):.1f}")

P("\n  Sensitivity: smallest |r| (time x outcome) detectable with 80/90/95% power")
for nn, lab in [(len(pp), "pooled"), (int(len(pp) / 3), "per scene")]:
    P(f"    {lab:10s} n~{nn}: " + ", ".join(f"{mdes_r(nn, w):.3f}" for w in [0.8, 0.9, 0.95])
      + f";  TOST bound attainable at 80%: {tost_bound_r(nn, 0.8):.3f}")
# translate r to pp of commitment per doubling of time
sd_x = pp.logt.std(); sd_y = pp.Surface.std()
P(f"    (For Surface, r = .10 corresponds to a slope of ~{100*0.10*sd_y/sd_x:.1f} percentage points per doubling of time,")
P(f"     i.e. ~{3*100*0.10*sd_y/sd_x:.1f} pp from 250 to 2000 ms.)")

# ---- B2. Person/Table/Floor vs Surface "similar" claim ----------------------
P("\nB2. 'Non-essential entity commitment is similar to Surface' (Push p=.79, Pull p=.86)")
P("    paired within participant: entity item (0/1) minus mean of the 3 surface items")
P(f"    {'scene':6s}{'n':>5s}{'diff(pp)':>10s}{'95% CI':>18s}{'90% CI':>18s}{'BF01':>8s}{'TOST ±5pp':>11s}{'TOST ±10pp':>12s}")
for sc in ["Drop", "Push", "Pull"]:
    d = pp[pp.scene == sc]; diff = (d.Entity - d.Surface).to_numpy(); n = len(diff)
    m, se = diff.mean(), diff.std(ddof=1) / np.sqrt(n)
    t = m / se
    c95 = m + np.array([-1, 1]) * stats.t.ppf(.975, n - 1) * se
    c90 = m + np.array([-1, 1]) * stats.t.ppf(.95, n - 1) * se
    def tost(bd):
        return max(stats.t.sf((m + bd) / se, n - 1), stats.t.cdf((m - bd) / se, n - 1))
    P(f"    {sc:6s}{n:5d}{100*m:+10.1f}   [{100*c95[0]:+.1f}, {100*c95[1]:+.1f}]   [{100*c90[0]:+.1f}, {100*c90[1]:+.1f}]{1/bf_jzs_t(t, n):8.2f}{tost(0.05):11.4f}{tost(0.10):12.4f}")
    dz_mdes = (stats.norm.ppf(.975) + stats.norm.ppf(.8)) / np.sqrt(n)
    P(f"          sensitivity: MDES dz = {dz_mdes:.3f} (~{100*dz_mdes*diff.std(ddof=1):.1f} pp) at 80% power")
P("    Reference gap for SESOI: Identity - Surface = "
  + ", ".join(f"{sc} {100*(pp[pp.scene==sc].Identity-pp[pp.scene==sc].Surface).mean():.0f} pp" for sc in ["Drop", "Push", "Pull"]))

with open("null_result_analyses_output.txt", "w") as fh:
    fh.write("\n".join(out))
