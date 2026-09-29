"""Helper functions for Lab 1 (Gaia M31-field light curves).

All functions take in data.
Conventions
time = t
magnitude = m
error = e
weight = w
design matric = A
covariance = C
params = beta
"""
import numpy as np

LC_DTYPE = [("source_id", "i8"), ("t_year", "f8"), ("gmag", "f8"), ("gmag_err", "f8")]


# ---------------------------------------------------------------- loading
def load_lightcurves(path):
    """Read the CSV and sort by star. Source_id is read as int64
    """
    lc = np.genfromtxt(path, delimiter=",", names=True, dtype=LC_DTYPE)
    lc = lc[np.lexsort((lc["t_year"], lc["source_id"]))]
    ids, star_start, n_epochs = np.unique(lc["source_id"], return_index=True, return_counts=True)
    return lc, ids, star_start, n_epochs


def star_rows(lc, star_start, n_epochs, i):
    """Rows of star i (a view into lc)."""
    return lc[star_start[i]:star_start[i] + n_epochs[i]]


# ---------------------------------------------------------------- fitting
def wls(t, m, e):
    """Weighted least squares for m = slope * t + intercept via the normal equations.

    beta = (A^T W A)^-1 A^T W m,  Cov(beta) = (A^T W A)^-1,  W = diag(1/e^2).
    Returns beta = [slope, intercept] and the 2x2 covariance C.
    """
    A = np.column_stack([t, np.ones_like(t)])
    w = 1.0 / e**2
    ATWA = A.T @ (A * w[:, None])
    C = np.linalg.inv(ATWA)
    beta = np.linalg.solve(ATWA, A.T @ (w * m))
    return beta, C


def fit_all_loop(lc, star_start, n_epochs, t0, f=0.0, perms=None):
    """Task 5 reference: one wls() call per star.
    Returns slopes (mag/yr) and their 1-sigma errors.
    """
    n_stars = len(star_start)
    slopes, sig = np.empty(n_stars), np.empty(n_stars)
    for i, (a, k) in enumerate(zip(star_start, n_epochs)):
        rows = lc[a:a + k]
        m = rows["gmag"]
        e = np.sqrt(rows["gmag_err"]**2 + f**2)
        if perms is not None:
            m, e = m[perms[i]], e[perms[i]]
        beta, C = wls(rows["t_year"] - t0, m, e)
        slopes[i], sig[i] = beta[0], np.sqrt(C[0, 0])
    return slopes, sig


def fit_all(lc, star_start, t0, f=0.0, perm_idx=None, mags=None):
    """Same fits as fit_all_loop, solved for all stars at once.
    The 2x2 normal equations only need five weighted sums per star:
      S = sum w, St = sum w t, Stt = sum w t^2, Sm = sum w m, Stm = sum w t m
      slope = (S Stm - St Sm) / D,  var(slope) = S / D,  D = S Stt - St^2
    np.add.reduceat computes each sum over every star's contiguous block in one call.
    perm_idx : global index array from perms_to_index() (shuffles every star at once).
    mags     : optional replacement magnitudes, same order as lc (e.g. zero-point corrected).

    I used AI to figure this part out
    """
    t = lc["t_year"] - t0
    m = lc["gmag"] if mags is None else np.asarray(mags)
    e2 = lc["gmag_err"]**2 + f**2
    if perm_idx is not None:
        m, e2 = m[perm_idx], e2[perm_idx]
    w = 1.0 / e2
    S, St, Stt, Sm, Stm = (np.add.reduceat(x, star_start) for x in (w, w * t, w * t * t, w * m, w * t * m))
    D = S * Stt - St**2
    return (S * Stm - St * Sm) / D, np.sqrt(S / D)


def perms_to_index(star_start, perms):
    """Turn per-star permutations into one index array over all rows of lc."""
    return np.concatenate([a + p for a, p in zip(star_start, perms)])


def make_perms(n_epochs, seed):
    """One random permutation per star."""
    rng = np.random.default_rng(seed)
    return [rng.permutation(k) for k in n_epochs]


def slopes_after(lc, star_start, n_epochs, t0, tmin, f, min_n=10, min_base=1.0):
    """Weighted slopes using only epochs with t >= tmin.
    Stars left with fewer than min_n epochs or less than min_base years of baseline get NaN.
    """
    out = np.full(len(star_start), np.nan)
    for i, (a, k) in enumerate(zip(star_start, n_epochs)):
        rows = lc[a:a + k]
        rows = rows[rows["t_year"] >= tmin]
        if len(rows) < min_n or np.ptp(rows["t_year"]) < min_base:
            continue
        e = np.sqrt(rows["gmag_err"]**2 + f**2)
        out[i] = wls(rows["t_year"] - t0, rows["gmag"], e)[0][0]
    return out


# ---------------------------------------------------------------- statistics
def rwidth(x):
    """Robust Gaussian width: 1.4826 * median absolute deviation."""
    x = np.asarray(x)
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def boot_median(x, n_boot=5000, seed=1, chunk=1000):
    """Median of x and its bootstrap standard error (resampling elements of x).
    Resamples in chunks so memory stays ~chunk * len(x) * 8 bytes.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    meds = np.empty(n_boot)
    for lo in range(0, n_boot, chunk):
        hi = min(lo + chunk, n_boot)
        idx = rng.integers(0, len(x), size=(hi - lo, len(x)))
        meds[lo:hi] = np.median(x[idx], axis=1)
    return np.median(x), meds.std(ddof=1)


# ---------------------------------------------------------------- plotting
def step_hist(ax, values, bins, n_total, **kwargs):
    """Step histogram as a density over all n_total values.
    Values outside the bin range are placed in the edge bins instead of being dropped,
    and the normalisation uses n_total so every star counts.
    """
    width = bins[1] - bins[0]
    clipped = np.clip(values, bins[0] + 1e-9, bins[-1] - 1e-9)
    weights = np.full(len(values), 1.0 / (n_total * width))
    ax.hist(clipped, bins, weights=weights, histtype="step", **kwargs)
