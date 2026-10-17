# Gaia M31: weighted stellar trends and field-wide diagnostics
#
# Purpose: Estimate each star's linear magnitude trend and uncertainty, then
# check whether the apparent trends may reflect time-correlated or shared
# variation across the field.
#
# Inputs: `light_curves` is a structured NumPy array with `t_year`, `gmag`,
# and `gmag_err` fields. Rows for each star must be contiguous; `star_starts`
# gives each group's first row and `epoch_counts` gives its row count.
# `reference_year` is the time origin for the trend fits.
#
# Method: Compare a per-star weighted fit with an equivalent vectorized fit.
# The vectorized fit reduces each star's regression to five weighted sums.
#
# Validation: The broader notebook compares the results with
# `numpy.polyfit` and reports numerical differences and runtime.
#
# Interpretation: The shuffled-slope comparison checks whether time ordering
# affects the trend distribution. The common-mode plot summarizes median
# residual magnitude over time. These patterns suggests common variation,
# but these diagnostics alone do not establish its physical or instrumental
# cause.

import numpy as np


def wls(t, mag, sigma):
    """Fit mag = slope * t + intercept; return coefficients and covariance."""
    A = np.column_stack([t, np.ones_like(t)])
    w = 1.0 / sigma**2
    AtWA = A.T @ (A * w[:, None])
    AtWmag = A.T @ (w * mag)

    cov = np.linalg.inv(AtWA)
    coeff = np.linalg.solve(AtWA, AtWmag)
    return coeff, cov


def fit_all_loop(
    light_curves, star_starts, epoch_counts, reference_year,
    error_floor=0.0, perms=None,
):
    """Reference implementation: fit each star separately."""
    n_stars = len(star_starts)
    slopes = np.empty(n_stars)
    slope_errors = np.empty(n_stars)

    for i, (start, n_epoch) in enumerate(zip(star_starts, epoch_counts)):
        rows = light_curves[start : start + n_epoch]
        mag = rows["gmag"]
        sigma = np.sqrt(rows["gmag_err"]**2 + error_floor**2)

        if perms is not None:
            perm = perms[i]
            mag, sigma = mag[perm], sigma[perm]

        t = rows["t_year"] - reference_year
        coeff, cov = wls(t, mag, sigma)
        slopes[i] = coeff[0]
        slope_errors[i] = np.sqrt(cov[0, 0])

    return slopes, slope_errors


def fit_all(
    light_curves, star_starts, reference_year,
    error_floor=0.0, perm_idx=None, mag_override=None,
):
    """Vectorized weighted fit for all stars using grouped sums."""
    t = light_curves["t_year"] - reference_year
    mag = (
        light_curves["gmag"]
        if mag_override is None
        else np.asarray(mag_override)
    )
    sigma2 = light_curves["gmag_err"]**2 + error_floor**2

    if perm_idx is not None:
        mag, sigma2 = mag[perm_idx], sigma2[perm_idx]

    w = 1.0 / sigma2
    sw, swt, swtt, swmag, swtmag = (
        np.add.reduceat(values, star_starts)
        for values in (w, w * t, w * t**2, w * mag, w * t * mag)
    )

    denom = sw * swtt - swt**2
    slopes = (sw * swtmag - swt * swmag) / denom
    slope_errors = np.sqrt(sw / denom)
    return slopes, slope_errors