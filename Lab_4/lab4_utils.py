"""Helper functions for Lab 4: dormant black hole candidates from Gaia DR3 summary statistics.

Replicates the method of Muller-Horn et al. (A&A 709, A62, 2026; arXiv:2510.05982):
infer the mass of an unseen companion from two numbers Gaia publishes for every star,
the astrometric RUWE and the radial-velocity scatter (radial_velocity_error), using a
forward model + MCMC.

Status of each piece:
    [DONE]  plumbing: config, ADQL query builders, cached archive download, cost estimate
    [TODO]  science: every function that raises NotImplementedError. Each docstring gives
            the equation / spec from the paper so you can implement and test it on its own.

Conventions
    theta    = MCMC parameter vector, ordered as PARAM_NAMES below
    star     = dict (or table row) of one star's Gaia columns + its primary mass "mstar"
    P        = orbital period [days]
    m_co     = companion (compact object) mass [Msun]
    mstar    = primary (visible star) mass [Msun]
    inc      = inclination [rad], 0 = face-on, pi/2 = edge-on
    e        = eccentricity
    Omega    = longitude of ascending node [rad]
    omega    = argument of periastron [rad]
    phi_p    = periastron phase, Tp / P in [0, 1)
    rng      = np.random.Generator
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

# =============================================================== config (from the paper)
DATA_DIR = Path(__file__).resolve().parent / "data"

# Likelihood: fractional scatter of the forward model, in log space (paper Sect. 3)
EPS_RUWE = 0.14          # scatter of ln(ruwe_pred / ruwe_obs), calibrated on 158,690 known orbits
EPS_SIGRV = 0.15         # scatter of ln(sigma_pred / sigma_obs), calibrated on SB1s
SIGRV_FLOOR = 0.113      # km/s, systematic floor in the radial_velocity_error model
SIGRV_SB1_SCALE = 1.07   # empirical rescaling of predicted sigma_RV found on SB1 systems

# Priors (paper Table: priors)
LOGP_RANGE = (1.0, 4.0)        # log10(P / day), uniform -> P in [10, 10^4] d
LOGMCO_RANGE = (np.log10(0.05), np.log10(30.0))   # log10(M_CO / Msun), uniform
E_RANGE = (0.0, 0.95)
FLUX_RATIO = 0.0               # dark companion: no light from the companion

# Sampler (paper Sect. 5)
N_WALKERS = 32
N_STEPS = 2500
N_BURN = 500
SEC_PER_STAR = 10.0            # paper's quoted cost per star per CPU core

# Selection cuts (paper Sect. 6)
CUTS_PAPER = dict(
    abs_g_max=3.0,             # M_G = G + 5 log10(parallax/100) < 3  (giants)
    bp_rp_min=1.0,
    ruwe_min=1.4,
    rv_amp_min=20.0,           # km/s, rv_amplitude_robust
    grvs_max=12.0,
    teff_max=6000.0,           # paper uses Zhang+2023 Teff; we substitute GSP-Phot (deviation)
)
M_CO_THRESHOLD = 3.0           # Msun: candidate if posterior-weighted mean M_CO >= 3

# Funnel numbers reported in the paper, for comparison in Task 2
PAPER_FUNNEL = {"step1_giants": 3_375_482, "step2_ruwe_rvamp": 37_421, "step3_zhang": 21_028,
                "mco_ge_3": 1_410, "final_arxiv_v1": 556, "final_published": 389}

PARAM_NAMES = ["logP", "logMco", "cos_i", "e", "Omega", "omega", "phi_p"]
NDIM = len(PARAM_NAMES)

# Known Gaia black holes for validation. Verify IDs against the discovery papers.
# BH1 is a main-sequence (Sun-like) star, so it does NOT pass the giant cut; run it directly.
KNOWN_BHS = {
    "Gaia BH1": 4373465352415301632,
    "Gaia BH2": 5870569352746779008,
    "Gaia BH3": 4318465066420528000,
}

G_SI = 6.674e-11
MSUN = 1.989e30
DAY = 86400.0
AU_M = 1.496e11

# =============================================================== data ingestion  [DONE]
GAIA_COLUMNS = [
    "g.source_id", "g.ra", "g.dec", "g.parallax", "g.parallax_over_error", "g.pmra", "g.pmdec",
    "g.phot_g_mean_mag", "g.phot_bp_mean_mag", "g.bp_rp", "g.ruwe", "g.astrometric_n_good_obs_al",
    "g.ipd_frac_multi_peak", "g.radial_velocity", "g.radial_velocity_error", "g.rv_nb_transits",
    "g.rv_amplitude_robust", "g.rv_method_used", "g.grvs_mag", "g.phot_g_n_obs",
    "g.phot_g_mean_flux_over_error", "g.non_single_star",
    # Simplification of the paper's XGBoost masses: Gaia's own FLAME / GSP-Phot parameters
    "ap.mass_flame", "ap.radius_flame", "ap.radius_gspphot", "ap.teff_gspphot",
    "ap.logg_gspphot", "ap.mh_gspphot",
]


def _where_clause(cuts, include_step2=True):
    """ADQL WHERE clause implementing the paper's step-1 (and optionally step-2) cuts."""
    c = cuts
    clauses = [
        "g.parallax > 0",
        f"g.phot_g_mean_mag + 5*LOG10(g.parallax/100) < {c['abs_g_max']}",
        f"g.bp_rp > {c['bp_rp_min']}",
        "g.ipd_frac_multi_peak = 0",
        "g.rv_method_used = 1",
        f"g.grvs_mag <= {c['grvs_max']}",
    ]
    if include_step2:
        clauses += [f"g.ruwe > {c['ruwe_min']}", f"g.rv_amplitude_robust > {c['rv_amp_min']}"]
    return "\n  AND ".join(clauses)


def build_parent_query(cuts=None, include_step2=True, extra_where=None):
    """ADQL that does the selection on the archive side, so you download ~10^4 rows, not 10^9.

    LEFT JOIN on gaiadr3.astrophysical_parameters brings in FLAME mass and GSP-Phot Teff.
    extra_where: optional extra ADQL condition string (e.g. a source_id range for chunking).
    """
    cuts = {**CUTS_PAPER, **(cuts or {})}
    where = _where_clause(cuts, include_step2)
    if extra_where:
        where += f"\n  AND {extra_where}"
    return (
        "SELECT " + ", ".join(GAIA_COLUMNS) + "\n"
        "FROM gaiadr3.gaia_source AS g\n"
        "LEFT JOIN gaiadr3.astrophysical_parameters AS ap USING (source_id)\n"
        f"WHERE {where}"
    )


def build_count_query(cuts=None, include_step2=False):
    """COUNT(*) version, for checking funnel numbers without downloading 3.4M rows."""
    cuts = {**CUTS_PAPER, **(cuts or {})}
    return ("SELECT COUNT(*) AS n\nFROM gaiadr3.gaia_source AS g\n"
            f"WHERE {_where_clause(cuts, include_step2)}")


def build_id_query(source_ids):
    """Fetch the same columns for a list of source_ids (known BHs, paper's catalogue subset)."""
    ids = ", ".join(str(int(s)) for s in source_ids)
    return ("SELECT " + ", ".join(GAIA_COLUMNS) + "\n"
            "FROM gaiadr3.gaia_source AS g\n"
            "LEFT JOIN gaiadr3.astrophysical_parameters AS ap USING (source_id)\n"
            f"WHERE g.source_id IN ({ids})")


def fetch_gaia(query, cache_path, overwrite=False, verbose=True):
    """Run an ADQL query on the ESA Gaia archive (async job) and cache the result as .ecsv.

    Re-running the notebook reads the cached file instead of hitting the archive again.
    Anonymous async jobs are limited to ~3M rows; the step-2 sample is ~4x10^4 rows.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not overwrite:
        from astropy.table import Table
        if verbose:
            print(f"loading cached {cache_path.name}")
        return Table.read(cache_path)

    from astroquery.gaia import Gaia
    Gaia.ROW_LIMIT = -1
    t0 = time.time()
    job = Gaia.launch_job_async(query)
    tbl = job.get_results()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tbl.write(cache_path, overwrite=True)
    if verbose:
        print(f"{len(tbl):,} rows in {time.time() - t0:.0f} s -> {cache_path.name}")
    return tbl


def estimate_cost(n_stars, n_cores=8, sec_per_star=SEC_PER_STAR):
    """CPU budget for running the MCMC on n_stars."""
    cpu_h = n_stars * sec_per_star / 3600
    wall_h = cpu_h / n_cores
    print(f"{n_stars:,} stars x {sec_per_star:.0f} s = {cpu_h:,.1f} CPU-hours "
          f"-> {wall_h:,.1f} h wall-clock on {n_cores} cores")
    return cpu_h, wall_h


def row_to_star(row, mstar_col="mass_flame"):
    """Convert one table row to the plain dict the forward model uses (adds 'mstar')."""
    star = {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in row.colnames}
    star["mstar"] = star.get(mstar_col)
    return star


# =============================================================== primary mass  [TODO]
def estimate_primary_mass(tbl):
    """Primary mass M_* for every star [Msun]. The paper FIXES M_* (it is not sampled).

    Paper: XGBoost regression on Zhang et al. (2023) XP parameters (Teff, logg, [M/H]).
    Suggested lab version: use ap.mass_flame where available; for missing values, fall back
    to a simple estimate (e.g. a median giant mass, or an isochrone lookup in Teff/logg),
    and state this deviation. Returns an array of length len(tbl) and a boolean mask of
    which stars used the fallback.
    """
    raise NotImplementedError


# =============================================================== Keplerian physics  [TODO]
def solve_kepler(M, e, tol=1e-10, maxiter=50):
    """Eccentric anomaly E from mean anomaly M:  M = E - e sin E  (Newton-Raphson, vectorised).

    Start from E = M (or M + e sin M for e > 0.8) and iterate
        E <- E - (E - e sin E - M) / (1 - e cos E)
    until max |dE| < tol.
    """
    raise NotImplementedError


def true_anomaly(E, e):
    """nu = 2 arctan( sqrt((1+e)/(1-e)) tan(E/2) )  (use arctan2 for the right quadrant)."""
    raise NotImplementedError


def semi_amplitude(mstar, m_co, P, e, inc):
    """RV semi-amplitude of the visible star [km/s]  (paper Sect. 2):

        K_* = M_CO / (M_* + M_CO)^(2/3) * (2 pi G / P)^(1/3) * (1 - e^2)^(-1/2) * sin i

    Work in SI (masses in kg, P in s) and convert m/s -> km/s at the end.
    Sanity check: Sun + Jupiter (m_co = 9.55e-4, P = 4332.6 d, e = 0.049, i = 90 deg) -> ~12.5 m/s.
    """
    raise NotImplementedError


def rv_curve(t, P, e, omega, phi_p, K, vz=0.0):
    """Keplerian RV at times t [days]:  v(t) = vz + K [cos(omega + nu(t)) + e cos(omega)].

    Mean anomaly M = 2 pi (t / P - phi_p); then E = solve_kepler(M, e); nu = true_anomaly(E, e).
    """
    raise NotImplementedError


def photocenter_semimajor_axis_mas(P, mstar, m_co, parallax):
    """Angular semi-major axis of the visible star's orbit [mas], dark companion (f = 0):

        a_tot [AU] = (G (M_* + M_CO) P^2 / 4 pi^2)^(1/3)
        a_*  = a_tot * M_CO / (M_* + M_CO)
        a_* [mas] = a_* [AU] * parallax [mas]

    Useful for sanity checks and for the simplified RUWE model below.
    """
    raise NotImplementedError


# =============================================================== forward model  [TODO]
def rv_epoch_times(star, rng, data_release="dr3"):
    """Times [days] of the star's rv_nb_transits RV epochs.

    The spectroscopic scanning law is not public, so the paper draws a random subset of
    size rv_nb_transits from the ASTROMETRIC scan times at the star's position
    (gaiamock.get_gost_one_position). Fallback without gaiamock: draw rv_nb_transits times
    uniformly over the DR3 baseline (34 months = 1038 d).
    """
    raise NotImplementedError


def predict_sigma_rv(theta, star, rng, t_rv=None):
    """Predicted radial_velocity_error [km/s] for parameters theta  (paper Sect. 2):

        sigma_tilde = std of the simulated epoch RVs  (rv_curve at t_rv)
        radial_velocity_error = sqrt( (sqrt(pi / (2N)) * sigma_tilde)^2 + 0.113^2 )

    with N = rv_nb_transits, then multiply by SIGRV_SB1_SCALE (= 1.07).
    Only valid for grvs_mag <= 12 and rv_method_used = 1.
    """
    raise NotImplementedError


def predict_ruwe_gaiamock(theta, star, c_funcs, data_release="dr3"):
    """Predicted RUWE using gaiamock (the paper's choice).

    Steps: gaiamock.predict_astrometry_luminous_binary(ra, dec, parallax, pmra, pmdec,
    m1=mstar, m2=m_co, period=P, Tp=phi_p*P, ecc=e, omega=Omega, inc=inc, w=omega,
    phot_g_mean_mag=G, f=0, data_release, c_funcs) -> epoch astrometry; then
    gaiamock.check_ruwe(t_ast_yr, psi, plx_factor, ast_obs, ast_err) -> RUWE.
    Needs the scanning-law data (healpix_scans) downloaded and c_funcs = read_in_C_functions().
    """
    raise NotImplementedError


def predict_ruwe_simple(theta, star, rng, n_transits=None, sigma_al=None):
    """Fallback RUWE model in pure numpy (use this if gaiamock will not build on Windows).

    1. n_transits ~ astrometric_n_good_obs_al / 8 (one measurement per FOV transit).
    2. Draw transit times uniformly over the DR3 baseline and random scan angles psi.
    3. Along-scan position = 5-parameter single-star motion (set to zero, it is fitted out)
       + projection of the photocentre orbit onto the scan direction + Gaussian noise sigma_al.
    4. Fit the 5-parameter linear model by weighted least squares (same algebra as Lab 1 wls),
       chi2 = sum(resid^2 / sigma_al^2), nu = n - 5, ruwe = sqrt(chi2 / nu).
    The missing piece vs gaiamock is the real scanning law (times + angles + parallax factors),
    so expect a wider error than EPS_RUWE; quantify that in Task 5.
    """
    raise NotImplementedError


# =============================================================== priors & likelihood  [TODO]
def log_prior(theta):
    """Uniform priors from the paper; return 0.0 inside the box, -np.inf outside.

    logP in LOGP_RANGE, logMco in LOGMCO_RANGE, cos_i in (0, 1], e in E_RANGE,
    Omega and omega in [0, 2 pi), phi_p in [0, 1).
    """
    raise NotImplementedError


def sample_prior(n, rng):
    """n draws from the prior, shape (n, NDIM). Used for walker initialisation and tests."""
    raise NotImplementedError


def log_likelihood(theta, star, predictors):
    """Gaussian in log space for each observable  (paper Sect. 3):

        lnL = -1/2 ln(2 pi eps_r^2)  - 1/2 [(ln ruwe_obs  - ln ruwe_pred ) / eps_r]^2
              -1/2 ln(2 pi eps_s^2)  - 1/2 [(ln sigma_obs - ln sigma_pred) / eps_s]^2

    predictors: dict with callables {"ruwe": f(theta, star), "sigma_rv": f(theta, star)}
    so you can swap gaiamock <-> simple model without touching this function.
    """
    raise NotImplementedError


def log_posterior(theta, star, predictors):
    """log_prior + log_likelihood (skip the likelihood when the prior is -inf)."""
    raise NotImplementedError


# =============================================================== samplers  [TODO]
def run_emcee(star, predictors, rng, n_walkers=N_WALKERS, n_steps=N_STEPS, progress=False):
    """The paper's sampler: emcee.EnsembleSampler(n_walkers, NDIM, log_posterior,
    args=(star, predictors)). Initialise walkers with sample_prior(n_walkers, rng).
    Return the full chain (n_steps, n_walkers, NDIM) and log-prob (n_steps, n_walkers).
    """
    raise NotImplementedError


def run_metropolis(theta_start, jump, n_steps, star, predictors, rng):
    """Your own Metropolis-Hastings sampler (same structure as Lab 3's run_chain), for the
    comparison against emcee. Returns chain (n_steps, NDIM), log-post (n_steps,), acceptance.
    Note: the posterior is multimodal in logP, so expect MH to struggle where emcee does not;
    that contrast is worth a slide.
    """
    raise NotImplementedError


def summarize_posterior(chain, n_burn=N_BURN, m_threshold=M_CO_THRESHOLD):
    """Flatten after burn-in and return the paper's summary statistics:

        P_mean, Mco_mean      posterior-weighted means (the candidate criterion uses Mco_mean)
        Mco_16, Mco_50, Mco_84
        p_gt_threshold        P(M_CO > 3 Msun | data)
    """
    raise NotImplementedError


def convergence_checks(chain, n_burn=N_BURN):
    """Paper's checks, plus one standard diagnostic:
    (a) compare marginal P and M_CO between the 2nd and final thirds of the chain,
    (b) emcee.autocorr.integrated_time; want n_steps >> 50 tau.
    Return a dict of the numbers so they can go in a table.
    """
    raise NotImplementedError


# =============================================================== injection-recovery  [TODO]
def make_mock_star(host, m_co, P, rng, predictors):
    """Copy a real host star's row (position, brightness, transits), give it a dark companion
    with mass m_co and period P (thermal e, isotropic orientation, uniform phase), and REPLACE
    its ruwe and radial_velocity_error with forward-model predictions. Return the mock star
    and the true theta.
    """
    raise NotImplementedError


# =============================================================== batch running  [TODO]
def run_batch(stars, predictors, out_path, rng_seed=0, n_jobs=-1, checkpoint_every=200):
    """Run run_emcee + summarize_posterior over many stars in parallel (joblib), writing a
    summary row per star to out_path (.csv) every checkpoint_every stars so a crash or a
    closed laptop does not lose work. Skip source_ids already in out_path on restart.
    Give each star its own seed: np.random.default_rng([rng_seed, source_id]).
    """
    raise NotImplementedError


# =============================================================== contaminant filters  [TODO]
def flag_sigma_rv_outliers(tbl, sims):
    """Cut A: radial_velocity_error inconsistent with rv_amplitude_robust (> 3 sigma from the
    relation seen in simulated BH binaries). Paper removed 56 stars."""
    raise NotImplementedError


def flag_parallax_disagreement(tbl, plx_alt_col):
    """Cut B: Gaia parallax vs an independent (spectrophotometric) parallax differ by > x2.
    Paper removed 61 stars using Zhang+2023 XP parallaxes."""
    raise NotImplementedError


def predicted_ellipsoidal_amplitude(theta, star):
    """Cut C helper (Morris & Naftilan 1993 with the Gomel+2021 correction):
        A = c2 [ a2 (R/a)^3 q sin^2 i + b2 (R/a)^5 q (6 sin^2 i - 7 sin^4 i) ]
    Observed: A_obs = sqrt(2) sqrt(phot_g_n_obs) / phot_g_mean_flux_over_error."""
    raise NotImplementedError


def flag_photometric_variables(tbl, summaries, noise_floor=0.127):
    """Cut C: A_obs > A_pred + noise_floor  (paper removed 251; plus 36 known EBs / LPVs)."""
    raise NotImplementedError


def flag_has_nss_orbit(tbl):
    """Cut D: non_single_star > 0 (already has a DR3 orbit). Note this also removed BH2."""
    raise NotImplementedError
