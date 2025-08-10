# sgt_kernel_sim.py — SGT kernel v0.3.1 (with Metao fixes)
# - Proper time–scale separation: multiply slow dynamics by dt_fast
# - Semantic noise uses sqrt(dt_fast)
# - Optional χ exponential smoothing
# - Sinkhorn iterations default -> 5
#
# Quick demo:
#   from sgt_kernel_sim import *
#   cfg = SimConfig(N=300, d=12, T=8.0, seed=42)
#   out = run_sim(cfg)

from dataclasses import dataclass, asdict
import numpy as np

# ---------- Utilities ----------

def robust_scale(x):
    """Median/MAD scale to stabilize β/γ identifiability."""
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-8
    return (x - med) / mad

def sinkhorn_knopp(M, iters=5, eps=1e-12):
    """Make a matrix approximately doubly stochastic via Sinkhorn iterations."""
    # Row/col positive assumption
    A = M.copy()
    A = np.maximum(A, eps)
    for _ in range(iters):
        A /= (A.sum(axis=1, keepdims=True) + eps)
        A /= (A.sum(axis=0, keepdims=True) + eps)
    return A

def proj_tangent_sphere(U, V):
    """Project V onto tangent space of unit-sphere at U (row-wise)."""
    # For each row: v - (u · v) u
    dot = (U * V).sum(axis=1, keepdims=True)
    return V - dot * U

def normalize_rows(U, eps=1e-12):
    nrm = np.linalg.norm(U, axis=1, keepdims=True) + eps
    return U / nrm

def principal_order_param(theta):
    R = np.mean(np.exp(1j * theta))
    lam = (np.abs(R)) ** 2
    return R, lam

def semantic_order_param(U):
    m = np.mean(U, axis=0)
    R = np.linalg.norm(m)
    lam_sem = R
    m_hat = m / (np.linalg.norm(m) + 1e-12)
    return m_hat, lam_sem

def structural_persistence(theta, theta_lag):
    if theta_lag is None:
        return 0.0
    d = theta[:, None] - theta[None, :]
    d_lag = theta_lag[:, None] - theta_lag[None, :]
    s = np.sign(np.cos(d))
    s_lag = np.sign(np.cos(d_lag))
    # Upper triangle average (excluding diag)
    N = theta.shape[0]
    iu = np.triu_indices(N, k=1)
    chi = np.mean(s[iu] * s_lag[iu])
    # Map from [-1,1] -> [0,1] just in case (optional). Keep as is per spec.
    return chi

def laplacian_spectrum_radius(A):
    # Row-stochastic A => define symmetric normalized Laplacian approx via (A + A^T)/2
    S = 0.5 * (A + A.T)
    deg = S.sum(axis=1)
    L = np.diag(deg) - S
    # Power method for largest eigenvalue of L (spectral radius for stability heuristic)
    x = np.random.default_rng().normal(size=L.shape[0])
    x /= np.linalg.norm(x) + 1e-12
    for _ in range(30):
        x = L @ x
        n = np.linalg.norm(x) + 1e-12
        x /= n
    lam_max = np.linalg.norm(L @ x) / (np.linalg.norm(x) + 1e-12)
    return lam_max

# ---------- Config ----------

@dataclass
class SimConfig:
    N: int = 400
    d: int = 16
    T: float = 10.0             # total sim time (seconds in fast scale)
    dt_fast: float = 0.01       # fast integrator (theta, U)
    eps: float = 0.05           # slow/fast separation ratio (ε)
    seed: int = 7

    # Kuramoto / noise
    K0: float = 1.5
    D_ind: float = 0.05
    D_com: float = 0.02
    omega_std: float = 0.3

    # Semantic alignment
    alpha: float = 0.2
    eta_sem: float = 0.06       # noise strength (will be scaled by sqrt(dt))

    # Attention (SGT)
    tau_attn: float = 1.0
    sinkhorn_iters: int = 5

    # Slow β, γ dynamics (OU-like drifts)
    beta_init: float = 1.0
    gamma_init: float = 0.6
    kappa_beta: float = 0.6
    kappa_gamma: float = 0.5
    beta_star: float = 1.2
    gamma_star: float = 0.8
    c_beta_u: float = 0.8
    c_beta_chi: float = 0.7
    c_gamma_u: float = 0.7
    c_gamma_lam: float = 0.3
    sigma_beta: float = 0.02
    sigma_gamma: float = 0.02

    # Structural persistence
    chi_lag_steps: int = 100    # lag in fast steps (tau ≈ chi_lag_steps*dt_fast)
    chi_smooth: float = 0.2     # EMA smoothing factor ρ; 0 => off
    chi_star: float = 0.9

    # Recompute semantic similarity every k steps to save cost
    recompute_sem_every: int = 10

# ---------- Core Simulation ----------

def run_sim(cfg: SimConfig):
    rng = np.random.default_rng(cfg.seed)
    N, d = cfg.N, cfg.d
    steps = int(cfg.T / cfg.dt_fast)

    # States
    theta = rng.uniform(0, 2*np.pi, size=N)
    omega = rng.normal(0.0, cfg.omega_std, size=N)
    U = rng.normal(size=(N, d))
    U = normalize_rows(U)

    # Precompute static syntactic distances (toy: ring lattice distances)
    idx = np.arange(N)
    Dsyn = np.abs(idx[:, None] - idx[None, :])
    Dsyn = np.minimum(Dsyn, N - Dsyn).astype(float)
    Dsyn = robust_scale(Dsyn)

    # Init semantic similarity
    S_sem = U @ U.T  # cosine if U normalized; in [-1,1]

    beta = cfg.beta_init
    gamma = cfg.gamma_init

    # Attention weights (start uniform)
    A = np.ones((N, N)) / N

    # Logs
    lam_hist = np.zeros(steps)
    lam_sem_hist = np.zeros(steps)
    chi_hist = np.zeros(steps)
    V_total_hist = np.zeros(steps)
    beta_hist = np.zeros(steps)
    gamma_hist = np.zeros(steps)

    # Lag buffers
    theta_buffer = [theta.copy()]
    chi_ema = None

    # Common noise
    dW_common = rng.normal(0.0, 1.0, size=steps) * np.sqrt(cfg.dt_fast)

    for t in range(steps):
        # --- Fast: Kuramoto theta ---
        # Row-stochastic A => degree ~ 1; keep ki for safety
        ki = np.clip(A.sum(axis=1), 1e-6, None)
        coupling = (A * np.sin(theta[None, :] - theta[:, None])).sum(axis=1) / ki
        dtheta = (omega + cfg.K0 * coupling) * cfg.dt_fast \
                 + np.sqrt(2*cfg.D_ind) * rng.normal(size=N) * np.sqrt(cfg.dt_fast) \
                 + np.sqrt(2*cfg.D_com) * dW_common[t]
        theta = (theta + dtheta) % (2*np.pi)

        # --- Fast: semantic vectors (projected vMF-like drift + noise sqrt(dt)) ---
        m_hat, lam_sem = semantic_order_param(U)
        drift = proj_tangent_sphere(U, np.tile(m_hat, (N,1))) * cfg.alpha
        noise = rng.normal(0.0, cfg.eta_sem, size=U.shape) * np.sqrt(cfg.dt_fast)
        U = U + (drift + noise) * cfg.dt_fast
        U = normalize_rows(U)

        # --- Order params ---
        _, lam = principal_order_param(theta)

        # Structural persistence χ with lag
        theta_buffer.append(theta.copy())
        if len(theta_buffer) > cfg.chi_lag_steps + 1:
            theta_lag = theta_buffer[-(cfg.chi_lag_steps+1)]
        else:
            theta_lag = None
        chi_hat = structural_persistence(theta, theta_lag)
        if cfg.chi_smooth > 0 and chi_ema is not None:
            chi = (1 - cfg.chi_smooth) * chi_ema + cfg.chi_smooth * chi_hat
        else:
            chi = chi_hat
        chi_ema = chi

        # --- Slow: update attention A via SGT (softmax + Sinkhorn), scaled by dt_fast ---
        if (t % cfg.recompute_sem_every) == 0:
            S_sem = U @ U.T  # cosine in [-1,1]
        logits = (-beta * Dsyn + gamma * S_sem) / cfg.tau_attn
        # Stable softmax row-wise
        logits = logits - logits.max(axis=1, keepdims=True)
        A_target = np.exp(logits)
        A_target = sinkhorn_knopp(A_target, iters=cfg.sinkhorn_iters)
        # Slow Euler step with dt_fast factor (fix)
        A = A + cfg.eps * cfg.dt_fast * (A_target - A)

        # --- Slow: β, γ OU-like with dt_fast factor (fix) ---
        beta += cfg.eps * cfg.dt_fast * (
            cfg.kappa_beta * (cfg.beta_star - beta)
            + cfg.c_beta_u * (lam_sem - 0.6)
            - cfg.c_beta_chi * (chi - cfg.chi_star)
        ) + cfg.sigma_beta * np.sqrt(cfg.dt_fast) * rng.normal()

        gamma += cfg.eps * cfg.dt_fast * (
            cfg.kappa_gamma * (cfg.gamma_star - gamma)
            + cfg.c_gamma_u * (lam_sem - 0.6)
            + cfg.c_gamma_lam * (lam - 0.5)
        ) + cfg.sigma_gamma * np.sqrt(cfg.dt_fast) * rng.normal()

        # --- Lyapunov-like monitors ---
        V_theta = 0.5 * (1 - lam) ** 2
        # For semantic, target R_u* ≈ lam_sem steady-state proxy; keep moving target simple here
        V_sem = 0.5 * (max(lam_sem, 1e-6) - lam_sem) ** 2  # placeholder; user can set target
        V_chi = 0.5 * (cfg.chi_star - chi) ** 2
        V_total = V_theta + V_sem + V_chi

        # Log
        lam_hist[t] = lam
        lam_sem_hist[t] = lam_sem
        chi_hist[t] = chi if np.isfinite(chi) else 0.0
        V_total_hist[t] = V_total
        beta_hist[t] = beta
        gamma_hist[t] = gamma

    # Spectrum radius snapshot at final A (optional diagnostic)
    rho_max = laplacian_spectrum_radius(A)

    return {
        "config": asdict(cfg),
        "lam": lam_hist,
        "lam_sem": lam_sem_hist,
        "chi": chi_hist,
        "V_total": V_total_hist,
        "beta": beta_hist,
        "gamma": gamma_hist,
        "A_final": A,
        "rho_max_L": rho_max,
    }

# Optional plotting (kept minimal; no fixed colors/style as per policy)
def plot_summary(out):
    import matplotlib.pyplot as plt
    t = np.arange(len(out["lam"])) * out["config"]["dt_fast"]
    fig, ax = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    ax[0].plot(t, out["lam"]);        ax[0].set_ylabel(r"$\lambda$")
    ax[1].plot(t, out["lam_sem"]);    ax[1].set_ylabel(r"$\lambda_{\rm sem}$")
    ax[2].plot(t, out["chi"]);        ax[2].set_ylabel(r"$\chi$")
    ax[3].plot(t, out["V_total"]);    ax[3].set_ylabel(r"$V_{\rm total}$"); ax[3].set_xlabel("time")
    plt.tight_layout()
    return fig
