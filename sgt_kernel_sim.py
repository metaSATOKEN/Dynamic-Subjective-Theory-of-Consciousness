# sgt_kernel_sim_v0_3_2.py
# Self-contained scaffold for SGT kernel simulation (v0.3.2)
# Implements: fast θ / slow (A, β, γ), τ=1 identifiability, Sinkhorn double-stochastic,
# local semantic drift (A @ U), dt_fast scaling in slow dynamics, √dt noise in SDEs,
# Lyapunov-friendly K(λ), β/γ >= 0 clipping, tiny entropy in attention logits exp.

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------- Utilities ----------------------------

def robust_scale(D):
    """Median/MAD robust scaling to fix distance scale (for identifiability)."""
    med = np.median(D)
    mad = np.median(np.abs(D - med)) + 1e-12
    return (D - med) / mad

def normalize_rows(X):
    nrm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / nrm

def proj_tangent_sphere(U, V):
    """Project V onto tangent space at U on the unit sphere (row-wise)."""
    # remove radial component: (U·V)U
    dot = np.sum(U * V, axis=1, keepdims=True)
    return V - dot * U

def sinkhorn(X, iters=5):
    """Make X approximately doubly stochastic via Sinkhorn-Knopp."""
    M = X.copy()
    for _ in range(iters):
        M /= (M.sum(axis=1, keepdims=True) + 1e-12)  # row normalize
        M /= (M.sum(axis=0, keepdims=True) + 1e-12)  # col normalize
    return M

# ---------------------------- Config ----------------------------

@dataclass
class SimConfig:
    # Graph / population
    N: int = 300
    d: int = 12
    T: float = 8.0
    dt_fast: float = 0.01   # fast step (θ/U updates)
    seed: int = 42

    # Attention / similarity
    tau_attn: float = 1.0
    sinkhorn_iters: int = 5
    recompute_sem_every: int = 20  # steps
    add_small_entropy: bool = True

    # Slow rates (ε)
    eps: float = 0.05  # separation ratio (slow dynamics speed)

    # Phase dynamics
    K0: float = 1.5
    D_ind: float = 0.05
    D_com: float = 0.02

    # Semantic dynamics
    alpha: float = 0.2
    eta_sem: float = 0.05

    # Targets for Lyapunov-style energy
    lam_sem_star: float = 0.8
    chi_star: float = 1.0

    # β, γ (OU-like) hyper
    beta_init: float = 1.0
    gamma_init: float = 0.8
    beta_star: float = 1.2
    gamma_star: float = 1.0
    kappa_beta: float = 0.3
    kappa_gamma: float = 0.3
    c_beta_u: float = 0.2
    c_beta_chi: float = 0.2
    c_gamma_u: float = 0.25
    c_gamma_lam: float = 0.15
    sigma_beta: float = 0.02
    sigma_gamma: float = 0.02

    # χ estimator
    chi_lag_steps: int = 100  # τ ≈ chi_lag_steps * dt_fast
    chi_smooth_rho: float = 0.2  # EMA (0 disables smoothing)

    # Lyapunov weights
    w_theta: float = 1.0
    w_sem: float = 0.6
    w_chi: float = 0.4

def setup_initial_state(cfg: SimConfig):
    rng = np.random.default_rng(cfg.seed)
    # phases and natural frequencies
    theta = rng.uniform(0, 2*np.pi, size=cfg.N)
    omega = rng.normal(0.0, 0.5, size=cfg.N)
    # semantic unit vectors on sphere
    U = rng.normal(size=(cfg.N, cfg.d))
    U = normalize_rows(U)
    # distances in "syntax" space (use random embeddings as a placeholder)
    Xsyn = rng.normal(size=(cfg.N, cfg.d))
    Dsyn = np.linalg.norm(Xsyn[:, None, :] - Xsyn[None, :, :], axis=2)
    Dsyn = robust_scale(Dsyn)
    # initial semantic similarity (cosine, since U rows are unit)
    S_sem = U @ U.T
    # initial attention (uniform doubly-stochastic)
    A = np.ones((cfg.N, cfg.N))
    A = sinkhorn(A, iters=cfg.sinkhorn_iters)
    # beta, gamma
    beta = cfg.beta_init
    gamma = cfg.gamma_init
    return theta, omega, U, Dsyn, S_sem, A, beta, gamma, rng

# ---------------------------- Core Simulation ----------------------------

def run_sim(cfg: SimConfig):
    theta, omega, U, Dsyn, S_sem, A, beta, gamma, rng = setup_initial_state(cfg)

    steps = int(cfg.T / cfg.dt_fast)
    # buffers
    lam_hist = np.zeros(steps)
    lam_sem_hist = np.zeros(steps)
    chi_hist = np.zeros(steps)
    beta_hist = np.zeros(steps)
    gamma_hist = np.zeros(steps)
    K_hist = np.zeros(steps)
    V_hist = np.zeros(steps)

    # chi lag buffer (store cos Δ_ij signs compactly by sampling pairs once)
    # For stability/efficiency we use a fixed subset of pairs
    pair_rng = np.random.default_rng(cfg.seed + 7)
    idx_i = pair_rng.integers(0, cfg.N, size=min(2000, cfg.N*5))
    idx_j = pair_rng.integers(0, cfg.N, size=idx_i.shape[0])
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]

    # history of phase differences for lagged chi
    theta_hist = np.tile(theta, (cfg.chi_lag_steps+1, 1))

    chi_ema = None

    for t in range(steps):
        # --- recompute semantic similarity occasionally (cosine because U normalized) ---
        if (t % cfg.recompute_sem_every) == 0:
            S_sem = U @ U.T
            np.fill_diagonal(S_sem, 1.0)

        # --- Attention target (softmax logits / tau, then Sinkhorn) ---
        logits = (-beta * Dsyn + gamma * S_sem) / cfg.tau_attn
        if cfg.add_small_entropy:
            W = np.exp(logits) + 1e-12
        else:
            # safer exp via max subtraction (though DS will rescale anyway)
            logits = logits - logits.max(axis=1, keepdims=True)
            W = np.exp(logits) + 1e-12
        A_target = sinkhorn(W, iters=cfg.sinkhorn_iters)

        # --- Slow Attention update (dt_fast scaled) ---
        A = A + cfg.eps * cfg.dt_fast * (A_target - A)

        # --- Local semantic drift: towards normalized (A @ U) ---
        Mloc = normalize_rows(A @ U)
        drift = proj_tangent_sphere(U, Mloc) * cfg.alpha
        noise = rng.normal(0.0, cfg.eta_sem, size=U.shape) * np.sqrt(cfg.dt_fast)
        U = U + drift * cfg.dt_fast + noise
        U = normalize_rows(U)  # re-project to sphere

        # --- Order parameters ---
        R = np.mean(np.exp(1j * theta))
        lam = np.abs(R)**2
        m = np.mean(U, axis=0)
        lam_sem = np.linalg.norm(m)

        # --- Chi estimator (sign agreement of cos Δ now vs lagged) ---
        # update theta history ring buffer
        theta_hist[:-1] = theta_hist[1:]  # shift up
        theta_hist[-1] = theta
        theta_lag = theta_hist[0]

        d_now = np.cos(theta[idx_i] - theta[idx_j])
        d_lag = np.cos(theta_lag[idx_i] - theta_lag[idx_j])
        chi_hat = np.mean(np.sign(d_now) * np.sign(d_lag))
        if cfg.chi_smooth_rho > 0:
            chi_ema = chi_hat if chi_ema is None else (1-cfg.chi_smooth_rho)*chi_ema + cfg.chi_smooth_rho*chi_hat
            chi = chi_ema
        else:
            chi = chi_hat

        # --- Phase dynamics ---
        # Lyapunov-friendly feedback: reduce K as λ approaches 1
        K = cfg.K0 * (1.0 + (1.0 - lam))
        ki = A.sum(axis=1)
        ki = np.clip(ki, 1e-6, None)
        coupling = (A * np.sin(theta[None, :] - theta[:, None])).sum(axis=1) / ki

        dW_ind = rng.normal(0.0, np.sqrt(cfg.dt_fast), size=cfg.N)
        dW_com = rng.normal(0.0, np.sqrt(cfg.dt_fast))
        dtheta = (omega + K * coupling) * cfg.dt_fast \
                 + np.sqrt(2*cfg.D_ind) * dW_ind + np.sqrt(2*cfg.D_com) * dW_com
        theta = (theta + dtheta) % (2*np.pi)

        # --- β, γ slow OU-like updates (dt_fast scaled) ---
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

        # Nonnegativity clips (interpretability)
        beta = max(0.0, beta); gamma = max(0.0, gamma)

        # --- Energies (Lyapunov candidates) ---
        V_theta = 0.5 * (1.0 - lam)**2
        V_sem = 0.5 * (cfg.lam_sem_star - lam_sem)**2
        V_chi = 0.5 * (cfg.chi_star - chi)**2
        V_total = cfg.w_theta * V_theta + cfg.w_sem * V_sem + cfg.w_chi * V_chi

        # log
        lam_hist[t] = lam
        lam_sem_hist[t] = lam_sem
        chi_hist[t] = chi
        beta_hist[t] = beta
        gamma_hist[t] = gamma
        K_hist[t] = K
        V_hist[t] = V_total

    return {
        "t": np.arange(steps) * cfg.dt_fast,
        "lam": lam_hist,
        "lam_sem": lam_sem_hist,
        "chi": chi_hist,
        "beta": beta_hist,
        "gamma": gamma_hist,
        "K": K_hist,
        "V_total": V_hist,
        "cfg": cfg
    }

# ---------------------------- Plotting ----------------------------

def plot_summary(out):
    t = out["t"]
    fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(t, out["lam"]); axes[0].set_ylabel("λ (phase sync)")
    axes[1].plot(t, out["lam_sem"]); axes[1].set_ylabel("λ_sem (semantic)")
    axes[2].plot(t, out["chi"]); axes[2].set_ylabel("χ (struct.)")
    axes[3].plot(t, out["beta"], label="β"); axes[3].plot(t, out["gamma"], label="γ"); axes[3].legend(); axes[3].set_ylabel("β, γ")
    axes[4].plot(t, out["V_total"]); axes[4].set_ylabel("V_total"); axes[4].set_xlabel("time")
    fig.tight_layout()
    return fig

# ---------------------------- Save as file ----------------------------

code_text = open('/mnt/data/sgt_kernel_sim_v0_3_2.py','w',encoding='utf-8')
code_text.write("""\
\"\"\"SGT kernel simulation v0.3.2

Usage:
    from sgt_kernel_sim_v0_3_2 import SimConfig, run_sim, plot_summary
    cfg = SimConfig(N=300, d=12, T=8.0, seed=42)
    out = run_sim(cfg)
    fig = plot_summary(out)

Notes:
- fast θ / slow (A, β, γ) with dt_fast scaling
- local semantic drift via A @ U
- doubly-stochastic attention via Sinkhorn
- τ=1 identifiability, robust distance scaling
- Lyapunov-friendly K(λ), nonnegativity clip for β,γ
- χ estimator with EMA smoothing
\"\"\"

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

def robust_scale(D):
    med = np.median(D); mad = np.median(np.abs(D - med)) + 1e-12
    return (D - med) / mad

def normalize_rows(X):
    nrm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / nrm

def proj_tangent_sphere(U, V):
    dot = np.sum(U * V, axis=1, keepdims=True)
    return V - dot * U

def sinkhorn(X, iters=5):
    M = X.copy()
    for _ in range(iters):
        M /= (M.sum(axis=1, keepdims=True) + 1e-12)
        M /= (M.sum(axis=0, keepdims=True) + 1e-12)
    return M

@dataclass
class SimConfig:
    N:int=300; d:int=12; T:float=8.0; dt_fast:float=0.01; seed:int=42
    tau_attn:float=1.0; sinkhorn_iters:int=5; recompute_sem_every:int=20; add_small_entropy:bool=True
    eps:float=0.05
    K0:float=1.5; D_ind:float=0.05; D_com:float=0.02
    alpha:float=0.2; eta_sem:float=0.05
    lam_sem_star:float=0.8; chi_star:float=1.0
    beta_init:float=1.0; gamma_init:float=0.8; beta_star:float=1.2; gamma_star:float=1.0
    kappa_beta:float=0.3; kappa_gamma:float=0.3
    c_beta_u:float=0.2; c_beta_chi:float=0.2; c_gamma_u:float=0.25; c_gamma_lam:float=0.15
    sigma_beta:float=0.02; sigma_gamma:float=0.02
    chi_lag_steps:int=100; chi_smooth_rho:float=0.2
    w_theta:float=1.0; w_sem:float=0.6; w_chi:float=0.4

def setup_initial_state(cfg: SimConfig):
    rng = np.random.default_rng(cfg.seed)
    theta = rng.uniform(0, 2*np.pi, size=cfg.N)
    omega = rng.normal(0.0, 0.5, size=cfg.N)
    U = rng.normal(size=(cfg.N, cfg.d)); U = normalize_rows(U)
    Xsyn = rng.normal(size=(cfg.N, cfg.d))
    Dsyn = np.linalg.norm(Xsyn[:, None, :] - Xsyn[None, :, :], axis=2); Dsyn = robust_scale(Dsyn)
    S_sem = U @ U.T; np.fill_diagonal(S_sem, 1.0)
    A = np.ones((cfg.N, cfg.N)); A = sinkhorn(A, iters=cfg.sinkhorn_iters)
    beta = cfg.beta_init; gamma = cfg.gamma_init
    return theta, omega, U, Dsyn, S_sem, A, beta, gamma, rng

def run_sim(cfg: SimConfig):
    theta, omega, U, Dsyn, S_sem, A, beta, gamma, rng = setup_initial_state(cfg)
    steps = int(cfg.T / cfg.dt_fast)
    lam_hist = np.zeros(steps); lam_sem_hist = np.zeros(steps); chi_hist = np.zeros(steps)
    beta_hist = np.zeros(steps); gamma_hist = np.zeros(steps); K_hist = np.zeros(steps); V_hist = np.zeros(steps)

    pair_rng = np.random.default_rng(cfg.seed + 7)
    idx_i = pair_rng.integers(0, cfg.N, size=min(2000, cfg.N*5))
    idx_j = pair_rng.integers(0, cfg.N, size=idx_i.shape[0]); mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]

    theta_hist = np.tile(theta, (cfg.chi_lag_steps+1, 1))
    chi_ema = None

    for t in range(steps):
        if (t % cfg.recompute_sem_every) == 0:
            S_sem = U @ U.T; np.fill_diagonal(S_sem, 1.0)

        logits = (-beta * Dsyn + gamma * S_sem) / cfg.tau_attn
        W = np.exp(logits) + 1e-12 if cfg.add_small_entropy else np.exp(logits - logits.max(axis=1, keepdims=True)) + 1e-12
        A_target = sinkhorn(W, iters=cfg.sinkhorn_iters)
        A = A + cfg.eps * cfg.dt_fast * (A_target - A)

        Mloc = normalize_rows(A @ U)
        drift = proj_tangent_sphere(U, Mloc) * cfg.alpha
        noise = rng.normal(0.0, cfg.eta_sem, size=U.shape) * np.sqrt(cfg.dt_fast)
        U = normalize_rows(U + drift * cfg.dt_fast + noise)

        R = np.mean(np.exp(1j * theta)); lam = np.abs(R)**2
        m = np.mean(U, axis=0); lam_sem = np.linalg.norm(m)

        theta_hist[:-1] = theta_hist[1:]; theta_hist[-1] = theta
        theta_lag = theta_hist[0]
        d_now = np.cos(theta[idx_i] - theta[idx_j]); d_lag = np.cos(theta_lag[idx_i] - theta_lag[idx_j])
        chi_hat = np.mean(np.sign(d_now) * np.sign(d_lag))
        chi = chi_hat if cfg.chi_smooth_rho<=0 else (chi_hat if (chi_ema is None) else (1-cfg.chi_smooth_rho)*chi_ema + cfg.chi_smooth_rho*chi_hat)
        chi_ema = chi

        K = cfg.K0 * (1.0 + (1.0 - lam))
        ki = np.clip(A.sum(axis=1), 1e-6, None)
        coupling = (A * np.sin(theta[None, :] - theta[:, None])).sum(axis=1) / ki
        dW_ind = rng.normal(0.0, np.sqrt(cfg.dt_fast), size=cfg.N); dW_com = rng.normal(0.0, np.sqrt(cfg.dt_fast))
        theta = (theta + (omega + K * coupling) * cfg.dt_fast + np.sqrt(2*cfg.D_ind)*dW_ind + np.sqrt(2*cfg.D_com)*dW_com) % (2*np.pi)

        beta += cfg.eps * cfg.dt_fast * (cfg.kappa_beta * (cfg.beta_star - beta) + cfg.c_beta_u * (lam_sem - 0.6) - cfg.c_beta_chi * (chi - cfg.chi_star)) \
                + cfg.sigma_beta * np.sqrt(cfg.dt_fast) * rng.normal()
        gamma += cfg.eps * cfg.dt_fast * (cfg.kappa_gamma * (cfg.gamma_star - gamma) + cfg.c_gamma_u * (lam_sem - 0.6) + cfg.c_gamma_lam * (lam - 0.5)) \
                + cfg.sigma_gamma * np.sqrt(cfg.dt_fast) * rng.normal()
        beta = max(0.0, beta); gamma = max(0.0, gamma)

        V_theta = 0.5 * (1.0 - lam)**2
        V_sem = 0.5 * (cfg.lam_sem_star - lam_sem)**2
        V_chi = 0.5 * (cfg.chi_star - chi)**2
        V_total = cfg.w_theta * V_theta + cfg.w_sem * V_sem + cfg.w_chi * V_chi

        lam_hist[t]=lam; lam_sem_hist[t]=lam_sem; chi_hist[t]=chi; beta_hist[t]=beta; gamma_hist[t]=gamma; K_hist[t]=K; V_hist[t]=V_total

    return {"t": np.arange(steps)*cfg.dt_fast, "lam":lam_hist, "lam_sem":lam_sem_hist, "chi":chi_hist,
            "beta":beta_hist, "gamma":gamma_hist, "K":K_hist, "V_total":V_hist, "cfg":cfg}

def plot_summary(out):
    t = out["t"]
    fig, axes = plt.subplots(5, 1, figsize=(8,10), sharex=True)
    axes[0].plot(t, out["lam"]); axes[0].set_ylabel("λ (phase)")
    axes[1].plot(t, out["lam_sem"]); axes[1].set_ylabel("λ_sem")
    axes[2].plot(t, out["chi"]); axes[2].set_ylabel("χ")
    axes[3].plot(t, out["beta"], label="β"); axes[3].plot(t, out["gamma"], label="γ"); axes[3].legend(); axes[3].set_ylabel("β, γ")
    axes[4].plot(t, out["V_total"]); axes[4].set_ylabel("V_total"); axes[4].set_xlabel("time")
    fig.tight_layout(); return fig
""")
code_text.close()

print("Saved to /mnt/data/sgt_kernel_sim_v0_3_2.py")
