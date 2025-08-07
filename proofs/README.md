# DSTC Mathematical Verification Suite

This repository provides automated mathematical verification and numerical experiments for the **Dynamic Subjective Theory of Consciousness (DSTC)** using Codex/GPT-based symbolic engines.

## 📌 Verification Targets

| ID     | Formula / Concept                           | File                            | Purpose |
|--------|---------------------------------------------|----------------------------------|---------|
| Eq-1   | Kuramoto oscillator dynamics                | `kuramoto_simulation.py`         | Simulate λ(t) |
| Eq-5   | Strength synchronization: λ = r²            | `kuramoto_simulation.py`         | λ(t) time series |
| Eq-9   | vMF estimation of semantic sync (κ̂)        | `von_mises_kappa_check.ipynb`    | λ_sem validation |
| Eq-17  | Φ_G ≈ -N log(1 - λ)                         | `phi_verification.ipynb`         | IIT connection |
| App-A  | Covariance matrix & Φ_G derivation          | `phi_verification.ipynb`         | eigenvalue check |
| App-B  | Energy landscape V(r)                       | `energy_landscape_plot.ipynb`    | visual & symbolic |
| App-C  | Dynamic noise feedback D(t) = D₀·g(χ(t))    | `dynamic_noise_control.ipynb`    | verify Resync |
| App-D  | MPC formulation of Resync                   | `resync_mpc_formulation.md`      | symbolic MPC modeling |

## ⚙️ Libraries Required

```bash
pip install numpy scipy matplotlib sympy jupyter
Optional for symbolic: sympy, torch, sklearn, networkx

For notebooks: jupyter or VSCode + Jupyter plugin
```
