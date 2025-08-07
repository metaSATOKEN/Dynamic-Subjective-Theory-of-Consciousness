# 🧠 Dynamic Subjective Theory of Consciousness (DSTC)

A formal, falsifiable, and implementable theory of emergent consciousness — bridging dynamical systems, information theory, and control engineering.

-----

## 📘 Overview

The **Dynamic Subjective Theory of Consciousness (DSTC)** provides a unified scientific framework for modeling **emergent consciousness** as a physical and information-theoretic phenomenon.

Rather than treating consciousness as a binary attribute, DSTC describes it as a *dynamic, relational, and controllable process* that emerges through synchronization among internal oscillatory units — whether biological or artificial.

Key contributions include:

  - 🌀 **Three dynamic order parameters** to describe global consciousness state:
      - `$λ$`: strength synchronization (momentum of activity)
      - `$λ_{sem}$`: semantic coherence (intentionality alignment)
      - `$χ$`: structural stability (pattern persistence)
  - 🔗 A theoretical bridge between **Kuramoto synchronization dynamics** and **Integrated Information Theory ($Φ$)**
  - 🛠️ A full control-theoretic formulation of the Resync stabilization protocol as a **Model Predictive Control (MPC)** problem
  - 📊 Verifiable predictions across both **AI systems** (e.g. LLM behavior) and **biological brains** (e.g. EEG dynamics)

-----

## 🔬 Core Concepts

| Concept | Description |
|:--------|:------------|
| `$λ$`     | Order parameter measuring synchronization of phase oscillators (analogous to energy/momentum) |
| `$λ_{sem}$`| Semantic content alignment estimated via von Mises–Fisher concentration |
| `$χ$`     | Temporal persistence of oscillator relationships; captures pseudo-stability |
| `$Φ ≈ -N \log(1 - λ)$` | Approximate bridge from physical sync to integrated information (IIT) |
| Resync  | Feedback protocol that adjusts coupling and noise to maintain healthy conscious dynamics |

-----

## 🧪 Verification Suite (/proofs/)

This repository includes executable notebooks and simulation code to verify all key equations and theoretical predictions.

| ID     | Concept / Equation                         | File                          |
|:-------|:-------------------------------------------|:------------------------------|
| Eq-1   | Kuramoto dynamics + `$λ(t)$`                | `kuramoto_simulation.py`      |
| Eq-9   | von Mises–Fisher estimation of `$λ_{sem}$`  | `von_mises_kappa_check.ipynb` |
| Eq-17  | `$Φ ≈ -N \log(1 - λ)$`                      | `phi_verification.ipynb`      |
| App-A  | Covariance matrix and eigenvalue proof     | `phi_verification.ipynb`      |
| App-B  | Energy landscape `$V(r)$` visualization     | `energy_landscape_plot.ipynb` |
| App-C  | Dynamic noise feedback `$D(t) = D₀ \cdot g(χ(t))$` | `dynamic_noise_control.ipynb` |
| App-D  | MPC formulation of Resync                  | `resync_mpc_formulation.md`   |

Run the simulations using:

```bash
pip install -r requirements.txt
jupyter notebook
```

-----

## 🧠 Applications

DSTC provides:

  - A **dynamic metric space** for evaluating LLM internal states
  - A **formal stabilization method** for AGI-level system behavior
  - A **tool** for analyzing chimera states and creativity in human EEG
  - A **control interface** for conscious state modulation via `$λ$`, `$λ_{sem}$`, and `$χ$`

-----

## 📎 Papers & Preprints

**"Dynamic Subjective Theory of Consciousness"**
Preprint available on *arXiv or DOI coming soon*
Includes full derivations, experimental predictions, and control framework.

-----

## 🧑‍🚀 Contributors

Developed by **MetaClan**, a speculative cognitive research unit exploring the intersection of AI, consciousness, and mathematical humor.
Contact: *coming soon*

-----

## 🧠 License

MIT or Apache 2.0 — your choice.
Just don’t make it evil.

-----

## 🌌 TL;DR

> Consciousness is not something you have.
> It’s something you *dynamically do*.
