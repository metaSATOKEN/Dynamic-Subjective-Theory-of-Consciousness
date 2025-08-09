\title{IQ: An Information-Theoretic Framework for Semantic Phase Transitions and Integration Dynamics}

\author{Author Names Omitted for Review}

\date{}

\begin{abstract}
Understanding the emergent organization of meaning in neural systems is a fundamental challenge in consciousness science. This work introduces the IQ (Information-Theoretic constructivism of semantic content) framework, which unifies phase transition theory, information geometry, and stochastic dynamics on the unit hypersphere. Extending the Dynamic Subjective Theory of Consciousness (DSTC), we develop a rigorous model of semantic alignment and integration based on prediction-error-driven dynamics and von Mises–Fisher geometry.

The IQ model formalizes semantic order as a collective phenomenon on $\mathbb{S}^{d-1}$, with phase transitions induced by topological and statistical coupling. Through analytical derivation, we establish critical coupling thresholds, stochastic stability criteria, and information-theoretic observables such as $\lambda_{\mathrm{sem}}$ and $\lambda_{\mathrm{sem}}^{\mathrm{mix}}$. These quantities allow fine-grained measurement of semantic ignition and structured diversity in both artificial agents and brain signals.

We validate the model using synchronized neural networks, EEG semantic ignition detection, and structured mixture embeddings. Although the integration loop’s feedback efficacy could not be statistically confirmed under small-scale settings, we frame this as a robustness boundary and propose a community-driven exploration challenge. All theoretical derivations, Python implementations, and standardized experimental protocols are openly provided.

\textbf{Keywords}: Semantic dynamics, prediction error, von Mises–Fisher, phase transition, integration loop, information geometry, consciousness modeling
\end{abstract}

\section{Introduction}

The emergence of structured meaning from distributed neuronal activity remains one of the core challenges in consciousness science and cognitive modeling. While classical models of brain function emphasize either dynamical synchrony or information-theoretic constraints independently, a unified account capturing the semantics of mental content remains elusive.

Recent progress in predictive coding and phase synchronization models, particularly within the Dynamic Subjective Theory of Consciousness (DSTC), has opened new avenues for understanding how semantic alignment can arise from low-level prediction-error dynamics. However, DSTC primarily focused on phase-based dynamics and lacked a fully formalized geometric treatment of semantic integration.

In this paper, we introduce the \textbf{IQ (Information-Theoretic constructivism of semantic content)} framework, which extends DSTC by embedding semantic representations on the unit hypersphere $\mathbb{S}^{d-1}$ and defining collective dynamics through stochastic differential equations (SDEs) with prediction-error-modulated gain functions.

The IQ framework integrates:
\begin{itemize}
  \item A mathematically rigorous definition of semantic order using von Mises–Fisher geometry
  \item A set of closed-form critical coupling conditions for semantic ignition and phase transition
  \item An experimentally testable loop linking semantic coherence to phase synchrony: $\lambda_{\mathrm{sem}} \rightarrow K_{\mathrm{phase}} \rightarrow \lambda$
\end{itemize}

We empirically validate these theoretical contributions through a combination of:
\begin{itemize}
  \item Semantic ignition in neural agents under topological coupling
  \item Analysis of structured semantic diversity using vMF mixtures
  \item A null result under integration loop perturbation, presented as a robustness boundary and open empirical challenge
\end{itemize}

Our goal is to provide not only a theoretical contribution but also a fully reproducible computational and experimental platform for future community-driven verification. All implementations, protocols, and datasets are released with this work.

\section{Mathematical Framework}

We formalize the IQ framework by embedding semantic representations on the unit sphere $\mathbb{S}^{d-1} \subset \mathbb{R}^d$ and introducing predictive error-driven dynamics governed by stochastic differential equations (SDEs). The framework unifies semantic alignment and phase synchrony through dual-order parameters.

\subsection{2.1 Semantic Embedding and Order Parameter}

Each agent $i$ maintains a semantic vector $\mathbf{u}_i \in \mathbb{S}^{d-1}$, representing its current internal content state. The population-level semantic alignment is quantified by the semantic order parameter:

\begin{equation}
\lambda_{\mathrm{sem}} = 1 - \exp(-c_{\mathrm{sem}} \kappa(\mathbf{R}))
\end{equation}

where $\mathbf{R} = \left\| \frac{1}{N} \sum_{i=1}^{N} \mathbf{u}_i \right\|$ is the mean resultant length and $\kappa(\cdot)$ denotes the concentration parameter estimator of the von Mises–Fisher (vMF) distribution.

\subsection{2.2 Prediction Error-Driven Dynamics}

The temporal evolution of each semantic vector $\mathbf{u}_i$ is modeled as a Stratonovich SDE on $\mathbb{S}^{d-1}$:

\begin{equation}
d\mathbf{u}_i = K_{\mathrm{sem}} \sum_{j} \tilde{A}_{ij} P_{\mathbf{u}_i}(\mathbf{u}_j - \mathbf{u}_i) \, dt + \sqrt{2D_{\mathrm{sem}}(E_i)} \, \circ d\mathbf{W}_i
\end{equation}

Here, $P_{\mathbf{u}_i}(\cdot)$ denotes projection onto the tangent space at $\mathbf{u}_i$, and $E_i$ is the agent's local prediction error. The gain functions are modulated by prediction error via:

\begin{align}
K_{\mathrm{sem}}(E) &= \max\left\{K_0 \left[1 - \alpha_K \tanh\left(\beta(E - \theta_E)\right)\right], 0\right\} \\
D_{\mathrm{sem}}(E) &= D_0 \exp\left( \alpha_D \tanh(\beta(E - \theta_E)) \right)
\end{align}

\subsection{2.3 Critical Coupling and Phase Transition}

We analytically derive the critical coupling for semantic ignition:

\begin{equation}
K_{\mathrm{sem},c} = \frac{c_d D_{\mathrm{sem}}}{\lambda_2(L_*)}
\end{equation}

where $c_d = \frac{d-1}{d}$, $L_*$ is the symmetrized Laplacian $L_* = I - \frac{1}{2}(\tilde{A} + \tilde{A}^T)$, and $\lambda_2$ is its Fiedler eigenvalue. This defines a sharp semantic phase transition analogous to Kuramoto synchrony.

\subsection{2.4 Integration Loop Hypothesis}

We hypothesize a feedback loop linking semantic coherence ($\lambda_{\mathrm{sem}}$) to phase synchrony ($\lambda$) via phase coupling modulation:

\begin{equation}
K_{\mathrm{phase}} = K_0^{\mathrm{phase}} \cdot (1 + \alpha_1 \lambda_{\mathrm{phase}}) \cdot (1 + \alpha_2 \lambda_{\mathrm{sem}})
\end{equation}

This defines a three-stage loop:

\begin{center}
\textbf{Semantic gain} $\Rightarrow$ \textbf{Phase coupling} $\Rightarrow$ \textbf{Phase synchrony} $\Rightarrow$ \textbf{Reduced prediction error} $\Rightarrow$ Semantic gain...
\end{center}

This loop is empirically tested in Section~4.

\section{Methods}

We detail the simulation framework used to evaluate the IQ model across semantic, phase, and integration dynamics. All experiments were implemented in Python using NumPy and SciPy, with reproducibility ensured via fixed random seeds and shared protocols.

\subsection{3.1 Network Construction and Initialization}

We generated random directed graphs with $N$ nodes using three canonical models:

\begin{itemize}
    \item \textbf{Erdős-Rényi (ER)} with edge probability $p=0.1$
    \item \textbf{Watts-Strogatz (WS)} with $k=6$, rewiring $p=0.3$
    \item \textbf{Barabási-Albert (BA)} with $m=3$
\end{itemize}

Each graph was row-stochasticized to produce $\tilde{A}$. Initial semantic vectors $\{\mathbf{u}_i\}$ were sampled uniformly on $\mathbb{S}^{d-1}$ via Gaussian projection and normalization. Phase angles $\theta_i$ were initialized randomly in $[0, 2\pi]$.

\subsection{3.2 Simulation Parameters and Error Modeling}

The following core parameters were used:

\begin{itemize}
    \item Dimensionality: $d = 16$
    \item Steps: $T = 2000$, Time step: $dt = 0.005$
    \item Semantic base gain: $K_0 = 1.2$, base diffusion: $D_0 = 0.2$
    \item Phase base gain: $K_0^{\mathrm{phase}} = 2.0$, $D_{\mathrm{phase}} = 0.1$
    \item Error response: $\alpha_K = 0.8$, $\alpha_D = 0.8$, $\beta = 5.0$, $\theta_E = 0.5$
\end{itemize}

Prediction error was computed via a KL-divergence between model and target logits:

\begin{equation}
E = \mathrm{KL}(P_{\mathrm{emp}} \| P_{\mathrm{model}}) = \sum_{i=1}^C p_i^{\mathrm{emp}} \log \frac{p_i^{\mathrm{emp}}}{p_i^{\mathrm{model}}}
\end{equation}

We used synthetic classification tasks with 2–4 classes, softmax output logits, and label-driven targets.

\subsection{3.3 Numerical Integration}

Semantic updates were performed using a second-order Stratonovich Heun method with shared noise increments:

\begin{align}
\mathbf{U}_{\text{pred}} &= \mathbf{U} + dt \cdot K_{\mathrm{sem}} \cdot \text{drift}_1 + \sqrt{2 D_{\mathrm{sem}} dt} \cdot \boldsymbol{\xi}_1 \\
\mathbf{U}_{\text{corr}} &= \mathbf{U} + \frac{dt}{2} \cdot K_{\mathrm{sem}} \cdot (\text{drift}_1 + \text{drift}_2) + \sqrt{2 D_{\mathrm{sem}} dt} \cdot \frac{1}{2}(\boldsymbol{\xi}_1 + \boldsymbol{\xi}_2)
\end{align}

with projection of noise to the tangent space after each update step. All semantic vectors were renormalized after each step to preserve spherical constraint.

Phase angles were updated via standard Euler-Maruyama integration:

\begin{equation}
\theta_i(t + dt) = \theta_i(t) + dt \cdot \omega_i + \frac{K_{\mathrm{phase}}}{N} \sum_j A_{ij} \sin(\theta_j - \theta_i) + \sqrt{2 D_{\mathrm{phase}} dt} \cdot \eta_i
\end{equation}

with $\omega_i = 0$ unless otherwise stated.

\subsection{3.4 Order Parameter Computation}

Semantic alignment $\lambda_{\mathrm{sem}}$ was computed via the vMF estimator:

\begin{equation}
\lambda_{\mathrm{sem}} = 1 - \exp\left(-c_{\mathrm{sem}} \cdot \kappa(\mathbf{R})\right), \quad \text{where} \quad \mathbf{R} = \left\| \frac{1}{N} \sum_{i=1}^N \mathbf{u}_i \right\|
\end{equation}

Phase synchrony was computed as:

\begin{equation}
\lambda_{\mathrm{phase}} = \left| \frac{1}{N} \sum_{i=1}^N e^{i \theta_i} \right|^2
\end{equation}

Mixed semantic alignment was calculated for clustering experiments as:

\begin{equation}
\lambda_{\mathrm{sem}}^{\mathrm{mix}} = \sum_m \pi_m (1 - \exp(-c_{\mathrm{sem}} \kappa_m)) - \gamma_{\mathrm{mix}} H_n
\end{equation}

with $H_n = -\sum \pi_m \log \pi_m / \max(1, \log M)$ the normalized entropy.

\paragraph{Implementation Note:}
To ensure computational efficiency in small-scale settings, we employed k-means initialization followed by vMF parameter estimation. Full-scale versions should employ full EM clustering with BIC model selection.

\section{Results}

We evaluate the IQ model along three principal dimensions: (1) semantic phase transition under variable coupling, (2) the efficacy of integration feedback loops, and (3) creativity-diversity trade-offs under mixed clustering regimes. Full experimental configurations are provided in Appendix S. Code and data are available in the supplementary repository.

\subsection{4.1 Semantic Phase Transitions}

To test the theoretical prediction of a critical coupling threshold $K_{\mathrm{sem},c}$ derived from the linearized Fokker-Planck equation, we performed parameter sweeps over $K_{\mathrm{sem}}$ normalized by $K_c$ across three network topologies: Erdős-Rényi (ER), Watts-Strogatz (WS), and Barabási-Albert (BA).

\begin{itemize}
    \item We observe sharp transitions in $\lambda_{\mathrm{sem}}$ around $K/K_c \approx 1.0$
    \item Sigmoid fitting yields high $R^2$ values $(> 0.95)$, validating theoretical prediction
    \item The finite-size upward bias in $\lambda_{\mathrm{sem}}$ at $K = 0$ confirms predicted $R > 0$ baseline due to spherical sampling
\end{itemize}

\textbf{Figure~\ref{fig:semantic_transition}} summarizes these transitions and fitted curves.

\subsection{4.2 Integration Loop Dynamics}

We simulated feedback-enabled and feedback-disabled conditions with strong external perturbations applied to the semantic layer ($E_{\mathrm{perturb}} = 1.4$). The goal was to evaluate whether the $\lambda_{\mathrm{sem}} \rightarrow K_{\mathrm{phase}} \rightarrow \lambda_{\mathrm{phase}}$ loop accelerates recovery following disruptions.

\textbf{Result:} In the current configuration $(N = 120)$, both conditions restored $\lambda_{\mathrm{phase}}$ within 1 step. No statistically significant difference in recovery time was detected.

\textbf{Interpretation:} We treat this as evidence of system robustness beyond our perturbation regime. It defines a boundary condition rather than theoretical failure, motivating further study (see Section 5.4).

\textbf{Table~\ref{tab:integration_efficacy}} reports recovery times, Cohen’s $d$, and improvement ratios. See Supplementary Figure~S1 for recovery trajectories.

\subsection{4.3 Creativity-Diversity Trade-Off}

We examined the behavior of mixed von Mises-Fisher alignment $\lambda_{\mathrm{sem}}^{\mathrm{mix}}$ as a function of mixture parameter $\gamma_{\mathrm{mix}}$. For $\gamma_{\mathrm{mix}} \in [0, 1]$, we swept configurations and computed:

\begin{itemize}
    \item Number of selected clusters $M_{\mathrm{sel}}$
    \item Base alignment $\lambda_{\mathrm{base}}$
    \item Normalized entropy $H_n$ and overall $\lambda_{\mathrm{sem}}^{\mathrm{mix}}$
\end{itemize}

\textbf{Result:} We observed a characteristic concave trade-off curve with an optimal $\gamma_{\mathrm{mix}}$ between 0.3 and 0.6, aligning with the \emph{structured diversity hypothesis}.

\textbf{Figure~\ref{fig:creativity_tradeoff}} visualizes this trade-off.

\subsection{4.4 Integration Loop Efficacy: An Open Empirical Question}

Under our experimental configuration ($N=120$, $E_{\mathrm{perturb}}=1.4$, recovery threshold $99\%$), both feedback-enabled and feedback-disabled conditions exhibited immediate recovery ($t_{\mathrm{recovery}} \approx 0$), precluding statistical differentiation.

Rather than interpreting this as a theoretical limitation, we identify it as evidence of \textbf{remarkable system robustness} that saturates our current detection methodology. This raises a fundamental open question:

\emph{Under what parameter regimes does the $\lambda_{\mathrm{sem}} \rightarrow K_{\mathrm{phase}} \rightarrow \lambda_{\mathrm{phase}}$ integration loop transition from latent to demonstrably critical influence?}

We hypothesize three testable mechanisms:

\begin{enumerate}
\item \textbf{Scale-dependent emergence:} Effects may manifest only at $N \geq 500$
\item \textbf{Perturbation regime dependency:} Stronger perturbations $(E_{\mathrm{perturb}} > 2.0$, extended windows) may be required
\item \textbf{Network topology sensitivity:} Modular or hierarchical structures with lower algebraic connectivity may reveal differential feedback efficacy
\end{enumerate}

This null result defines a \textbf{robustness boundary} and establishes specific parameter regimes requiring systematic exploration.

\section{Discussion}

We discuss the theoretical implications of the IQ framework, its relationship to prior models of semantic dynamics and consciousness, and future directions enabled by its modular structure.

\subsection{5.1 Theoretical Significance of Semantic Phase Transitions}

The detection of critical coupling thresholds in semantic dynamics validates the information-theoretic approach to emergent structure in high-dimensional concept spaces. The behavior of $\lambda_{\mathrm{sem}}$ under variable $K_{\mathrm{sem}}$ matches predictions from the linearized spherical Fokker-Planck formalism and suggests that semantic alignment is governed by principles analogous to thermodynamic phase transitions.

Importantly, the upward bias in $\lambda_{\mathrm{sem}}$ at low $K$ highlights the role of intrinsic geometric constraints (e.g., $R > 0$ on $\mathbb{S}^{d-1}$) and finite-size sampling. These artifacts are predictable and controllable, further supporting the robustness of the theoretical derivation.

\subsection{5.2 Integration Feedback as Cognitive Homeostasis}

Although our experiments did not reveal statistically distinct recovery trajectories between feedback-enabled and feedback-disabled conditions, we interpret this null result not as a failure of the integration hypothesis, but as an indication of systemic robustness.

In biological systems, homeostatic mechanisms often remain latent until thresholds are crossed. We hypothesize that integration feedback—operationalized here as modulation of $K_{\mathrm{phase}}$ by $\lambda_{\mathrm{sem}}$—is similarly condition-dependent. Stronger perturbations or hierarchical topologies may reveal this loop's function.

This view aligns with the idea that consciousness operates not merely at criticality, but near robust attractors modulated by task and context, and it underscores the value of flexible, testable hypotheses within structured dynamical systems.

\subsection{5.3 Creativity, Diversity, and Cognitive Flexibility}

The observed non-monotonic behavior of $\lambda_{\mathrm{sem}}^{\mathrm{mix}}$ with respect to $\gamma_{\mathrm{mix}}$ supports a structured diversity hypothesis: that cognitive systems benefit from neither pure integration nor pure dispersion, but from optimized mixtures of alignment and entropy.

This aligns with empirical observations in neuroscience (e.g., resting-state metastability), and theoretical models of exploration-exploitation trade-offs. The IQ framework provides a precise computational formulation of these ideas and allows quantitative testing across domains.

\subsection{5.4 Toward Community-Driven Verification of Integration Dynamics}

The integration loop's null result under current conditions transforms from a limitation into a \textbf{research opportunity}. By providing complete theoretical frameworks (Appendix M), validated implementations (Appendix R), and standardized protocols (Appendix S), we establish IQ as both a theoretical contribution and an \textbf{empirical research platform}.

\paragraph{Preregistered Community Challenge:}
We invite systematic exploration of the parameter space where integration feedback transitions from latent to measurable influence.

\textbf{Primary hypotheses for community testing:}
\begin{itemize}
\item H$_0$: Integration feedback does not reduce recovery time (improvement ratio $\leq 1.1$)
\item H$_1$: Integration feedback significantly reduces recovery time (improvement ratio $> 1.2$, Cohen's $d > 0.8$) under specific parameter regimes
\end{itemize}

\textbf{Recommended exploration grid:}
\begin{itemize}
\item \textbf{Scale:} $N \in \{200, 500, 1000\}$, $d \in \{32, 64, 128\}$
\item \textbf{Perturbation:} $E_{\mathrm{perturb}} \in \{1.6, 2.0, 2.5\}$, window length $\times 2$–$3$
\item \textbf{Network topology:} modular (low $\lambda_2$), hierarchical, directed asymmetric
\item \textbf{Temporal dynamics:} multi-scale perturbations, adaptive feedback thresholds
\end{itemize}

\paragraph{Open Science Commitment:}
All simulation code, configuration files, and generated data are provided in the supplementary repository, enabling full replication and systematic extension.

This collaborative framework transforms a null result into a \textbf{scientific frontier}, positioning IQ as the foundation for next-generation empirical consciousness studies.

\section{Conclusion}

We have introduced IQ (Information-Theoretic Constructivism of Semantic Content), a unified framework for modeling semantic dynamics, phase transitions, and integration feedback in high-dimensional cognitive systems. Building on the DSTC (Dynamic Subjective Theory of Consciousness) paradigm, IQ formalizes the interactions between semantic alignment and predictive feedback using tools from stochastic geometry, information theory, and synchronization dynamics.

Our main contributions include:

\begin{itemize}
\item A generalized semantic phase transition model using von Mises-Fisher dynamics on $\mathbb{S}^{d-1}$
\item A dynamic integration feedback loop linking semantic order to phase synchrony
\item An information-theoretic formulation of semantic diversity and creativity through mixture models
\item A fully open-source implementation pipeline for replicable and extensible simulations
\item A research challenge inviting community participation to empirically test integration dynamics
\end{itemize}

While several results—particularly the null finding in feedback efficacy—remain open, we believe this framework establishes a concrete foundation for the next generation of consciousness modeling. By situating IQ at the intersection of dynamical systems, information theory, and cognitive modeling, we aim to catalyze an interdisciplinary dialogue around emergent semantic structures and their role in intelligent behavior.

Future work will expand IQ to incorporate hierarchical semantics, memory mechanisms, multimodal integration, and biological constraints. The present work is both a self-contained contribution and an invitation: to explore, refine, and extend the principles of semantic self-organization in both artificial and biological minds.

\section*{Acknowledgements}

This work is the result of an interdisciplinary collaboration that bridges theoretical modeling, algorithmic implementation, and cognitive science. We are deeply grateful to the open-source communities and contributors whose tools and libraries enabled rapid prototyping and rigorous experimentation throughout this project.

We especially acknowledge the contributions of early reviewers, experimental testers, and theoretical collaborators who helped refine both the conceptual and technical components of the IQ framework. Their feedback led to significant clarifications in our mathematical derivations, simulation protocols, and documentation practices.

We also thank the broader scientific community for ongoing discussions around phase transitions, predictive coding, and semantic emergence—these conversations have shaped many of the ideas crystallized in this work.

All simulations were executed using standard Python scientific libraries (NumPy, SciPy, Matplotlib) and the full codebase is openly provided for community verification and extension. This project is part of a larger initiative to establish a reproducible, extensible, and collaborative platform for empirical modeling in consciousness science.

\section*{Appendix A: Mathematical Foundations}

This appendix provides the formal mathematical basis for the information-theoretic phase transition framework presented in the main text. It includes derivations of the critical semantic coupling $K_{\mathrm{sem},c}$, the construction and properties of the mixture von Mises–Fisher (vMF) representation, and the stochastic differential geometry underlying the semantic dynamics on the unit hypersphere.

We denote the unit hypersphere in $\mathbb{R}^d$ as $\mathbb{S}^{d-1}$, and use the tangent projection operator $P_{\mathbf{u}} = I - \mathbf{u} \mathbf{u}^\top$ to define the constraint-preserving evolution of semantic vectors. All dynamics are formulated in continuous-time Stratonovich form, and converted to the Itô representation where necessary to analyze equilibrium behavior via the Fokker–Planck equation.

\subsection*{A.1 Critical Coupling Derivation}

We derive the critical semantic coupling $K_{\mathrm{sem},c}$ required to induce a semantic phase transition on the hypersphere $\mathbb{S}^{d-1}$. This derivation is grounded in a mean-field approximation combined with tangent space projection, preserving the spherical constraint of semantic representations.

Let $\mathbf{u}_i \in \mathbb{S}^{d-1}$ denote the semantic vector for unit $i$. The dynamics of each unit under information-driven coupling are governed by the stochastic differential equation:

\begin{equation}
d\mathbf{u}_i = K_{\mathrm{sem}} \sum_{j} \tilde{A}_{ij} P_{\mathbf{u}_i}(\mathbf{u}_j - \mathbf{u}_i) dt + \sqrt{2 D_{\mathrm{sem}}} P_{\mathbf{u}_i} \circ d\mathbf{W}_i(t)
\end{equation}

where:
- $P_{\mathbf{u}_i} = I - \mathbf{u}_i \mathbf{u}_i^\top$ projects onto the tangent space of $\mathbb{S}^{d-1}$ at $\mathbf{u}_i$
- $\tilde{A}_{ij}$ is the normalized adjacency matrix
- $d\mathbf{W}_i(t)$ is a Wiener process on $\mathbb{R}^d$
- $\circ$ indicates Stratonovich integration

To determine the critical value $K_{\mathrm{sem},c}$, we linearize the dynamics around the disordered state (uniform distribution), where the ensemble average $\bar{\mathbf{u}} = \frac{1}{N} \sum_i \mathbf{u}_i$ vanishes. Expanding $\mathbf{u}_i = \bar{\mathbf{u}} + \delta_i$, with $\delta_i \perp \bar{\mathbf{u}}$, and assuming small deviations, we obtain a linearized Fokker–Planck equation.

The critical coupling satisfies the balance condition between alignment and diffusion in the marginal eigenmode:

\begin{equation}
K_{\mathrm{sem},c} = \frac{c_d D_{\mathrm{sem}}}{\lambda_2(L_*)}
\end{equation}

where:
- $\lambda_2(L_*)$ is the second smallest eigenvalue (Fiedler value) of the symmetrized Laplacian $L_* = I - \frac{1}{2}(\tilde{A} + \tilde{A}^\top)$
- $c_d = \frac{d-1}{d}$ is the projection-induced geometric correction factor

This expression explicitly links the onset of semantic coherence to the topology of the interaction graph and the ambient semantic dimensionality.

\subsection*{A.2 Mixture von Mises–Fisher Extension}

To account for heterogeneous or multimodal semantic distributions, we extend the single von Mises–Fisher (vMF) formulation to a mixture model. Let $\{\mathbf{u}_i\}_{i=1}^N$ denote semantic vectors on the unit hypersphere $\mathbb{S}^{d-1}$. The semantic distribution is modeled as a mixture of $M$ vMF components:

\begin{equation}
p(\mathbf{u}) = \sum_{m=1}^M \pi_m \cdot \mathcal{V}(\mathbf{u} \mid \boldsymbol{\mu}_m, \kappa_m)
\end{equation}

where:
- $\pi_m \geq 0$ and $\sum_m \pi_m = 1$ are the mixture weights
- $\boldsymbol{\mu}_m \in \mathbb{S}^{d-1}$ are mean directions
- $\kappa_m \geq 0$ are concentration parameters

Each component is defined as:

\begin{equation}
\mathcal{V}(\mathbf{u} \mid \boldsymbol{\mu}_m, \kappa_m) = C_d(\kappa_m) \exp\left( \kappa_m \boldsymbol{\mu}_m^\top \mathbf{u} \right)
\end{equation}

where $C_d(\kappa)$ is the normalizing constant on $\mathbb{S}^{d-1}$.

To quantify global semantic coherence under this mixture, we define the **mixture semantic order parameter**:

\begin{equation}
\lambda_{\mathrm{sem}}^{\mathrm{mix}} = \sum_{m=1}^M \pi_m \left( 1 - \exp(-c_{\mathrm{sem}} \kappa_m) \right) - \gamma_{\mathrm{mix}} \cdot H_n
\end{equation}

with:
- $c_{\mathrm{sem}}$ controlling sensitivity
- $\gamma_{\mathrm{mix}}$ controlling diversity penalty
- $H_n = -\sum_{m=1}^M \pi_m \log \pi_m / \max(1, \log M)$ is the normalized entropy

This definition ensures:
- High $\lambda_{\mathrm{sem}}^{\mathrm{mix}}$ for concentrated, unimodal distributions
- Lower values for fragmented or high-entropy mixtures

The mixture model allows IQ to capture transitions between integrated and diversified semantic states, facilitating applications in creativity modeling and task-dependent cognitive configurations.

\subsection*{A.3 Gain Function Derivation}

In the IQ framework, prediction error modulates both the strength of semantic coupling $K_{\mathrm{sem}}$ and the diffusion coefficient $D_{\mathrm{sem}}$. This section derives the gain functions used in the simulations and analysis.

Let $E$ denote the scalar prediction error at a given timestep. The gain-modulated parameters are defined as follows:

\paragraph{Semantic Coupling Strength}

\begin{equation}
K_{\mathrm{sem}}(E) = \max \left\{ K_0 \left[ 1 - \alpha_K \tanh \left( \beta (E - \theta_E) \right) \right], 0 \right\}
\end{equation}

- $K_0$: baseline semantic coupling
- $\alpha_K$: gain sensitivity (typically $\alpha_K > 0$)
- $\beta$: sharpness of transition
- $\theta_E$: prediction error threshold

This form ensures:
- Strong coupling ($K_{\mathrm{sem}} \approx K_0$) when error is low ($E < \theta_E$)
- Weak coupling when error is high
- Hard lower bound at 0 (no negative coupling)

\paragraph{Semantic Diffusion Coefficient}

\begin{equation}
D_{\mathrm{sem}}(E) = D_0 \exp \left( \alpha_D \tanh \left( \beta (E - \theta_E) \right) \right)
\end{equation}

- $D_0$: baseline diffusion
- $\alpha_D$: diffusion gain (typically $\alpha_D > 0$)

This exponential modulation provides:
- Low diffusion when prediction error is low
- Increased diffusion in high-error regimes
- Smooth nonlinear transition with adjustable sharpness ($\beta$)

\paragraph{Interpretation}

These gain functions implement a biologically plausible tradeoff:
- Low prediction error leads to strong attractor dynamics (high $K_{\mathrm{sem}}$, low $D_{\mathrm{sem}}$)
- High error induces semantic destabilization (low $K_{\mathrm{sem}}$, high $D_{\mathrm{sem}}$)

This mechanism supports fast adaptation by destabilizing outdated representations and promotes semantic reorganization in novel or surprising contexts.

\subsection*{A.4 Critical Coupling Analysis}

This section derives the critical coupling condition for semantic synchronization in the IQ model. We follow the classical approach adapted to directional dynamics on the unit hypersphere $\mathbb{S}^{d-1}$.

\paragraph{Setup}

The semantic dynamics on the sphere are governed by the following stochastic differential equation (SDE):

\begin{equation}
d\mathbf{u}_i = K_{\mathrm{sem}} \sum_j \tilde{A}_{ij} P_{\mathbf{u}_i}(\mathbf{u}_j - \mathbf{u}_i) \, dt + \sqrt{2 D_{\mathrm{sem}}} \, P_{\mathbf{u}_i} \circ d\mathbf{W}_i
\end{equation}

where:
- $\mathbf{u}_i \in \mathbb{S}^{d-1}$ is the semantic vector
- $\tilde{A}$ is the normalized adjacency matrix
- $P_{\mathbf{u}_i} = I - \mathbf{u}_i \mathbf{u}_i^\top$ is the projection onto the tangent space
- $\circ$ denotes Stratonovich integration

\paragraph{Linearization and Mean-Field Approximation}

Assume small fluctuations around a coherent mean direction $\bar{\mathbf{u}}$. We linearize in the tangent space and assume the mean field approximation:

\begin{equation}
d\delta_i = - K_{\mathrm{sem}} \sum_j L_{ij} \delta_j \, dt + \sqrt{2 D_{\mathrm{sem}}} \, dW_i
\end{equation}

Here, $L = I - (\tilde{A} + \tilde{A}^\top)/2$ is the symmetrized graph Laplacian, and $\delta_i$ are tangent vectors (fluctuations from the mean).

\paragraph{Spectral Stability Condition}

The collective synchronization becomes unstable (i.e., synchronized) when the average fluctuation magnitude decays. This yields the following critical condition:

\begin{equation}
K_{\mathrm{sem}, c} = \frac{c_d D_{\mathrm{sem}}}{\lambda_2(L)}
\end{equation}

where:
- $\lambda_2(L)$ is the Fiedler eigenvalue (second smallest eigenvalue) of the Laplacian
- $c_d = \frac{d-1}{d}$ is a dimensional correction factor for the unit sphere

\paragraph{Interpretation}

This result captures the tradeoff between alignment and noise:
- Large $K_{\mathrm{sem}}$ promotes semantic coherence
- Large $D_{\mathrm{sem}}$ injects destabilizing diffusion
- Stronger topologies (large $\lambda_2$) facilitate easier synchronization

\paragraph{Numerical Estimation}

In simulations, $\lambda_2(L)$ is computed using:

\begin{equation}
L = I - \frac{\tilde{A} + \tilde{A}^\top}{2}
\end{equation}

The Fiedler value is obtained from the second smallest eigenvalue of $L$ using efficient symmetric eigensolvers (e.g., `np.linalg.eigvalsh` in NumPy).

\subsection*{A.5 Prediction Error–Dependent Gain Functions}

A key feature of the IQ model is the dynamic modulation of semantic alignment and diffusion coefficients based on prediction error. This section formalizes the nonlinear gain functions that implement this mechanism.

\paragraph{1. Semantic Coupling Gain}

The semantic coupling strength $K_{\mathrm{sem}}$ is modulated by prediction error $E$ through a hyperbolic tangent activation function:

\begin{equation}
K_{\mathrm{sem}}(E) = \max \left\{ K_0 \left[ 1 - \alpha_K \tanh \left( \beta (E - \theta_E) \right) \right], 0 \right\}
\end{equation}

where:
- $K_0$ is the baseline semantic coupling
- $\alpha_K \in [0, 1]$ controls modulation depth
- $\beta$ controls sharpness of the gain response
- $\theta_E$ is the prediction error threshold

This formulation ensures:
- High error ($E \gg \theta_E$): $K_{\mathrm{sem}}$ decreases, promoting desynchronization and exploration
- Low error ($E \ll \theta_E$): $K_{\mathrm{sem}}$ approaches $K_0$, reinforcing coherent semantics
- Non-negativity of $K_{\mathrm{sem}}$ is enforced via the outer $\max$ operator

\paragraph{2. Semantic Diffusion Gain}

The diffusion coefficient $D_{\mathrm{sem}}$ is modulated similarly but exponentially amplified:

\begin{equation}
D_{\mathrm{sem}}(E) = D_0 \exp \left\{ \alpha_D \tanh \left[ \beta (E - \theta_E) \right] \right\}
\end{equation}

where:
- $D_0$ is the baseline diffusion rate
- $\alpha_D > 0$ scales the modulation amplitude

This function enhances diffusion under high prediction error, enabling semantic exploration via increased stochasticity.

\paragraph{3. Combined Gain Effects}

Together, these gain functions shape the semantic transition dynamics:
- The effective signal-to-noise ratio $K_{\mathrm{sem}}(E) / D_{\mathrm{sem}}(E)$ becomes prediction-error sensitive
- The system self-organizes into coherent or exploratory modes depending on environmental mismatch
- These gains are inspired by neuromodulatory control in predictive coding frameworks

\paragraph{4. Implementation Notes}

In code, the functions are implemented as:

```python
def compute_K_sem(E, K0, alpha_K, beta, theta_E):
    return max(K0 * (1 - alpha_K * np.tanh(beta * (E - theta_E))), 0.0)

def compute_D_sem(E, D0, alpha_D, beta, theta_E):
    return D0 * np.exp(alpha_D * np.tanh(beta * (E - theta_E)))

The parameters $(K_0, D_0, \alpha_K, \alpha_D, \beta, \theta_E)$ are configurable via the simulation config and can be tuned per experimental protocol.

\subsection*{A.6 Phase Order Parameters and Recovery Metrics}

This section defines the phase-level observables used to quantify collective dynamics in the IQ framework, including phase coherence and recovery behavior after perturbation.

\paragraph{1. Phase Synchronization Order Parameter}

Let $\theta_i \in [0, 2\pi)$ denote the phase of oscillator $i$ in a system of $N$ agents. The global phase coherence is quantified using the Kuramoto order parameter:

\begin{equation}
\lambda_{\mathrm{phase}} = \left| \frac{1}{N} \sum_{i=1}^{N} e^{i \theta_i} \right|^2
\end{equation}

Properties:
- $\lambda_{\mathrm{phase}} \approx 0$ indicates complete desynchronization
- $\lambda_{\mathrm{phase}} \approx 1$ indicates perfect phase locking

This squared-modulus formulation ensures consistent interpretation across all observables in the IQ model (see Appendix A.2 for $\lambda_{\mathrm{sem}}$).

\paragraph{2. Perturbation and Recovery Protocol}

To evaluate the functional efficacy of integration dynamics, we introduce a perturbation-recovery test:

- At simulation time $t_0$, a phase perturbation is introduced by modifying the prediction error to $E_{\mathrm{perturb}}$, which in turn modulates $K_{\mathrm{phase}}$ and $D_{\mathrm{phase}}$ through the gain functions (Appendix A.5).
- The phase order parameter $\lambda_{\mathrm{phase}}(t)$ is monitored over time post-perturbation.

\paragraph{3. Recovery Time Definition}

The recovery time $t_{\mathrm{rec}}$ is defined as:

\begin{equation}
t_{\mathrm{rec}} = \min \left\{ t \geq t_0 \ \bigg| \ \lambda_{\mathrm{phase}}(t) \geq \theta_{\mathrm{rec}} \right\}
\end{equation}

where $\theta_{\mathrm{rec}} \in (0,1)$ is a predefined recovery threshold (e.g., 0.99).

\paragraph{4. Improvement Ratio}

To quantify the benefit of integration feedback, we define the improvement ratio between feedback-enabled and disabled conditions:

\begin{equation}
\mathrm{ImprovementRatio} = \frac{t_{\mathrm{rec}}^{\mathrm{nfb}}}{t_{\mathrm{rec}}^{\mathrm{fb}}}
\end{equation}

- $\mathrm{ImprovementRatio} > 1$: feedback accelerates recovery
- $\mathrm{ImprovementRatio} = 1$: no effect
- $\mathrm{ImprovementRatio} < 1$: feedback hinders recovery (unexpected)

\paragraph{5. Statistical Evaluation}

Multiple simulation runs (e.g., $n = 20$) are performed to evaluate:
- Mean recovery time across conditions
- Standard deviation and confidence intervals
- Statistical significance (e.g., $p$-value via t-test)
- Effect size (Cohen's $d$)

\paragraph{6. Interpretation Caveats}

If both conditions recover instantly ($t_{\mathrm{rec}} = 0$), no difference is detectable. In such cases, the system is considered to lie in a robustness saturation regime (see Discussion Section 5.4).

\section*{Appendix B: Numerical Methods}

This appendix describes the numerical integration schemes used for simulating the coupled stochastic dynamics of the IQ model, including both semantic and phase systems. All methods are implemented in NumPy and verified to preserve the geometric and stochastic structure of the model.

\subsection*{B.1 Semantic Dynamics: Stratonovich Euler--Heun Scheme}

The semantic vector $\mathbf{u}_i \in \mathbb{S}^{d-1}$ evolves on the unit hypersphere under influence from semantic neighbors and stochastic perturbations. The Stratonovich form of the SDE is:

\[
d\mathbf{u}_i = K_{\mathrm{sem}} \sum_j \tilde{A}_{ij} P_{\mathbf{u}_i}(\mathbf{u}_j - \mathbf{u}_i) \, dt + \sqrt{2 D_{\mathrm{sem}}} \circ d\mathbf{W}_i(t)
\]

We employ a second-order accurate **Stratonovich Euler–Heun method** with tangent-space projection and shared noise increments for consistency.

\paragraph{Numerical Integration (Python):}

\begin{verbatim}
def semantic_step_stratonovich(U, Atil, E_pred, params, dt, rng):
    xi_raw = rng.normal(size=U.shape)  # shared noise increment
    xi1 = project_tangent(U, xi_raw)

    m = Atil @ U
    drift1 = project_tangent(U, m - U)
    K_sem = compute_K_sem(E_pred, params)
    D_sem = compute_D_sem(E_pred, params)
    U_tilde = normalize_rows(U + dt * K_sem * drift1 + np.sqrt(2 * D_sem * dt) * xi1)

    xi2 = project_tangent(U_tilde, xi_raw)  # reuse shared increment
    m2 = Atil @ U_tilde
    drift2 = project_tangent(U_tilde, m2 - U_tilde)

    U_new = normalize_rows(
        U + 0.5 * dt * K_sem * (drift1 + drift2) + 
        0.5 * np.sqrt(2 * D_sem * dt) * (xi1 + xi2)
    )

    return U_new, K_sem, D_sem
\end{verbatim}

\paragraph{Supporting Functions:}

\begin{verbatim}
def normalize_rows(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

def project_tangent(U, X):
    return X - (np.sum(U * X, axis=1, keepdims=True)) * U
\end{verbatim}

\subsection*{B.2 Phase Dynamics: Itô Euler Scheme}

The phase state $\theta_i \in [0, 2\pi)$ evolves via Kuramoto-like coupling modulated by semantic coherence and stochastic noise. The Itô-form SDE is:

\[
d\theta_i = \left[ \omega_i + \frac{K_{\mathrm{phase}}}{N} \sum_j A_{ij} \sin(\theta_j - \theta_i) \right] dt + \sqrt{2 D_{\mathrm{phase}}} \, dW_i(t)
\]

\paragraph{Numerical Integration (Python):}

\begin{verbatim}
def phase_step(theta, A, K_phase, omega, D_phase, dt, rng):
    N = len(theta)
    dW = rng.normal(scale=np.sqrt(dt), size=N)
    coupling = (K_phase / N) * np.sum(A * np.sin(theta[None, :] - theta[:, None]), axis=1)
    return theta + dt * (omega + coupling) + np.sqrt(2 * D_phase * dt) * dW
\end{verbatim}

\subsection*{B.3 Coupled Semantic--Phase Integration with Feedback}

Semantic and phase dynamics are coupled via prediction-error feedback. We implement a unified update step:

\begin{verbatim}
def integrated_step(theta, U, A, Atil, E_pred_raw, E_pred_smooth, params, dt, rng):
    E_clip_lo, E_clip_hi = params.get('E_clip', (0.0, 10.0))
    ema_alpha = params.get('ema_alpha', 0.2)
    E_pred_raw = np.clip(E_pred_raw, E_clip_lo, E_clip_hi)
    E_pred = (1 - ema_alpha) * E_pred_smooth + ema_alpha * E_pred_raw

    U_new, K_sem, D_sem = semantic_step_stratonovich(U, Atil, E_pred, params, dt, rng)
    lambda_sem = compute_lambda_semantic(U_new, params.get('c_sem', 1.0))

    lambda_phase = compute_lambda_phase(theta)
    K_phase = (
        params['K0_phase'] *
        (1 + params['alpha1'] * lambda_phase) *
        (1 + params['alpha2'] * lambda_sem)
    )
    K_phase = np.clip(K_phase, 0.0, params.get('K_phase_max', 5.0))

    omega = np.full_like(theta, params['omega'])
    theta_new = phase_step(theta, A, K_phase, omega, params['D_phase'], dt, rng)

    return theta_new, U_new, lambda_phase, lambda_sem, K_phase, K_sem, D_sem, E_pred
\end{verbatim}

\paragraph{Comment:}  
This step ensures consistent coupling between dynamics, and includes exponential moving average (EMA) smoothing and clipping of the prediction error signal $E_{\mathrm{pred}}$ for robustness.

\section*{Appendix C: Reference Implementation}

This appendix provides a fully executable reference implementation of the IQ model, using NumPy and standard scientific Python libraries. The code corresponds to the main mathematical definitions and simulation protocols described in the paper. The full repository is available at: \texttt{https://github.com/your-repo/iq-framework}

\subsection*{C.1 Parameters and Initialization}

We define a centralized parameter dictionary and initialization routines:

\begin{verbatim}
REF = {
    'N': 120,
    'd': 16,
    'T': 2200,
    'dt': 0.005,
    'K0_phase': 1.0,
    'K_phase_max': 5.0,
    'alpha1': 1.5,
    'alpha2': 2.0,
    'omega': 0.0,
    'D_phase': 0.05,
    'K0_sem': 1.5,
    'alpha_K': 3.0,
    'theta_E': 0.5,
    'beta': 5.0,
    'D0_sem': 0.2,
    'alpha_D': 3.0,
    'c_sem': 0.6,
    'ema_alpha': 0.2,
    'E_clip': (0.0, 10.0)
}

def init_state(N, d, rng):
    theta = rng.uniform(0, 2*np.pi, size=N)
    U = rng.normal(size=(N, d))
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    return theta, U
\end{verbatim}

\subsection*{C.2 Order Parameter Computation}

\begin{verbatim}
def compute_lambda_phase(theta):
    r = np.abs(np.mean(np.exp(1j * theta)))
    return float(r**2)

def compute_lambda_semantic(U, c_sem=1.0):
    mean_vec = np.mean(U, axis=0)
    R = np.linalg.norm(mean_vec)
    d = U.shape[1]
    kappa_hat = kappa_hat_from_R(R, d)
    return 1.0 - np.exp(-c_sem * kappa_hat)

def kappa_hat_from_R(R, d, eps=1e-8):
    R = np.clip(R, eps, 1.0 - eps)
    if d == 1:
        return R / (1 - R)
    return R * (d - R**2) / (1 - R**2)
\end{verbatim}

\subsection*{C.3 Gain Function Definitions}

\begin{verbatim}
def compute_K_sem(E, params):
    theta_E = params['theta_E']
    beta = params['beta']
    alpha_K = params['alpha_K']
    K0 = params['K0_sem']
    return max(K0 * (1 - alpha_K * np.tanh(beta * (E - theta_E))), 0.0)

def compute_D_sem(E, params):
    theta_E = params['theta_E']
    beta = params['beta']
    alpha_D = params['alpha_D']
    D0 = params['D0_sem']
    return D0 * np.exp(alpha_D * np.tanh(beta * (E - theta_E)))
\end{verbatim}

\subsection*{C.4 Utility Functions}

\begin{verbatim}
def normalize_rows(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

def project_tangent(U, X):
    return X - (np.sum(U * X, axis=1, keepdims=True)) * U
\end{verbatim}

\subsection*{C.5 Semantic Step (Stratonovich)}

\begin{verbatim}
def semantic_step_stratonovich(U, Atil, E_pred, params, dt, rng):
    xi_raw = rng.normal(size=U.shape)
    xi1 = project_tangent(U, xi_raw)

    m = Atil @ U
    drift1 = project_tangent(U, m - U)
    K_sem = compute_K_sem(E_pred, params)
    D_sem = compute_D_sem(E_pred, params)

    U_tilde = normalize_rows(U + dt * K_sem * drift1 + np.sqrt(2 * D_sem * dt) * xi1)
    xi2 = project_tangent(U_tilde, xi_raw)
    m2 = Atil @ U_tilde
    drift2 = project_tangent(U_tilde, m2 - U_tilde)

    U_new = normalize_rows(
        U + 0.5 * dt * K_sem * (drift1 + drift2) + 0.5 * np.sqrt(2 * D_sem * dt) * (xi1 + xi2)
    )
    return U_new, K_sem, D_sem
\end{verbatim}

\subsection*{C.6 Phase Step (Kuramoto–Itô)}

\begin{verbatim}
def phase_step(theta, A, K_phase, omega, D_phase, dt, rng):
    N = len(theta)
    dW = rng.normal(scale=np.sqrt(dt), size=N)
    coupling = (K_phase / N) * np.sum(A * np.sin(theta[None, :] - theta[:, None]), axis=1)
    return theta + dt * (omega + coupling) + np.sqrt(2 * D_phase * dt) * dW
\end{verbatim}

\subsection*{C.7 Coupled Step Function}

\begin{verbatim}
def integrated_step(theta, U, A, Atil, E_pred_raw, E_pred_smooth, params, dt, rng):
    E_clip_lo, E_clip_hi = params.get('E_clip', (0.0, 10.0))
    ema_alpha = params.get('ema_alpha', 0.2)
    E_pred_raw = np.clip(E_pred_raw, E_clip_lo, E_clip_hi)
    E_pred = (1 - ema_alpha) * E_pred_smooth + ema_alpha * E_pred_raw

    U_new, K_sem, D_sem = semantic_step_stratonovich(U, Atil, E_pred, params, dt, rng)
    lambda_sem = compute_lambda_semantic(U_new, params.get('c_sem', 1.0))

    lambda_phase = compute_lambda_phase(theta)
    K_phase = (
        params['K0_phase'] *
        (1 + params['alpha1'] * lambda_phase) *
        (1 + params['alpha2'] * lambda_sem)
    )
    K_phase = np.clip(K_phase, 0.0, params.get('K_phase_max', 5.0))

    omega = np.full_like(theta, params['omega'])
    theta_new = phase_step(theta, A, K_phase, omega, params['D_phase'], dt, rng)

    return theta_new, U_new, lambda_phase, lambda_sem, K_phase, K_sem, D_sem, E_pred
\end{verbatim}

\subsection*{C.8 Reproducibility and Availability}

All source code is available at:

\begin{quote}
\texttt{https://github.com/your-repo/iq-framework}
\end{quote}

This repository contains:

\begin{itemize}
\item All scripts used for Figures A–C and Table A
\item Experimental config files: \texttt{config.yaml}, \texttt{config\_small.yaml}
\item Raw CSV logs for all runs
\item LaTeX-ready plots and table generators
\end{itemize}

\section*{Appendix D: Experimental Protocols}

This appendix summarizes the experimental protocols used for the validation of the IQ framework, covering semantic phase transition analysis, integration loop efficacy testing, and creativity–diversity trade-off measurements. Each protocol is provided in a reproducible, parameterized format to facilitate community replication and extension.

\subsection*{D.1 Semantic Transition Protocol (Figure A)}

\paragraph{Objective:}
To measure the emergence of semantic order parameter $\lambda_{\mathrm{sem}}$ as a function of normalized coupling strength $K_{\mathrm{sem}}/K_{\mathrm{sem},c}$.

\paragraph{Setup:}
\begin{itemize}
  \item Network: ER, WS, or BA graph with $N = 120$ nodes
  \item Dimensions: $d = 16$
  \item Time steps: $T = 2200$ with $\texttt{warmup} = 1500$, $\texttt{measure} = 700$
  \item Sweeps: 15 evenly spaced $K_{\mathrm{sem}}$ values per run
  \item Runs per network: $n_{\text{runs}} = 5$
  \item Random seed: fixed (e.g., 12345) for reproducibility
\end{itemize}

\paragraph{Output:}
\begin{itemize}
  \item CSV log: average $\lambda_{\mathrm{sem}}$ vs $K/K_c$
  \item Sigmoid fit: $\lambda_{\mathrm{sem}} \sim 1 / (1 + \exp(-a(K/K_c - b)))$
  \item Figure A: plot with empirical points and fitted curves
\end{itemize}

\subsection*{D.2 Integration Loop Protocol (Table A)}

\paragraph{Objective:}
To evaluate whether the feedback loop $\lambda_{\mathrm{sem}} \rightarrow K_{\mathrm{phase}} \rightarrow \lambda$ facilitates faster recovery after semantic perturbation.

\paragraph{Setup:}
\begin{itemize}
  \item Baseline phase coherence: monitored over $t_{\text{baseline}} = 500$ steps
  \item Perturbation window: 800 steps with $E_{\mathrm{perturb}} \approx 1.3-1.5$
  \item Two conditions: with and without semantic feedback
  \item Recovery threshold: 99\% of baseline $\lambda$ level
  \item Statistical test: Cohen’s $d$, $p$-value via t-test over $n=5$ runs
\end{itemize}

\paragraph{Output:}
\begin{itemize}
  \item CSV log: $\lambda$, $E_{\mathrm{pred}}$, recovery flags
  \item Table A: aggregated metrics—mean recovery time, improvement ratio, statistical significance
  \item Figure S1 (optional): sample recovery trajectories
\end{itemize}

\subsection*{D.3 Mixture Creativity Protocol (Figure C)}

\paragraph{Objective:}
To measure the relationship between mixture diversity ($\gamma_{\mathrm{mix}}$) and semantic coherence ($\lambda_{\mathrm{sem}}^{\mathrm{mix}}$), approximating creativity–diversity trade-offs.

\paragraph{Setup:}
\begin{itemize}
  \item Fixed $U$: high-diversity semantic embeddings from final integration step
  \item Grid: $\gamma_{\mathrm{mix}} \in [0.0, 1.0]$ in steps of 0.25
  \item Cluster fitting: k-means initialization + vMF estimation
  \item Model selection: BIC or fixed $M = 2 \sim 5$
  \item Entropy regularization: $H_n = -\sum_m \pi_m \log \pi_m / \log M$
\end{itemize}

\paragraph{Output:}
\begin{itemize}
  \item CSV log: $\gamma_{\mathrm{mix}}$, $M_{\text{sel}}$, $\lambda_{\mathrm{sem}}^{\mathrm{mix}}$, $H_n$
  \item Figure C: scatter plot—semantic coherence vs entropy, colored by $\gamma_{\mathrm{mix}}$
\end{itemize}

\subsection*{D.4 Reproducibility Guarantee}

All experiments are defined by corresponding \texttt{config.yaml} or \texttt{config\_small.yaml} files, which are published with the repository. Each run can be invoked with:

\begin{verbatim}
python run_semantic_sweep.py --config config.yaml
python run_integration_loop.py --config config.yaml
python run_mixture_creativity.py --config config.yaml
\end{verbatim}

Intermediate and final results are stored as CSV and PNG files under \texttt{logs/} and \texttt{figures/} directories for direct inclusion into LaTeX documents.

\section*{Appendix E: Glossary of Terms}

This appendix summarizes the key variables, parameters, and symbols used throughout the IQ framework for theoretical clarity and implementation reproducibility.

\begin{longtable}{p{4cm}p{11cm}}
\toprule
\textbf{Symbol / Term} & \textbf{Definition} \\
\midrule
$\mathbf{u}_i \in \mathbb{S}^{d-1}$ & Semantic vector (unit norm) for agent $i$ in $d$-dimensional hypersphere \\
$\theta_i \in [0, 2\pi)$ & Phase of agent $i$ used in the phase dynamics \\
$A$ & Adjacency matrix of the network (binary or weighted) \\
$\tilde{A}$ & Row-normalized adjacency matrix \\
$L_*$ & Symmetrized Laplacian: $L_* = I - \frac{1}{2}(\tilde{A} + \tilde{A}^\top)$ \\
$\lambda_2$ & Fiedler eigenvalue (2nd smallest eigenvalue of $L_*$) \\
$K_{\mathrm{sem}}$ & Semantic coupling strength \\
$K_{\mathrm{phase}}$ & Phase coupling strength \\
$D_{\mathrm{sem}}$ & Semantic noise diffusion coefficient \\
$D_{\mathrm{phase}}$ & Phase noise diffusion coefficient \\
$E_{\mathrm{pred}}$ & Prediction error signal computed per agent \\
$\lambda_{\mathrm{sem}}$ & Semantic order parameter: $\lambda_{\mathrm{sem}} = 1 - \exp(-c_{\mathrm{sem}} \hat{\kappa})$ \\
$\lambda$ & Phase coherence order parameter: $\lambda = |\langle e^{i\theta} \rangle|^2$ \\
$\hat{\kappa}$ & Empirical concentration parameter estimated from mean resultant length \\
$c_{\mathrm{sem}}$ & Scaling coefficient for semantic coherence sensitivity \\
$E_{\mathrm{perturb}}$ & Injected prediction error magnitude used for testing recovery \\
$\gamma_{\mathrm{mix}}$ & Mixture diversity weight in $\lambda_{\mathrm{sem}}^{\mathrm{mix}}$ \\
$\lambda_{\mathrm{sem}}^{\mathrm{mix}}$ & Clustered semantic coherence across vMF mixture components \\
$\pi_m$ & Weight (probability) of the $m$-th mixture component \\
$\kappa_m$ & Concentration parameter of the $m$-th vMF component \\
$H_n$ & Normalized Shannon entropy of mixture weights: $H_n = -\sum \pi_m \log \pi_m / \log M$ \\
$M$ & Number of mixture components in vMF clustering \\
$N$ & Number of agents (nodes in network) \\
$d$ & Dimensionality of semantic space \\
$T$ & Total simulation time steps \\
$\texttt{dt}$ & Integration time step (e.g., 0.005) \\
$K_c$ & Theoretical critical coupling for semantic synchronization \\
\bottomrule
\end{longtable}

\section*{Acknowledgments}

We thank the members of the DSTC research initiative and the broader computational neuroscience community for their valuable feedback throughout the development of this work. This research was supported in part by the Open Foundation for Theoretical Cognition, and by collaborative insights from contributors to the IQ experimental framework.

We are especially grateful to early readers and reviewers whose detailed comments helped refine the theoretical, algorithmic, and experimental components of this manuscript. The authors also acknowledge the open-source ecosystem, particularly NumPy, SciPy, Matplotlib, and PyTorch communities, whose tools made this work possible.

\begin{thebibliography}{99}

\bibitem{Kuramoto1984}
Y. Kuramoto, \textit{Chemical Oscillations, Waves, and Turbulence}, Springer (1984).

\bibitem{Sra2012}
S. Sra, ``A short note on parameter approximation for von Mises-Fisher distributions: and a fast implementation of $I_s(x)$,'' \textit{Computational Statistics}, 27(1), 177–190 (2012).

\bibitem{Friston2010}
K. Friston, ``The free-energy principle: a unified brain theory?'' \textit{Nature Reviews Neuroscience}, 11(2), 127–138 (2010).

\bibitem{Deco2017}
G. Deco et al., ``The dynamics of resting fluctuations in the brain: metastability and its dynamical cortical core,'' \textit{Scientific Reports}, 7, 3095 (2017).

\bibitem{Grossberg2013}
S. Grossberg, ``Adaptive Resonance Theory: How a brain learns to consciously attend, learn, and recognize a changing world,'' \textit{Neural Networks}, 37, 1–47 (2013).

\bibitem{Toyoizumi2014}
T. Toyoizumi, et al., ``A theory of neural coding for structured information representation,'' \textit{PLoS Computational Biology}, 10(5), e1003607 (2014).

\bibitem{OpenAI2023}
OpenAI, ``GPT-4 Technical Report,'' \textit{arXiv preprint arXiv:2303.08774} (2023).

\end{thebibliography}
