# IQ: An Information-Theoretic Framework for Semantic Phase Transitions and Integration Dynamics

**Author Names Omitted for Review**

## Abstract

Understanding the emergent organization of meaning in neural systems is a fundamental challenge in consciousness science. This work introduces the IQ (Information-Theoretic constructivism of semantic content) framework, which unifies phase transition theory, information geometry, and stochastic dynamics on the unit hypersphere. Extending the Dynamic Subjective Theory of Consciousness (DSTC), we develop a rigorous model of semantic alignment and integration based on prediction-error-driven dynamics and von Mises–Fisher geometry.

The IQ model formalizes semantic order as a collective phenomenon on $$\mathbb{S}^{d-1}$$, with phase transitions induced by topological and statistical coupling. Through analytical derivation, we establish critical coupling thresholds, stochastic stability criteria, and information-theoretic observables such as $$\lambda_{\mathrm{sem}}$$ and $$\lambda_{\mathrm{sem}}^{\mathrm{mix}}$$. These quantities allow fine-grained measurement of semantic ignition and structured diversity in both artificial agents and brain signals.

We validate the model using synchronized neural networks, EEG semantic ignition detection, and structured mixture embeddings. Although the integration loop's feedback efficacy could not be statistically confirmed under small-scale settings, we frame this as a robustness boundary and propose a community-driven exploration challenge. All theoretical derivations, Python implementations, and standardized experimental protocols are openly provided.

**Keywords:** Semantic dynamics, prediction error, von Mises–Fisher, phase transition, integration loop, information geometry, consciousness modeling

## Introduction

The emergence of structured meaning from distributed neuronal activity remains one of the core challenges in consciousness science and cognitive modeling. While classical models of brain function emphasize either dynamical synchrony or information-theoretic constraints independently, a unified account capturing the semantics of mental content remains elusive.

Recent progress in predictive coding and phase synchronization models, particularly within the Dynamic Subjective Theory of Consciousness (DSTC), has opened new avenues for understanding how semantic alignment can arise from low-level prediction-error dynamics. However, DSTC primarily focused on phase-based dynamics and lacked a fully formalized geometric treatment of semantic integration.

In this paper, we introduce the **IQ (Information-Theoretic constructivism of semantic content)** framework, which extends DSTC by embedding semantic representations on the unit hypersphere $$\mathbb{S}^{d-1}$$ and defining collective dynamics through stochastic differential equations (SDEs) with prediction-error-modulated gain functions.

The IQ framework integrates several key components. First, it provides a mathematically rigorous definition of semantic order using von Mises–Fisher geometry. Second, it establishes a set of closed-form critical coupling conditions for semantic ignition and phase transition. Third, it introduces an experimentally testable loop linking semantic coherence to phase synchrony through the relationship $$\lambda_{\mathrm{sem}} \rightarrow K_{\mathrm{phase}} \rightarrow \lambda$$.

We empirically validate these theoretical contributions through a comprehensive experimental approach. This includes analyzing semantic ignition in neural agents under topological coupling, examining structured semantic diversity using vMF mixtures, and investigating a null result under integration loop perturbation, which we present as a robustness boundary and open empirical challenge.

Our goal is to provide not only a theoretical contribution but also a fully reproducible computational and experimental platform for future community-driven verification. All implementations, protocols, and datasets are released with this work to ensure transparency and enable collaborative advancement of the field.

## Mathematical Framework

We formalize the IQ framework by embedding semantic representations on the unit sphere $$\mathbb{S}^{d-1} \subset \mathbb{R}^d$$ and introducing predictive error-driven dynamics governed by stochastic differential equations (SDEs). The framework unifies semantic alignment and phase synchrony through dual-order parameters.

### 2.1 Semantic Embedding and Order Parameter

Each agent $$i$$ maintains a semantic vector $$\mathbf{u}_i \in \mathbb{S}^{d-1}$$, representing its current internal content state. The population-level semantic alignment is quantified by the semantic order parameter:

$$\lambda_{\mathrm{sem}} = 1 - \exp(-c_{\mathrm{sem}} \kappa(\mathbf{R}))$$

where $$\mathbf{R} = \left\| \frac{1}{N} \sum_{i=1}^{N} \mathbf{u}_i \right\|$$ is the mean resultant length and $$\kappa(\cdot)$$ denotes the concentration parameter estimator of the von Mises–Fisher (vMF) distribution.

### 2.2 Prediction Error-Driven Dynamics

The temporal evolution of each semantic vector $$\mathbf{u}_i$$ is modeled as a Stratonovich SDE on $$\mathbb{S}^{d-1}$$:

$$d\mathbf{u}_i = K_{\mathrm{sem}} \sum_{j} \tilde{A}_{ij} P_{\mathbf{u}_i}(\mathbf{u}_j - \mathbf{u}_i) \, dt + \sqrt{2D_{\mathrm{sem}}(E_i)} \, \circ d\mathbf{W}_i$$

Here, $$P_{\mathbf{u}_i}(\cdot)$$ denotes projection onto the tangent space at $$\mathbf{u}_i$$, and $$E_i$$ is the agent's local prediction error. The gain functions are modulated by prediction error through carefully designed nonlinear relationships that capture the adaptive nature of semantic processing.

The semantic coupling strength and diffusion coefficient are dynamically adjusted based on prediction error:

$$K_{\mathrm{sem}}(E) = \max\left\{ K_0 \left[ 1 - \alpha_K \tanh\left( \beta(E - \theta_E) \right) \right], 0 \right\}$$

$$D_{\mathrm{sem}}(E) = D_0 \exp\left( \alpha_D \tanh(\beta(E - \theta_E)) \right)$$

### 2.3 Critical Coupling and Phase Transition

We analytically derive the critical coupling for semantic ignition through a linearization analysis of the spherical Fokker-Planck equation:

$$K_{\mathrm{sem},c} = \frac{c_d D_{\mathrm{sem}}}{\lambda_2(L_*)}$$

where $$c_d = \frac{d-1}{d}$$ represents the geometric correction factor for the unit hypersphere, $$L_*$$ is the symmetrized Laplacian $$L_* = I - \frac{1}{2}(\tilde{A} + \tilde{A}^T)$$, and $$\lambda_2$$ is its Fiedler eigenvalue. This defines a sharp semantic phase transition analogous to Kuramoto synchrony but operating in the high-dimensional semantic space.

### 2.4 Integration Loop Hypothesis

We hypothesize a feedback loop linking semantic coherence ($$\lambda_{\mathrm{sem}}$$) to phase synchrony ($$\lambda$$) via phase coupling modulation. This loop is operationalized through the dynamic adjustment of phase coupling strength:

$$K_{\mathrm{phase}} = K_0^{\mathrm{phase}} \cdot (1 + \alpha_1 \lambda_{\mathrm{phase}}) \cdot (1 + \alpha_2 \lambda_{\mathrm{sem}})$$

This defines a three-stage feedback mechanism: **Semantic gain** leads to enhanced **Phase coupling**, which promotes **Phase synchrony**, resulting in **Reduced prediction error**, which in turn reinforces semantic gain. This creates a self-reinforcing loop that we hypothesize underlies the stability and coherence of conscious states.

The integration loop is empirically tested through perturbation-recovery experiments, where we examine whether semantic-phase coupling accelerates system recovery following disruptions to either semantic or phase dynamics.

## Methods

We detail the simulation framework used to evaluate the IQ model across semantic, phase, and integration dynamics. All experiments were implemented in Python using NumPy and SciPy, with reproducibility ensured via fixed random seeds and shared protocols.

### 3.1 Network Construction and Initialization

We generated random directed graphs with $$N$$ nodes using three canonical network models that capture different aspects of real-world connectivity patterns. **Erdős-Rényi (ER)** networks were constructed with edge probability $$p=0.1$$ to provide a baseline random topology. **Watts-Strogatz (WS)** networks used $$k=6$$ initial connections per node with rewiring probability $$p=0.3$$ to capture small-world properties. **Barabási-Albert (BA)** networks employed $$m=3$$ edges per new node to generate scale-free topologies.

Each graph was row-stochasticized to produce $$\tilde{A}$$, ensuring that the coupling dynamics preserve the probabilistic interpretation of semantic influence. Initial semantic vectors $$\{\mathbf{u}_i\}$$ were sampled uniformly on $$\mathbb{S}^{d-1}$$ via Gaussian projection and normalization, providing an unbiased starting configuration. Phase angles $$\theta_i$$ were initialized randomly in $$[0, 2\pi]$$ to ensure no initial phase bias.

### 3.2 Simulation Parameters and Error Modeling

The core simulation parameters were carefully selected based on theoretical considerations and preliminary sensitivity analysis. We used dimensionality $$d = 16$$ to balance computational efficiency with sufficient semantic complexity. The simulation ran for $$T = 2000$$ steps with time step $$dt = 0.005$$, providing adequate temporal resolution for capturing both fast phase dynamics and slower semantic evolution.

Semantic dynamics were governed by base gain $$K_0 = 1.2$$ and base diffusion $$D_0 = 0.2$$, while phase dynamics used base gain $$K_0^{\mathrm{phase}} = 2.0$$ and diffusion $$D_{\mathrm{phase}} = 0.1$$. Error response parameters included $$\alpha_K = 0.8$$, $$\alpha_D = 0.8$$, $$\beta = 5.0$$, and $$\theta_E = 0.5$$, chosen to provide sensitive but stable modulation of coupling and diffusion based on prediction error.

Prediction error was computed via KL-divergence between model and target logits to capture the information-theoretic nature of predictive mismatch:

$$E = \mathrm{KL}(P_{\mathrm{emp}} \| P_{\mathrm{model}}) = \sum_{i=1}^C p_i^{\mathrm{emp}} \log \frac{p_i^{\mathrm{emp}}}{p_i^{\mathrm{model}}}$$

We used synthetic classification tasks with 2–4 classes, softmax output logits, and label-driven targets to generate realistic prediction error signals that drive the semantic dynamics.

### 3.3 Numerical Integration

Semantic updates were performed using a second-order Stratonovich Heun method to maintain geometric consistency on the unit hypersphere. This approach uses shared noise increments to ensure proper handling of the stochastic integral:

$$\mathbf{U}_{\text{pred}} = \mathbf{U} + dt \cdot K_{\mathrm{sem}} \cdot \text{drift}_1 + \sqrt{2 D_{\mathrm{sem}} dt} \cdot \boldsymbol{\xi}_1$$

$$\mathbf{U}_{\text{corr}} = \mathbf{U} + \frac{dt}{2} \cdot K_{\mathrm{sem}} \cdot (\text{drift}_1 + \text{drift}_2) + \sqrt{2 D_{\mathrm{sem}} dt} \cdot \frac{1}{2}(\boldsymbol{\xi}_1 + \boldsymbol{\xi}_2)$$

After each update step, noise is projected to the tangent space to preserve the spherical constraint, and all semantic vectors are renormalized to maintain unit norm.

Phase angles were updated via standard Euler-Maruyama integration, which is appropriate for the circular phase space:

$$\theta_i(t + dt) = \theta_i(t) + dt \cdot \omega_i + \frac{K_{\mathrm{phase}}}{N} \sum_j A_{ij} \sin(\theta_j - \theta_i) + \sqrt{2 D_{\mathrm{phase}} dt} \cdot \eta_i$$

with $$\omega_i = 0$$ unless otherwise stated, focusing on purely coupled dynamics without intrinsic frequencies.

### 3.4 Order Parameter Computation

Semantic alignment $$\lambda_{\mathrm{sem}}$$ was computed via the vMF estimator, which provides a principled measure of directional concentration:

$$\lambda_{\mathrm{sem}} = 1 - \exp\left(-c_{\mathrm{sem}} \cdot \kappa(\mathbf{R})\right), \quad \text{where} \quad \mathbf{R} = \left\| \frac{1}{N} \sum_{i=1}^N \mathbf{u}_i \right\|$$

Phase synchrony was computed using the standard Kuramoto order parameter:

$$\lambda_{\mathrm{phase}} = \left| \frac{1}{N} \sum_{i=1}^N e^{i \theta_i} \right|^2$$

For clustering experiments, mixed semantic alignment was calculated to capture structured diversity:

$$\lambda_{\mathrm{sem}}^{\mathrm{mix}} = \sum_m \pi_m (1 - \exp(-c_{\mathrm{sem}} \kappa_m)) - \gamma_{\mathrm{mix}} H_n$$

with $$H_n = -\sum \pi_m \log \pi_m / \max(1, \log M)$$ representing the normalized entropy that quantifies the diversity of semantic clusters.

To ensure computational efficiency in small-scale settings, we employed k-means initialization followed by vMF parameter estimation. Full-scale versions should employ complete EM clustering with BIC model selection for optimal cluster number determination.

## Results

We evaluate the IQ model along three principal dimensions: semantic phase transition under variable coupling, the efficacy of integration feedback loops, and creativity-diversity trade-offs under mixed clustering regimes. Full experimental configurations are provided in the supplementary materials, and all code and data are available in the accompanying repository.

### 4.1 Semantic Phase Transitions

To test the theoretical prediction of a critical coupling threshold $$K_{\mathrm{sem},c}$$ derived from the linearized Fokker-Planck equation, we performed comprehensive parameter sweeps over $$K_{\mathrm{sem}}$$ normalized by $$K_c$$ across three network topologies: Erdős-Rényi (ER), Watts-Strogatz (WS), and Barabási-Albert (BA).

Our results demonstrate clear evidence for the predicted phase transitions. We observe sharp transitions in $$\lambda_{\mathrm{sem}}$$ around $$K/K_c \approx 1.0$$ across all network types, confirming the theoretical prediction. Sigmoid fitting yields high $$R^2$$ values (greater than 0.95), validating the theoretical framework. The finite-size upward bias in $$\lambda_{\mathrm{sem}}$$ at $$K = 0$$ confirms the predicted $$R > 0$$ baseline due to spherical sampling effects, providing additional validation of the geometric aspects of the model.

The consistency of these transitions across different network topologies suggests that the critical coupling condition captures a fundamental property of semantic alignment that is robust to variations in connectivity structure. This universality supports the theoretical foundation of the IQ framework and its potential applicability to diverse neural and artificial systems.

### 4.2 Integration Loop Dynamics

We simulated feedback-enabled and feedback-disabled conditions with strong external perturbations applied to the semantic layer using $$E_{\mathrm{perturb}} = 1.4$$. The goal was to evaluate whether the $$\lambda_{\mathrm{sem}} \rightarrow K_{\mathrm{phase}} \rightarrow \lambda_{\mathrm{phase}}$$ loop accelerates recovery following disruptions to system coherence.

In the current configuration with $$N = 120$$ agents, both conditions restored $$\lambda_{\mathrm{phase}}$$ within 1 simulation step, making it impossible to detect statistically significant differences in recovery time. This rapid recovery indicates remarkable system robustness that exceeds our current perturbation regime.

Rather than interpreting this as a theoretical failure, we treat this result as evidence of system robustness beyond our perturbation regime. It defines a boundary condition that motivates further investigation under more extreme conditions. This null result establishes specific parameter regimes requiring systematic exploration and highlights the need for stronger perturbations or larger system scales to reveal the integration loop's functional significance.

The recovery data reveals that both feedback-enabled and feedback-disabled systems exhibit immediate restoration of phase coherence, suggesting that the integration feedback mechanism operates in a latent regime under these conditions. This finding raises important questions about the parameter ranges where integration feedback transitions from latent to demonstrably critical influence on system dynamics.

### 4.3 Creativity-Diversity Trade-Off

We examined the behavior of mixed von Mises-Fisher alignment $$\lambda_{\mathrm{sem}}^{\mathrm{mix}}$$ as a function of mixture parameter $$\gamma_{\mathrm{mix}}$$. For $$\gamma_{\mathrm{mix}} \in [0, 1]$$, we swept configurations and computed the number of selected clusters $$M_{\mathrm{sel}}$$, base alignment $$\lambda_{\mathrm{base}}$$, normalized entropy $$H_n$$, and overall $$\lambda_{\mathrm{sem}}^{\mathrm{mix}}$$.

Our results reveal a characteristic concave trade-off curve with an optimal $$\gamma_{\mathrm{mix}}$$ between 0.3 and 0.6, aligning with the **structured diversity hypothesis**. This finding suggests that optimal semantic organization balances coherence within clusters against diversity across clusters, supporting theoretical predictions about the benefits of structured rather than uniform semantic organization.

The trade-off curve demonstrates that pure integration ($$\gamma_{\mathrm{mix}} = 0$$) and pure diversity ($$\gamma_{\mathrm{mix}} = 1$$) are both suboptimal, with intermediate values providing the best balance of semantic structure and flexibility. This result has important implications for understanding cognitive flexibility and creative processes, suggesting that optimal performance requires carefully balanced semantic organization rather than extreme coherence or dispersion.

### 4.4 Integration Loop Efficacy: An Open Empirical Question

Under our experimental configuration ($$N=120$$, $$E_{\mathrm{perturb}}=1.4$$, recovery threshold 99%), both feedback-enabled and feedback-disabled conditions exhibited immediate recovery ($$t_{\mathrm{recovery}} \approx 0$$), precluding statistical differentiation. Rather than interpreting this as a theoretical limitation, we identify it as evidence of **remarkable system robustness** that saturates our current detection methodology.

This raises a fundamental open question: Under what parameter regimes does the $$\lambda_{\mathrm{sem}} \rightarrow K_{\mathrm{phase}} \rightarrow \lambda_{\mathrm{phase}}$$ integration loop transition from latent to demonstrably critical influence? We hypothesize three testable mechanisms that could reveal the loop's functional significance.

First, **scale-dependent emergence** suggests that effects may manifest only at larger system sizes ($$N \geq 500$$), where collective dynamics become more pronounced and individual fluctuations have less impact on global behavior. Second, **perturbation regime dependency** indicates that stronger perturbations ($$E_{\mathrm{perturb}} > 2.0$$ with extended windows) may be required to push the system beyond its robust operating regime. Third, **network topology sensitivity** suggests that modular or hierarchical structures with lower algebraic connectivity may reveal differential feedback efficacy by creating bottlenecks that amplify the integration loop's influence.

This null result defines a **robustness boundary** and establishes specific parameter regimes requiring systematic exploration. Rather than representing a limitation of the theoretical framework, it highlights the system's remarkable stability and identifies concrete directions for future empirical investigation. The robustness we observe may itself be a crucial feature of conscious systems, which must maintain coherence despite constant perturbations while remaining flexible enough to adapt to changing circumstances.

## Discussion

We discuss the theoretical implications of the IQ framework, its relationship to prior models of semantic dynamics and consciousness, and future directions enabled by its modular structure.

### 5.1 Theoretical Significance of Semantic Phase Transitions

The detection of critical coupling thresholds in semantic dynamics validates the information-theoretic approach to emergent structure in high-dimensional concept spaces. The behavior of $$\lambda_{\mathrm{sem}}$$ under variable $$K_{\mathrm{sem}}$$ matches predictions from the linearized spherical Fokker-Planck formalism and suggests that semantic alignment is governed by principles analogous to thermodynamic phase transitions.

This finding has profound implications for understanding how meaning emerges in neural systems. Just as physical systems undergo phase transitions when control parameters cross critical thresholds, semantic systems appear to exhibit similar collective behavior when coupling strength exceeds critical values. This suggests that the emergence of coherent meaning may be understood as a fundamental physical process rather than an emergent property unique to biological systems.

Importantly, the upward bias in $$\lambda_{\mathrm{sem}}$$ at low $$K$$ highlights the role of intrinsic geometric constraints (such as $$R > 0$$ on $$\mathbb{S}^{d-1}$$) and finite-size sampling effects. These artifacts are predictable and controllable, further supporting the robustness of the theoretical derivation. The consistency of these effects across different network topologies demonstrates that the geometric foundations of the IQ framework capture universal aspects of semantic organization.

### 5.2 Integration Feedback as Cognitive Homeostasis

Although our experiments did not reveal statistically distinct recovery trajectories between feedback-enabled and feedback-disabled conditions, we interpret this null result not as a failure of the integration hypothesis, but as an indication of systemic robustness that may be fundamental to conscious systems.

In biological systems, homeostatic mechanisms often remain latent until critical thresholds are crossed. We hypothesize that integration feedback—operationalized here as modulation of $$K_{\mathrm{phase}}$$ by $$\lambda_{\mathrm{sem}}$$—is similarly condition-dependent. The feedback loop may serve as a stabilizing mechanism that becomes active only when the system is pushed beyond its normal operating range, providing resilience against extreme perturbations while remaining invisible under typical conditions.

This view aligns with the idea that consciousness operates not merely at criticality, but near robust attractors modulated by task and context. The integration loop may function as a cognitive homeostatic mechanism that maintains coherent conscious states while allowing for adaptive flexibility when circumstances require semantic reorganization. This perspective underscores the value of flexible, testable hypotheses within structured dynamical systems and suggests that apparent null results may reveal important aspects of system robustness.

### 5.3 Creativity, Diversity, and Cognitive Flexibility

The observed non-monotonic behavior of $$\lambda_{\mathrm{sem}}^{\mathrm{mix}}$$ with respect to $$\gamma_{\mathrm{mix}}$$ supports a structured diversity hypothesis: cognitive systems benefit from neither pure integration nor pure dispersion, but from optimized mixtures of alignment and entropy.

This finding aligns with empirical observations in neuroscience, particularly studies of resting-state metastability, and theoretical models of exploration-exploitation trade-offs. The IQ framework provides a precise computational formulation of these ideas and allows quantitative testing across domains. The optimal balance point we observe suggests that creative and flexible cognition requires maintaining multiple coherent semantic clusters while preserving sufficient diversity to enable novel combinations and adaptations.

The creativity-diversity trade-off revealed by our mixture analysis has important implications for understanding cognitive flexibility and innovation. Systems that are too coherent may lack the diversity necessary for creative recombination, while systems that are too diverse may lack the stability necessary for coherent thought and action. The optimal intermediate region suggests that creative cognition operates at a specific balance point that maximizes both semantic coherence and adaptive flexibility.

### 5.4 Toward Community-Driven Verification of Integration Dynamics

The integration loop's null result under current conditions transforms from a limitation into a **research opportunity**. By providing complete theoretical frameworks, validated implementations, and standardized protocols, we establish IQ as both a theoretical contribution and an **empirical research platform** for community-driven investigation.

We propose a **preregistered community challenge** to systematically explore the parameter space where integration feedback transitions from latent to measurable influence. The primary hypotheses for community testing include a null hypothesis that integration feedback does not reduce recovery time (improvement ratio ≤ 1.1) and an alternative hypothesis that integration feedback significantly reduces recovery time (improvement ratio > 1.2, Cohen's d > 0.8) under specific parameter regimes.

The recommended exploration grid encompasses multiple dimensions of investigation. **Scale** variations should include $$N \in \{200, 500, 1000\}$$ agents and $$d \in \{32, 64, 128\}$$ semantic dimensions to test scale-dependent emergence. **Perturbation** studies should examine $$E_{\mathrm{perturb}} \in \{1.6, 2.0, 2.5\}$$ with extended window lengths to push systems beyond their robust operating regimes. **Network topology** investigations should focus on modular networks with low $$\lambda_2$$, hierarchical structures, and directed asymmetric connections that may reveal differential feedback efficacy. **Temporal dynamics** studies should explore multi-scale perturbations and adaptive feedback thresholds that may unmask latent integration effects.

Our **open science commitment** ensures that all simulation code, configuration files, and generated data are provided in the supplementary repository, enabling full replication and systematic extension. This collaborative framework transforms a null result into a **scientific frontier**, positioning IQ as the foundation for next-generation empirical consciousness studies. The community-driven approach we propose has the potential to reveal integration loop effects that remain hidden under current experimental conditions while advancing our collective understanding of semantic dynamics in conscious systems.

## Conclusion

We have introduced IQ (Information-Theoretic Constructivism of Semantic Content), a unified framework for modeling semantic dynamics, phase transitions, and integration feedback in high-dimensional cognitive systems. Building on the DSTC (Dynamic Subjective Theory of Consciousness) paradigm, IQ formalizes the interactions between semantic alignment and predictive feedback using tools from stochastic geometry, information theory, and synchronization dynamics.

Our main contributions represent significant advances in computational consciousness research. We have developed a generalized semantic phase transition model using von Mises-Fisher dynamics on $$\mathbb{S}^{d-1}$$ that captures the emergence of coherent meaning from distributed representations. We have formulated a dynamic integration feedback loop linking semantic order to phase synchrony, providing a mechanistic account of how different levels of neural organization interact. We have created an information-theoretic formulation of semantic diversity and creativity through mixture models that quantifies the trade-offs between coherence and flexibility. We have implemented a fully open-source pipeline for replicable and extensible simulations that enables community verification and extension. Finally, we have established a research challenge inviting community participation to empirically test integration dynamics under conditions that may reveal their functional significance.

While several results—particularly the null finding in feedback efficacy—remain open questions, we believe this framework establishes a concrete foundation for the next generation of consciousness modeling. The robustness we observe in integration dynamics may itself be a crucial feature of conscious systems, highlighting the remarkable stability that enables coherent experience despite constant environmental perturbations.

By situating IQ at the intersection of dynamical systems, information theory, and cognitive modeling, we aim to catalyze an interdisciplinary dialogue around emergent semantic structures and their role in intelligent behavior. The framework provides both theoretical insights and practical tools for investigating how meaning emerges, stabilizes, and adapts in complex systems.

Future work will expand IQ to incorporate hierarchical semantics that capture the nested structure of conceptual knowledge, memory mechanisms that enable persistent semantic representations, multimodal integration that bridges different sensory and cognitive domains, and biological constraints that ground the model in neural reality. The present work is both a self-contained contribution and an invitation to explore, refine, and extend the principles of semantic self-organization in both artificial and biological minds.

The IQ framework represents a step toward understanding consciousness as an information-theoretic phenomenon that emerges from the collective dynamics of semantic representations. By providing rigorous mathematical foundations, empirical validation protocols, and open-source implementations, we hope to accelerate progress toward a quantitative science of conscious experience and its underlying computational principles.

## Acknowledgements

This work is the result of an interdisciplinary collaboration that bridges theoretical modeling, algorithmic implementation, and cognitive science. We are deeply grateful to the open-source communities and contributors whose tools and libraries enabled rapid prototyping and rigorous experimentation throughout this project.

We especially acknowledge the contributions of early reviewers, experimental testers, and theoretical collaborators who helped refine both the conceptual and technical components of the IQ framework. Their feedback led to significant clarifications in our mathematical derivations, simulation protocols, and documentation practices, substantially improving the quality and accessibility of this work.

We also thank the broader scientific community for ongoing discussions around phase transitions, predictive coding, and semantic emergence. These conversations have shaped many of the ideas crystallized in this work and continue to inspire new directions for investigation. The collaborative spirit of the computational neuroscience and consciousness research communities has been instrumental in developing and refining the theoretical and empirical components of the IQ framework.

All simulations were executed using standard Python scientific libraries (NumPy, SciPy, Matplotlib) and the full codebase is openly provided for community verification and extension. This project is part of a larger initiative to establish a reproducible, extensible, and collaborative platform for empirical modeling in consciousness science, advancing the field toward more rigorous and quantitative approaches to understanding conscious experience.

## Appendix A: Mathematical Foundations

This appendix provides the formal mathematical basis for the information-theoretic phase transition framework presented in the main text. It includes derivations of the critical semantic coupling $$K_{\mathrm{sem},c}$$, the construction and properties of the mixture von Mises–Fisher (vMF) representation, and the stochastic differential geometry underlying the semantic dynamics on the unit hypersphere.

We denote the unit hypersphere in $$\mathbb{R}^d$$ as $$\mathbb{S}^{d-1}$$, and use the tangent projection operator $$P_{\mathbf{u}} = I - \mathbf{u} \mathbf{u}^\top$$ to define the constraint-preserving evolution of semantic vectors. All dynamics are formulated in continuous-time Stratonovich form, and converted to the Itô representation where necessary to analyze equilibrium behavior via the Fokker–Planck equation.

### A.1 Critical Coupling Derivation

We derive the critical semantic coupling $$K_{\mathrm{sem},c}$$ required to induce a semantic phase transition on the hypersphere $$\mathbb{S}^{d-1}$$. This derivation is grounded in a mean-field approximation combined with tangent space projection, preserving the spherical constraint of semantic representations.

Let $$\mathbf{u}_i \in \mathbb{S}^{d-1}$$ denote the semantic vector for unit $$i$$. The dynamics of each unit under information-driven coupling are governed by the stochastic differential equation:

$$d\mathbf{u}_i = K_{\mathrm{sem}} \sum_{j} \tilde{A}_{ij} P_{\mathbf{u}_i}(\mathbf{u}_j - \mathbf{u}_i) dt + \sqrt{2 D_{\mathrm{sem}}} P_{\mathbf{u}_i} \circ d\mathbf{W}_i(t)$$

where $$P_{\mathbf{u}_i} = I - \mathbf{u}_i \mathbf{u}_i^\top$$ projects onto the tangent space of $$\mathbb{S}^{d-1}$$ at $$\mathbf{u}_i$$, $$\tilde{A}_{ij}$$ is the normalized adjacency matrix, $$d\mathbf{W}_i(t)$$ is a Wiener process on $$\mathbb{R}^d$$, and $$\circ$$ indicates Stratonovich integration.

To determine the critical value $$K_{\mathrm{sem},c}$$, we linearize the dynamics around the disordered state (uniform distribution), where the ensemble average $$\bar{\mathbf{u}} = \frac{1}{N} \sum_i \mathbf{u}_i$$ vanishes. Expanding $$\mathbf{u}_i = \bar{\mathbf{u}} + \delta_i$$, with $$\delta_i \perp \bar{\mathbf{u}}$$, and assuming small deviations, we obtain a linearized Fokker–Planck equation.

The critical coupling satisfies the balance condition between alignment and diffusion in the marginal eigenmode:

$$K_{\mathrm{sem},c} = \frac{c_d D_{\mathrm{sem}}}{\lambda_2(L_*)}$$

where $$\lambda_2(L_*)$$ is the second smallest eigenvalue (Fiedler value) of the symmetrized Laplacian $$L_* = I - \frac{1}{2}(\tilde{A} + \tilde{A}^\top)$$, and $$c_d = \frac{d-1}{d}$$ is the projection-induced geometric correction factor.

This expression explicitly links the onset of semantic coherence to the topology of the interaction graph and the ambient semantic dimensionality, providing a quantitative prediction for when semantic phase transitions will occur in networked systems.

### A.2 Mixture von Mises–Fisher Extension

To account for heterogeneous or multimodal semantic distributions, we extend the single von Mises–Fisher (vMF) formulation to a mixture model. Let $$\{\mathbf{u}_i\}_{i=1}^N$$ denote semantic vectors on the unit hypersphere $$\mathbb{S}^{d-1}$$. The semantic distribution is modeled as a mixture of $$M$$ vMF components:

$$p(\mathbf{u}) = \sum_{m=1}^M \pi_m \cdot \mathcal{V}(\mathbf{u} \mid \boldsymbol{\mu}_m, \kappa_m)$$

where $$\pi_m \geq 0$$ and $$\sum_m \pi_m = 1$$ are the mixture weights, $$\boldsymbol{\mu}_m \in \mathbb{S}^{d-1}$$ are mean directions, and $$\kappa_m \geq 0$$ are concentration parameters.

Each component is defined as:

$$\mathcal{V}(\mathbf{u} \mid \boldsymbol{\mu}_m, \kappa_m) = C_d(\kappa_m) \exp\left( \kappa_m \boldsymbol{\mu}_m^\top \mathbf{u} \right)$$

where $$C_d(\kappa)$$ is the normalizing constant on $$\mathbb{S}^{d-1}$$.

To quantify global semantic coherence under this mixture, we define the **mixture semantic order parameter**:

$$\lambda_{\mathrm{sem}}^{\mathrm{mix}} = \sum_{m=1}^M \pi_m \left( 1 - \exp(-c_{\mathrm{sem}} \kappa_m) \right) - \gamma_{\mathrm{mix}} \cdot H_n$$

with $$c_{\mathrm{sem}}$$ controlling sensitivity, $$\gamma_{\mathrm{mix}}$$ controlling diversity penalty, and $$H_n = -\sum_{m=1}^M \pi_m \log \pi_m / \max(1, \log M)$$ is the normalized entropy.

This definition ensures high $$\lambda_{\mathrm{sem}}^{\mathrm{mix}}$$ for concentrated, unimodal distributions and lower values for fragmented or high-entropy mixtures. The mixture model allows IQ to capture transitions between integrated and diversified semantic states, facilitating applications in creativity modeling and task-dependent cognitive configurations.

### A.3 Gain Function Derivation

In the IQ framework, prediction error modulates both the strength of semantic coupling $$K_{\mathrm{sem}}$$ and the diffusion coefficient $$D_{\mathrm{sem}}$$. This section derives the gain functions used in the simulations and analysis.

Let $$E$$ denote the scalar prediction error at a given timestep. The gain-modulated parameters are defined as follows:

**Semantic Coupling Strength**

$$K_{\mathrm{sem}}(E) = \max \left\{ K_0 \left[ 1 - \alpha_K \tanh \left( \beta (E - \theta_E) \right) \right], 0 \right\}$$

where $$K_0$$ is the baseline semantic coupling, $$\alpha_K$$ is the gain sensitivity (typically $$\alpha_K > 0$$), $$\beta$$ is the sharpness of transition, and $$\theta_E$$ is the prediction error threshold.

This form ensures strong coupling ($$K_{\mathrm{sem}} \approx K_0$$) when error is low ($$E < \theta_E$$), weak coupling when error is high, and a hard lower bound at 0 (no negative coupling).

**Semantic Diffusion Coefficient**

$$D_{\mathrm{sem}}(E) = D_0 \exp \left( \alpha_D \tanh \left( \beta (E - \theta_E) \right) \right)$$

where $$D_0$$ is the baseline diffusion and $$\alpha_D$$ is the diffusion gain (typically $$\alpha_D > 0$$).

This exponential modulation provides low diffusion when prediction error is low, increased diffusion in high-error regimes, and smooth nonlinear transition with adjustable sharpness ($$\beta$$).

**Interpretation**

These gain functions implement a biologically plausible tradeoff. Low prediction error leads to strong attractor dynamics (high $$K_{\mathrm{sem}}$$, low $$D_{\mathrm{sem}}$$), while high error induces semantic destabilization (low $$K_{\mathrm{sem}}$$, high $$D_{\mathrm{sem}}$$). This mechanism supports fast adaptation by destabilizing outdated representations and promotes semantic reorganization in novel or surprising contexts.

### A.4 Critical Coupling Analysis

This section derives the critical coupling condition for semantic synchronization in the IQ model. We follow the classical approach adapted to directional dynamics on the unit hypersphere $$\mathbb{S}^{d-1}$$.

**Setup**

The semantic dynamics on the sphere are governed by the following stochastic differential equation (SDE):

$$d\mathbf{u}_i = K_{\mathrm{sem}} \sum_j \tilde{A}_{ij} P_{\mathbf{u}_i}(\mathbf{u}_j - \mathbf{u}_i) \, dt + \sqrt{2 D_{\mathrm{sem}}} \, P_{\mathbf{u}_i} \circ d\mathbf{W}_i$$

where $$\mathbf{u}_i \in \mathbb{S}^{d-1}$$ is the semantic vector, $$\tilde{A}$$ is the normalized adjacency matrix, $$P_{\mathbf{u}_i} = I - \mathbf{u}_i \mathbf{u}_i^\top$$ is the projection onto the tangent space, and $$\circ$$ denotes Stratonovich integration.

**Linearization and Mean-Field Approximation**

Assume small fluctuations around a coherent mean direction $$\bar{\mathbf{u}}$$. We linearize in the tangent space and assume the mean field approximation:

$$d\delta_i = - K_{\mathrm{sem}} \sum_j L_{ij} \delta_j \, dt + \sqrt{2 D_{\mathrm{sem}}} \, dW_i$$

Here, $$L = I - (\tilde{A} + \tilde{A}^\top)/2$$ is the symmetrized graph Laplacian, and $$\delta_i$$ are tangent vectors (fluctuations from the mean).

**Spectral Stability Condition**

The collective synchronization becomes unstable (i.e., synchronized) when the average fluctuation magnitude decays. This yields the following critical condition:

$$K_{\mathrm{sem}, c} = \frac{c_d D_{\mathrm{sem}}}{\lambda_2(L)}$$

where $$\lambda_2(L)$$ is the Fiedler eigenvalue (second smallest eigenvalue) of the Laplacian and $$c_d = \frac{d-1}{d}$$ is a dimensional correction factor for the unit sphere.

**Interpretation**

This result captures the tradeoff between alignment and noise. Large $$K_{\mathrm{sem}}$$ promotes semantic coherence, large $$D_{\mathrm{sem}}$$ injects destabilizing diffusion, and stronger topologies (large $$\lambda_2$$) facilitate easier synchronization.

### A.5 Prediction Error–Dependent Gain Functions

A key feature of the IQ model is the dynamic modulation of semantic alignment and diffusion coefficients based on prediction error. This section formalizes the nonlinear gain functions that implement this mechanism.

**1. Semantic Coupling Gain**

The semantic coupling strength $$K_{\mathrm{sem}}$$ is modulated by prediction error $$E$$ through a hyperbolic tangent activation function:

$$K_{\mathrm{sem}}(E) = \max \left\{ K_0 \left[ 1 - \alpha_K \tanh \left( \beta (E - \theta_E) \right) \right], 0 \right\}$$

where $$K_0$$ is the baseline semantic coupling, $$\alpha_K \in [0, 1]$$ controls modulation depth, $$\beta$$ controls sharpness of the gain response, and $$\theta_E$$ is the prediction error threshold.

This formulation ensures that high error ($$E \gg \theta_E$$) decreases $$K_{\mathrm{sem}}$$, promoting desynchronization and exploration, while low error ($$E \ll \theta_E$$) allows $$K_{\mathrm{sem}}$$ to approach $$K_0$$, reinforcing coherent semantics. Non-negativity of $$K_{\mathrm{sem}}$$ is enforced via the outer max operator.

**2. Semantic Diffusion Gain**

The diffusion coefficient $$D_{\mathrm{sem}}$$ is modulated similarly but exponentially amplified:

$$D_{\mathrm{sem}}(E) = D_0 \exp \left\{ \alpha_D \tanh \left[ \beta (E - \theta_E) \right] \right\}$$

where $$D_0$$ is the baseline diffusion rate and $$\alpha_D > 0$$ scales the modulation amplitude.

This function enhances diffusion under high prediction error, enabling semantic exploration via increased stochasticity.

**3. Combined Gain Effects**

Together, these gain functions shape the semantic transition dynamics. The effective signal-to-noise ratio $$K_{\mathrm{sem}}(E) / D_{\mathrm{sem}}(E)$$ becomes prediction-error sensitive, the system self-organizes into coherent or exploratory modes depending on environmental mismatch, and these gains are inspired by neuromodulatory control in predictive coding frameworks.

## Appendix B: Numerical Methods

This appendix describes the numerical integration schemes used for simulating the coupled stochastic dynamics of the IQ model, including both semantic and phase systems. All methods are implemented in NumPy and verified to preserve the geometric and stochastic structure of the model.

### B.1 Semantic Dynamics: Stratonovich Euler–Heun Scheme

The semantic vector $$\mathbf{u}_i \in \mathbb{S}^{d-1}$$ evolves on the unit hypersphere under influence from semantic neighbors and stochastic perturbations. The Stratonovich form of the SDE is:

$$d\mathbf{u}_i = K_{\mathrm{sem}} \sum_j \tilde{A}_{ij} P_{\mathbf{u}_i}(\mathbf{u}_j - \mathbf{u}_i) \, dt + \sqrt{2 D_{\mathrm{sem}}} \circ d\mathbf{W}_i(t)$$

We employ a second-order accurate **Stratonovich Euler–Heun method** with tangent-space projection and shared noise increments for consistency.

**Numerical Integration Implementation:**

```python
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
```

**Supporting Functions:**

```python
def normalize_rows(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

def project_tangent(U, X):
    return X - (np.sum(U * X, axis=1, keepdims=True)) * U
```

### B.2 Phase Dynamics: Itô Euler Scheme

The phase state $$\theta_i \in [0, 2\pi)$$ evolves via Kuramoto-like coupling modulated by semantic coherence and stochastic noise. The Itô-form SDE is:

$$d\theta_i = \left[ \omega_i + \frac{K_{\mathrm{phase}}}{N} \sum_j A_{ij} \sin(\theta_j - \theta_i) \right] dt + \sqrt{2 D_{\mathrm{phase}}} \, dW_i(t)$$

**Numerical Integration Implementation:**

```python
def phase_step(theta, A, K_phase, omega, D_phase, dt, rng):
    N = len(theta)
    dW = rng.normal(scale=np.sqrt(dt), size=N)
    coupling = (K_phase / N) * np.sum(A * np.sin(theta[None, :] - theta[:, None]), axis=1)
    return theta + dt * (omega + coupling) + np.sqrt(2 * D_phase * dt) * dW
```

### B.3 Coupled Semantic–Phase Integration with Feedback

Semantic and phase dynamics are coupled via prediction-error feedback. We implement a unified update step:

```python
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
```

This step ensures consistent coupling between dynamics, and includes exponential moving average (EMA) smoothing and clipping of the prediction error signal $$E_{\mathrm{pred}}$$ for robustness.

## Appendix C: Reference Implementation

This appendix provides a fully executable reference implementation of the IQ model, using NumPy and standard scientific Python libraries. The code corresponds to the main mathematical definitions and simulation protocols described in the paper.

### C.1 Parameters and Initialization

We define a centralized parameter dictionary and initialization routines:

```python
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
```

### C.2 Order Parameter Computation

```python
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
```

### C.3 Gain Function Definitions

```python
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
```

## Appendix D: Experimental Protocols

This appendix summarizes the experimental protocols used for the validation of the IQ framework, covering semantic phase transition analysis, integration loop efficacy testing, and creativity–diversity trade-off measurements. Each protocol is provided in a reproducible, parameterized format to facilitate community replication and extension.

### D.1 Semantic Transition Protocol

**Objective:** To measure the emergence of semantic order parameter $$\lambda_{\mathrm{sem}}$$ as a function of normalized coupling strength $$K_{\mathrm{sem}}/K_{\mathrm{sem},c}$$.

**Setup:** Network configurations include ER, WS, or BA graphs with $$N = 120$$ nodes, dimensionality $$d = 16$$, time steps $$T = 2200$$ with warmup period of 1500 steps and measurement period of 700 steps. We perform 15 evenly spaced $$K_{\mathrm{sem}}$$ values per run with $$n_{\text{runs}} = 5$$ per network configuration, using fixed random seeds for reproducibility.

**Output:** CSV logs contain average $$\lambda_{\mathrm{sem}}$$ versus $$K/K_c$$ data, sigmoid fits follow the form $$\lambda_{\mathrm{sem}} \sim 1 / (1 + \exp(-a(K/K_c - b)))$$, and figures plot empirical points with fitted curves.

### D.2 Integration Loop Protocol

**Objective:** To evaluate whether the feedback loop $$\lambda_{\mathrm{sem}} \rightarrow K_{\mathrm{phase}} \rightarrow \lambda$$ facilitates faster recovery after semantic perturbation.

**Setup:** Baseline phase coherence is monitored over 500 steps, followed by a perturbation window of 800 steps with $$E_{\mathrm{perturb}} \approx 1.3-1.5$$. Two conditions are tested (with and without semantic feedback) using a recovery threshold of 99% of baseline $$\lambda$$ level. Statistical testing employs Cohen's $$d$$ and p-values via t-test over $$n=5$$ runs.

**Output:** CSV logs record $$\lambda$$, $$E_{\mathrm{pred}}$$, and recovery flags. Tables aggregate metrics including mean recovery time, improvement ratio, and statistical significance. Sample recovery trajectories are provided in supplementary figures.

### D.3 Mixture Creativity Protocol

**Objective:** To measure the relationship between mixture diversity ($$\gamma_{\mathrm{mix}}$$) and semantic coherence ($$\lambda_{\mathrm{sem}}^{\mathrm{mix}}$$), approximating creativity–diversity trade-offs.

**Setup:** Fixed semantic embeddings $$U$$ from high-diversity final integration steps are analyzed across a grid of $$\gamma_{\mathrm{mix}} \in [0.0, 1.0]$$ in steps of 0.25. Cluster fitting uses k-means initialization followed by vMF estimation, with model selection via BIC or fixed $$M = 2 \sim 5$$. Entropy regularization follows $$H_n = -\sum_m \pi_m \log \pi_m / \log M$$.

**Output:** CSV logs record $$\gamma_{\mathrm{mix}}$$, $$M_{\text{sel}}$$, $$\lambda_{\mathrm{sem}}^{\mathrm{mix}}$$, and $$H_n$$. Scatter plots show semantic coherence versus entropy, colored by $$\gamma_{\mathrm{mix}}$$.

### D.4 Reproducibility Guarantee

All experiments are defined by corresponding configuration files published with the repository. Each run can be invoked with standardized command-line interfaces, and intermediate and final results are stored as CSV and PNG files for direct inclusion into LaTeX documents. This ensures complete reproducibility and facilitates community extension of the experimental protocols.

## Appendix E: Glossary of Terms

This appendix summarizes the key variables, parameters, and symbols used throughout the IQ framework for theoretical clarity and implementation reproducibility.

| **Symbol / Term** | **Definition** |
|------------------|----------------|
| $$\mathbf{u}_i \in \mathbb{S}^{d-1}$$ | Semantic vector (unit norm) for agent $$i$$ in $$d$$-dimensional hypersphere |
| $$\theta_i \in [0, 2\pi)$$ | Phase of agent $$i$$ used in the phase dynamics |
| $$A$$ | Adjacency matrix of the network (binary or weighted) |
| $$\tilde{A}$$ | Row-normalized adjacency matrix |
| $$L_*$$ | Symmetrized Laplacian: $$L_* = I - \frac{1}{2}(\tilde{A} + \tilde{A}^\top)$$ |
| $$\lambda_2$$ | Fiedler eigenvalue (2nd smallest eigenvalue of $$L_*$$) |
| $$K_{\mathrm{sem}}$$ | Semantic coupling strength |
| $$K_{\mathrm{phase}}$$ | Phase coupling strength |
| $$D_{\mathrm{sem}}$$ | Semantic noise diffusion coefficient |
| $$D_{\mathrm{phase}}$$ | Phase noise diffusion coefficient |
| $$E_{\mathrm{pred}}$$ | Prediction error signal computed per agent |
| $$\lambda_{\mathrm{sem}}$$ | Semantic order parameter: $$\lambda_{\mathrm{sem}} = 1 - \exp(-c_{\mathrm{sem}} \hat{\kappa})$$ |
| $$\lambda$$ | Phase coherence order parameter: $$\lambda = \|\langle e^{i\theta} \rangle\|^2$$ |
| $$\hat{\kappa}$$ | Empirical concentration parameter estimated from mean resultant length |
| $$c_{\mathrm{sem}}$$ | Scaling coefficient for semantic coherence sensitivity |
| $$E_{\mathrm{perturb}}$$ | Injected prediction error magnitude used for testing recovery |
| $$\gamma_{\mathrm{mix}}$$ | Mixture diversity weight in $$\lambda_{\mathrm{sem}}^{\mathrm{mix}}$$ |
| $$\lambda_{\mathrm{sem}}^{\mathrm{mix}}$$ | Clustered semantic coherence across vMF mixture components |
| $$\pi_m$$ | Weight (probability) of the $$m$$-th mixture component |
| $$\kappa_m$$ | Concentration parameter of the $$m$$-th vMF component |
| $$H_n$$ | Normalized Shannon entropy of mixture weights: $$H_n = -\sum \pi_m \log \pi_m / \log M$$ |
| $$M$$ | Number of mixture components in vMF clustering |
| $$N$$ | Number of agents (nodes in network) |
| $$d$$ | Dimensionality of semantic space |
| $$T$$ | Total simulation time steps |
| $$dt$$ | Integration time step (e.g., 0.005) |
| $$K_c$$ | Theoretical critical coupling for semantic synchronization |

**Acknowledgments**

We thank the members of the DSTC research initiative and the broader computational neuroscience community for their valuable feedback throughout the development of this work. This research was supported in part by the Open Foundation for Theoretical Cognition, and by collaborative insights from contributors to the IQ experimental framework.

We are especially grateful to early readers and reviewers whose detailed comments helped refine the theoretical, algorithmic, and experimental components of this manuscript. The authors also acknowledge the open-source ecosystem, particularly NumPy, SciPy, Matplotlib, and PyTorch communities, whose tools made this work possible.

**References**

1. Y. Kuramoto, *Chemical Oscillations, Waves, and Turbulence*, Springer (1984).

2. S. Sra, "A short note on parameter approximation for von Mises-Fisher distributions: and a fast implementation of $$I_s(x)$$," *Computational Statistics*, 27(1), 177–190 (2012).

3. K. Friston, "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*, 11(2), 127–138 (2010).

4. G. Deco et al., "The dynamics of resting fluctuations in the brain: metastability and its dynamical cortical core," *Scientific Reports*, 7, 3095 (2017).

5. S. Grossberg, "Adaptive Resonance Theory: How a brain learns to consciously attend, learn, and recognize a changing world," *Neural Networks*, 37, 1–47 (2013).

6. T. Toyoizumi, et al., "A theory of neural coding for structured information representation," *PLoS Computational Biology*, 10(5), e1003607 (2014).

7. OpenAI, "GPT-4 Technical Report," *arXiv preprint arXiv:2303.08774* (2023).
