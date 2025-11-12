# NeuroMHE Documentation

This folder contains comprehensive theoretical documentation for understanding the NeuroMHE framework and related approaches to dynamic state estimation and optimization.

---

## Table of Contents

1. [PSO-PF Theoretical Framework](#pso-pf-theoretical-framework)
2. [NeuroMHE Overview](#neuromhe-overview)
3. [Navigation Guide](#navigation-guide)
4. [For Researchers](#for-researchers)
5. [For Practitioners](#for-practitioners)

---

## PSO-PF Theoretical Framework

The **Predictive PSO-PF (Particle Swarm Optimization with Particle Filter)** framework provides a rigorous approach to dynamic optimization problems where:
- The optimum location changes over time
- Only fitness evaluations are available (no gradients)
- No explicit dynamics model exists

### Key Documents

#### [1. Theoretical Gaps Framework](./PSO-PF/1_Theoretical_Gaps_Framework.md)
**Purpose:** Addresses five critical theoretical gaps in predictive PSO-PF

**Contents:**
1. **Gap 1:** Tight convergence rate bounds with prediction (~24% speedup proven)
2. **Gap 2:** Optimal joint parameter selection (meta-learning framework)
3. **Gap 3:** Multi-modal extension (tracking K optima simultaneously)
4. **Gap 4:** Nonlinear prediction formulation (adaptive model bank)
5. **Gap 5:** Adaptive window selection (change detection + bias-variance optimization)

**Key Contributions:**
- Lyapunov-based convergence analysis
- Bayesian optimization for parameter tuning
- Species-based hierarchical framework
- PF-inspired model weighting
- Kalman-inspired adaptive windowing

**Target Audience:** Researchers working on:
- Dynamic optimization problems
- Evolutionary algorithms in time-varying environments
- Theoretical foundations of PSO and PF

---

#### [2. Rigorous Mathematical Formulation](./PSO-PF/2_Rigorous_Mathematical_Formulation.md)
**Purpose:** Complete mathematical foundation from first principles

**Contents:**
1. **Problem Setup:** Dynamic optimization problem (DOP) definition
2. **State-Space Formulation:** Hidden optimum as latent state [x*, v*]
3. **Particle Filter Framework:** Bayesian state estimation derivation
4. **PSO as Informed Proposal:** Integration insight (dual role of particles)
5. **Personal Best Decay:** Information-theoretic justification
6. **Resampling:** Rigorous treatment with diversity preservation
7. **Prediction Component:** Velocity estimation from history
8. **Complete Algorithm:** Step-by-step specification
9. **Theoretical Properties:** Convergence theorems with proofs
10. **Parameter Guidelines:** Principled selection rules

**Key Contributions:**
- Addresses 5 critical issues (unclear integration, unjustified Gaussian, etc.)
- Derives Gaussian from maximum entropy (not imposed arbitrarily)
- Shows PSO serves as fitness-guided proposal for PF
- Proves consistency, finite-sample bounds, convergence rates

**Target Audience:** Researchers needing:
- Formal mathematical foundations
- Proof sketches and theoretical guarantees
- Rigorous understanding of PSO-PF coupling

---

#### [3. Comparison Analysis](./PSO-PF/3_Comparison_Analysis.md)
**Purpose:** Compare PSO-PF with NeuroMHE and identify synergies

**Contents:**
1. **PSO-PF Main Contributions:** Detailed breakdown of 5 gaps + rigorous formulation
2. **NeuroMHE Approach:** Moving Horizon Estimation with neural weight tuning
3. **Theoretical Comparison:** Stochastic vs. deterministic paradigms
4. **Conceptual Mapping:** Particles ↔ MHE trajectory, weights ↔ covariances, etc.
5. **Unified View:** Both as sequential Bayesian inference approximations
6. **Potential Synergies:** 4 hybrid algorithm designs
7. **Integration Strategies:** How PSO-PF fits this repository
8. **Recommendations:** Documentation, implementation, publications

**Key Insights:**
- **PSO-PF:** Model-free, gradient-free, multi-modal, for DOP benchmarks
- **NeuroMHE:** Model-based, gradient-based, real-time, for robotic estimation
- **Complementary:** Different problem classes, potential for hybrid approaches

**Target Audience:** Researchers interested in:
- Understanding both paradigms
- Selecting appropriate method for their problem
- Developing hybrid algorithms

---

## NeuroMHE Overview

**NeuroMHE (Neural Moving Horizon Estimation)** is an auto-tuning optimal estimator that fuses neural networks with MHE for accurate state estimation in robotic systems.

### Core Components

1. **Moving Horizon Estimation (MHE)**
   - Solves nonlinear program (NLP) over sliding window
   - Balances model fit and measurement fit
   - Real-time capable (<10ms with warm-starting)

2. **Neural Network**
   - Learns optimal weight matrices [P, R, Q]
   - Trained via supervised learning or reinforcement learning
   - Adapts to varying noise characteristics

3. **Kalman Filter-Based Gradient Solver**
   - Efficiently computes ∂(MHE state)/∂(NN parameters)
   - Exploits structure of MHE optimization
   - Enables end-to-end training without differentiating through solver

### Main Paper

**"Neural Moving Horizon Estimation for Robust Flight Control"**
- Published in: IEEE Transactions on Robotics, 2024
- Authors: Bingheng Wang, Zhengtian Ma, Shupeng Lai, Lin Zhao
- DOI: 10.1109/TRO.2023.3331064

### Repository Structure

```
NeuroMHE/
├── SecVII-A (source code)/        # Supervised learning on NeuroBEM data
├── SecVII-B (source code)/        # Online learning in simulation
├── SecVII-C (source code)/        # Real-world quadrotor experiments
├── Documentation/                 # This folder
│   ├── PSO-PF/                    # PSO-PF theoretical framework
│   └── README.md                  # This file
└── README.md                      # Main repository README
```

---

## Navigation Guide

### For Understanding NeuroMHE

1. Start with main [README.md](../README.md)
2. Review paper on IEEE Xplore or arXiv
3. Examine core MHE implementation: `SecVII-A (source code)/Uav_mhe_supervisedlearning.py`
4. Check experiments: SecVII-A (accurate estimation), SecVII-B (online learning), SecVII-C (real-world)

### For Understanding PSO-PF

1. **Quick overview:** Read [Section 1](./PSO-PF/3_Comparison_Analysis.md#part-1-main-contributions-of-pso-pf-theoretical-framework) of comparison analysis
2. **Detailed gaps:** Read [Theoretical Gaps Framework](./PSO-PF/1_Theoretical_Gaps_Framework.md)
3. **Mathematical rigor:** Read [Rigorous Formulation](./PSO-PF/2_Rigorous_Mathematical_Formulation.md)
4. **Implementation:** Follow unified algorithm in §6 of gaps document or §8 of formulation

### For Comparing Both Approaches

1. Read [Part 3: Theoretical Comparison](./PSO-PF/3_Comparison_Analysis.md#part-3-theoretical-comparison-and-potential-synergies) in comparison analysis
2. Review table of differences
3. Examine potential synergies (§3.3)
4. Check decision guide (§2.4: "When to Use Each Approach")

---

## For Researchers

### Research Directions

**PSO-PF Extensions:**
1. Implement and validate convergence rate analysis
2. Meta-learning for parameter selection across DOP benchmarks
3. Multi-modal tracking with species-based approach
4. Nonlinear prediction models (GP, neural networks)
5. Adaptive mechanisms for varying environments

**NeuroMHE Extensions:**
1. Trust-region methods for training (see [TR-NeuroMHE](https://github.com/BinghengNUS/TR-NeuroMHE))
2. Extension to other robotic systems
3. Multi-objective MHE with learned weights
4. Model-free NeuroMHE variants

**Hybrid Approaches:**
1. PSO for NeuroMHE hyperparameter tuning
2. Ensemble MHE with PF-inspired weighting
3. Hybrid prediction (model-based + data-driven)
4. Multi-modal MHE using species formation

### Publication Opportunities

**PSO-PF Series (4 papers planned):**
1. Core framework (Swarm & Evolutionary Computation)
2. Theoretical foundations (IEEE TEVC or Automatica)
3. Multi-modal extension (Swarm & Evolutionary Computation)
4. Adaptive mechanisms (IEEE TEVC)

**Comparison/Hybrid Papers:**
1. Comparative study: PSO-PF vs. NeuroMHE (IEEE TAC or Automatica)
2. Hybrid PSO-MHE for robust estimation (IEEE T-RO)
3. Unified framework for dynamic systems (survey paper)

---

## For Practitioners

### Decision Tree: Which Method to Use?

```
Do you have an explicit dynamics model f(x,u)?
│
├─ YES → Do you need real-time estimation (<10ms)?
│   │
│   ├─ YES → Use NeuroMHE
│   │         - Fast NLP solve with warm-starting
│   │         - Learn weights from data
│   │         - Real-world validated
│   │
│   └─ NO → Consider both
│             - NeuroMHE for accuracy
│             - PSO-PF for robustness to model mismatch
│
└─ NO → Are you tracking an optimum in a fitness landscape?
    │
    ├─ YES → Use PSO-PF
    │         - No model required
    │         - Only needs fitness evaluations
    │         - Good for black-box optimization
    │
    └─ NO → Clarify your problem
              - State estimation? → Extend NeuroMHE
              - Parameter tuning? → Use PSO-PF
              - Control? → Depends on model availability
```

### Quick Start Guides

#### Running NeuroMHE

1. Install dependencies (Python 3.9, CasADi 3.5.5, PyTorch 1.12)
2. Choose experiment section (VII-A, VII-B, or VII-C)
3. Run `main_code.py` (or `main_code_supervisedlearning.py` for VII-A)
4. Select training or evaluation mode
5. Review results and visualizations

#### Implementing PSO-PF

1. Follow complete algorithm specification:
   - Gaps document §6.1
   - Formulation document §8.2
2. Implement core components:
   - Particle initialization
   - Fitness evaluation interface
   - Weight update (PF)
   - Velocity/position update (PSO)
   - Prediction module
   - Resampling logic
3. Tune parameters using guidelines:
   - Formulation document §10
4. Validate on benchmark DOPs:
   - Moving Peaks
   - GMPB suite

---

## Summary Table: PSO-PF vs. NeuroMHE

| Aspect | PSO-PF | NeuroMHE |
|--------|--------|----------|
| **Problem Class** | Dynamic optimization (DOP) | State estimation |
| **Model Required** | ❌ No (black-box) | ✅ Yes (known dynamics) |
| **Gradient Info** | ❌ No (fitness only) | ✅ Yes (autodiff) |
| **Observations** | Fitness evaluations | Direct measurements |
| **Uncertainty** | Particle weights | Covariance matrices |
| **Multi-Modal** | ✅ Natural (species) | ❌ Single trajectory |
| **Real-Time** | Depends on N, C_f | ✅ Fast (<10ms) |
| **Learning** | Parameter meta-learning | Weight neural network |
| **Applications** | Tuning, resource allocation | Robotics, control |
| **Validation** | GMPB benchmarks | Quadrotor experiments |

**Complementary Strengths:**
- **PSO-PF:** Robust to unknown models, handles multi-modal landscapes
- **NeuroMHE:** Efficient with known models, real-time capable, physically grounded

**Potential Integration:**
- Use PSO-PF for hyperparameter tuning of NeuroMHE
- Ensemble MHE with PF-inspired weighting
- Hybrid prediction combining model-based and data-driven approaches
- Fallback to PSO-PF when NeuroMHE model becomes inaccurate

---

## Citation

If you use the PSO-PF theoretical framework in your research, please cite the relevant documents from this repository.

If you use NeuroMHE, please cite:

```bibtex
@ARTICLE{10313083,
  author={Wang, Bingheng and Ma, Zhengtian and Lai, Shupeng and Zhao, Lin},
  journal={IEEE Transactions on Robotics},
  title={Neural Moving Horizon Estimation for Robust Flight Control},
  year={2024},
  volume={40},
  pages={639-659},
  doi={10.1109/TRO.2023.3331064}
}
```

---

## Contact and Contributions

**For NeuroMHE questions:**
- Dr. Bingheng Wang: wangbingheng@u.nus.edu
- GitHub Issues: https://github.com/RCL-NUS/NeuroMHE/issues

**For PSO-PF theoretical framework:**
- Open issues in this repository for discussion
- Contributions welcome via pull requests

---

## License

Please refer to the main repository [LICENSE](../LICENSE) file for licensing information.

---

*Last Updated: 2025-01-12*
