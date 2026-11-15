# TFVA: Reference Multiphysics & Architecture Benchmark Suite (`tfva-sim`)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24241909.svg)](https://doi.org/10.5281/zenodo.24241909)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Official reference simulation and verification suite for the paper:

> **Topological Field & Variational Architecture (TFVA): A Heterogeneous Spintronic-SAW & Thin-Film Photonic Co-Processing Paradigm**  
> **Manimaran T** (November 2026)  
> **Zenodo Preprint:** [https://doi.org/10.5281/zenodo.24241909](https://doi.org/10.5281/zenodo.24241909)

---

## Quick Start

```bash
pip install numpy matplotlib
python tfva_sim.py
```

Running `tfva_sim.py` produces all numerical benchmarks (`figures/benchmark_results.json`) and generates all publication figures (`figures/fig1_*.png` through `figures/fig4_*.png`).

---

## Summary of Verified Benchmarks

| Subsystem / Benchmark Metric | Unprotected / Baseline Reference | TFVA Verified Performance | Physical Mechanism / Advantage |
| :--- | :--- | :--- | :--- |
| **Track A: Trap Peak-to-Sidelobe Ratio (PSLR)** | $2.91\text{ dB}$ ($N_\omega = 1, 25\text{ MHz}$) | **$9.58\text{ dB}$** ($N_\omega = 32, 10\text{–}48\text{ MHz}$) | $71.3\%$ reduction in spurious standing-wave clutter |
| **Track B: Effective Pinning Barrier ($300\text{ K}$)** | $V_{\text{pin},0} = 22.0\,k_B T$ ($\tau_{\text{escape}} \approx 3.6\text{ s}$) | **$V_{\text{pin}}^{\text{eff}} = 1.76\,k_B T$** at $50\text{ MHz}$ trough | $-92.0\%$ dynamic barrier reduction ($<0.5\text{ W}$ global RF drive) |
| **Track B: Boltzmann Sampling KL Divergence** | $D_{\text{KL}} = 18.38$ (SAW OFF, pinned) | **$D_{\text{KL}} = 6.5 \times 10^{-4}$** (SAW ON, $80\,\mu\text{s}$) | $28,076\times$ improvement in $300\text{ K}$ Boltzmann fidelity |
| **Track B+C: Frustrated $N=64$ Ising Max-Cut** | $61.69\%$ (SAW OFF) / $84.28\%$ (DOPO) | **$96.41\%$ mean** (up to $100\%$ exact $E_0$) | Coherent optical basin search + SAW-spin thermal escape |
| **Track C: Single-Slip Reconstruction Error** | $11,870.9$ mean wrap-around error (CRT) | **$0.0\text{ error}$ ($100.0\%$ corrected) | 8-channel RRNS ($M_{\text{data}} = 30,030, p_7=17, p_8=19$) |

---

## Citation

```bibtex
@misc{manimaran2026tfva,
  author       = {T, Manimaran},
  title        = {{Topological Field & Variational Architecture (TFVA): A Heterogeneous Spintronic-SAW & Thin-Film Photonic Co-Processing Paradigm}},
  month        = nov,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.24241909},
  url          = {https://doi.org/10.5281/zenodo.24241909}
}
```
