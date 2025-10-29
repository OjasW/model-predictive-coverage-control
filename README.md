# Model Predictive Coverage Control

This repository provides a Python implementation of the **Model Predictive Coverage Control (MPCC)** framework — a control strategy that enables a group of agents (e.g., robots or drones) to cooperatively cover an environment while optimizing for control effort and system dynamics. The implementation follows the structure and algorithms described in the paper:

> *Carron, A., & Zeilinger, M. N.* (2020). **Model Predictive Coverage Control**. IFAC-PapersOnLine, 53(2), 6107–6112. https://doi.org/10.1016/j.ifacol.2020.12.1686


---

## Overview

The paper presents a receding‐horizon (MPC) approach for multi‐agent coverage control that explicitly accounts for nonlinear agent dynamics and polytopic state and input constraints. It uses a tracking‐MPC formulation where each agent tracks the centroid of its Voronoi region, with a terminal cost and terminal set design that ensure convergence to a centroidal Voronoi configuration under certain assumptions.

The approach uses:
- **Algorithm 1 - MPC Coverage Control:** Computes optimal control inputs for all agents by solving an MPC problem that drives each agent toward the centroid of its Voronoi region while satisfying dynamic and constraint limits.
- **Algorithm 2 - MPC with LQR Terminal Cost:** Extends Algorithm 1 by incorporating an LQR-based terminal cost and terminal set, ensuring asymptotic convergence and closed-loop stability of the coverage configuration.

This implementation includes both algorithms for experimentation and comparison.

---

## Repository Structure
```bash
mpcc/
│
├── mpcc_code.py                  # Full monolithic version of the code (legacy/unorganized)
│
├── requirements.txt              # List of required libraries
|
├── mpcc/                         # Organized, modular implementation
│   ├── algorithm1_only.py        # Implementation of Algorithm 1
│   ├── algorithm1_and_2.py       # Implementation of both Algorithm 1 and Algorithm 2
│   ├── dynamic_model.py          # Define your system dynamics here
│   ├── coverage_cost.py          # Define the coverage cost here
│   ├── phi_uniform
│   └── ... (other helper files)
│
└── README.md                     # Project documentation
```
---

## How to Use

### 1. Clone the Repository
```bash
git clone https://github.com/OjasW/model-predictive-coverage-control.git
cd model-predictive-coverage-control
```
### 2. Set Up Dependencies
```bash
pip install -r requirements.txt    # numpy, scipy, matplotlib, casadi, shapely
```
### 3. Define your System Model
Modify *__mpcc/dynamic_model.py__* to describe your system’s dynamics. You can also modify system parameters directly in the algorithm files

### 4. Run the algorithms
To run Algorithm 1 only:
```bash
python mpcc/algorithm1_only.py
```
To run both Algorithm 1 and 2
```bash
python mpcc/algorithm1_and_2.py
```
## Notes
- mpcc_code.py is a single-file legacy version of the entire codebase.
- It contains the same logic but is not modular and not recommended for modification.
- The modular version inside the mpcc/ folder is cleaner and easier to extend.
- You can modify simulation parameters (like initial positions, weights Q and R, or horizon length N) directly inside the algorithm files.

## Authors
* **Ojas Wani** - https://github.com/OjasW
