# Option Pricing via Finite Differences

Finite difference solvers for option pricing under Black-Scholes and Heston stochastic volatility, with full mathematical derivations.

## Contents

**`heston_model.ipynb`** — Heston stochastic volatility model for European call pricing:
- Derivation of the Heston PDE in log-price coordinates
- 2D ADI (Alternating Direction Implicit) Crank-Nicolson finite difference solver
- Analytic pricing via characteristic function inversion (Heston 1993)
- Grid convergence study verifying second-order accuracy
- Calibration to live SPY option data with implied volatility surface plots

**`black_scholes.ipynb`** — Black-Scholes finite difference methods:
- Backward Euler, Crank-Nicolson, and Crank-Nicolson with Rannacher smoothing for European options
- American put pricing via Projected SOR (PSOR)
- Convergence analysis across schemes

**`option_fd.py`** — Reusable solver module containing all Black-Scholes finite difference implementations.

## Requirements

```
numpy
scipy
matplotlib
pandas
yfinance
```

## References

- Heston, S. (1993). *A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options.*
- Shreve, S. (2004). *Stochastic Calculus for Finance II: Continuous-Time Models.* Springer.
- Duffy, D. (2006). *Finite Difference Methods in Financial Engineering.* Wiley.
