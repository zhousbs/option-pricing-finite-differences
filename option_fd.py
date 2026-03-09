"""Finite difference solvers for European and American options under Black-Scholes."""

import numpy as np
def thomas_solver(lower, diag, upper, rhs):
    n = len(diag)

    lower = np.array(lower, dtype=float, copy=True)
    diag  = np.array(diag,  dtype=float, copy=True)
    upper = np.array(upper, dtype=float, copy=True)
    rhs   = np.array(rhs,   dtype=float, copy=True)

    #forward elimination
    for k in range(1, n):
        m = lower[k-1] / diag[k-1]
        diag[k] -= m * upper[k-1]
        rhs[k]  -= m * rhs[k-1]

    #back substitution
    x = np.zeros(n)
    x[-1] = rhs[-1] / diag[-1]
    for k in range(n-2, -1, -1):
        x[k] = (rhs[k] - upper[k] * x[k+1]) / diag[k]

    return x


def price_euro_call_be(S_0, K, sigma, r, T, S_max, M, N):
    dt = T/N
    dS = S_max/M
    S  = np.linspace(0, S_max, M+1)
    t  = np.linspace(0, T, N+1)

    i  = np.arange(1, M)
    Si = S[i]

    a = np.zeros(M+1)
    b = np.zeros(M+1)
    c = np.zeros(M+1)

    a[i] = (r*Si*dt)/(2*dS) - 0.5*sigma**2*Si**2*dt/dS**2
    b[i] = 1 + sigma**2*Si**2*dt/dS**2 + r*dt
    c[i] = -0.5*sigma**2*Si**2*dt/dS**2 - (r*Si*dt)/(2*dS)

    lower = a[2:M]
    diag  = b[1:M]
    upper = c[1:M-1]

    V = np.maximum(S - K, 0.0)

    for n_idx in range(N-1, -1, -1):
        t_n = t[n_idx]
        V_M = S_max - K*np.exp(-r*(T-t_n))

        rhs = V[1:M].copy()
        rhs[-1] -= c[M-1] * V_M

        V[0]   = 0.0
        V[1:M] = thomas_solver(lower, diag, upper, rhs)
        V[M]   = V_M

    return S, V, np.interp(S_0, S, V)


def price_euro_put_be(S_0, K, sigma, r, T, S_max, M, N):
    dt = T/N
    dS = S_max/M
    S  = np.linspace(0, S_max, M+1)
    t  = np.linspace(0, T, N+1)

    i  = np.arange(1, M)
    Si = S[i]

    a = np.zeros(M+1)
    b = np.zeros(M+1)
    c = np.zeros(M+1)

    a[i] = (r*Si*dt)/(2*dS) - 0.5*sigma**2*Si**2*dt/dS**2
    b[i] = 1 + sigma**2*Si**2*dt/dS**2 + r*dt
    c[i] = -0.5*sigma**2*Si**2*dt/dS**2 - (r*Si*dt)/(2*dS)

    lower = a[2:M]
    diag  = b[1:M]
    upper = c[1:M-1]

    V = np.maximum(K - S, 0.0)

    for n_idx in range(N-1, -1, -1):
        t_n = t[n_idx]
        V_0 = K*np.exp(-r*(T-t_n))

        rhs = V[1:M].copy()
        rhs[0] -= a[1] * V_0

        V[0]   = V_0
        V[1:M] = thomas_solver(lower, diag, upper, rhs)
        V[M]   = 0.0

    return S, V, np.interp(S_0, S, V)


def price_euro_call_cn(S_0, K, sigma, r, T, S_max, M, N):
    dt = T/N
    dS = S_max/M
    S  = np.linspace(0, S_max, M+1)
    t  = np.linspace(0, T, N+1)

    i  = np.arange(1, M)
    Si = S[i]

    A_minus = np.zeros(M+1)
    A_0     = np.zeros(M+1)
    A_plus  = np.zeros(M+1)
    B_minus = np.zeros(M+1)
    B_0     = np.zeros(M+1)
    B_plus  = np.zeros(M+1)

    A_minus[i] = -(dt/2)*(0.5*sigma**2*Si**2/dS**2 - r*Si/(2*dS))
    A_0[i]     = 1 - (dt/2)*(-sigma**2*Si**2/dS**2 - r)
    A_plus[i]  = -(dt/2)*(0.5*sigma**2*Si**2/dS**2 + r*Si/(2*dS))
    B_minus[i] =  (dt/2)*(0.5*sigma**2*Si**2/dS**2 - r*Si/(2*dS))
    B_0[i]     = 1 + (dt/2)*(-sigma**2*Si**2/dS**2 - r)
    B_plus[i]  =  (dt/2)*(0.5*sigma**2*Si**2/dS**2 + r*Si/(2*dS))

    lower = A_minus[2:M]
    diag  = A_0[1:M]
    upper = A_plus[1:M-1]

    V = np.maximum(S - K, 0.0)

    for n_idx in range(N-1, -1, -1):
        t_n   = t[n_idx]
        t_np1 = t[n_idx+1]
        V_M_n   = S_max - K*np.exp(-r*(T-t_n))
        V_M_np1 = S_max - K*np.exp(-r*(T-t_np1))

        V_next = V[1:M].copy()
        rhs = B_0[1:M]*V_next
        rhs[1:]  += B_minus[2:M]  * V_next[:-1]
        rhs[:-1] += B_plus[1:M-1] * V_next[1:]
        rhs[-1]  += B_plus[M-1]*V_M_np1 - A_plus[M-1]*V_M_n

        V[0]   = 0.0
        V[1:M] = thomas_solver(lower, diag, upper, rhs)
        V[M]   = V_M_n

    return S, V, np.interp(S_0, S, V)


def price_euro_put_cn(S_0, K, sigma, r, T, S_max, M, N):
    dt = T/N
    dS = S_max/M
    S  = np.linspace(0, S_max, M+1)
    t  = np.linspace(0, T, N+1)

    i  = np.arange(1, M)
    Si = S[i]

    A_minus = np.zeros(M+1)
    A_0     = np.zeros(M+1)
    A_plus  = np.zeros(M+1)
    B_minus = np.zeros(M+1)
    B_0     = np.zeros(M+1)
    B_plus  = np.zeros(M+1)

    A_minus[i] = -(dt/2)*(0.5*sigma**2*Si**2/dS**2 - r*Si/(2*dS))
    A_0[i]     = 1 - (dt/2)*(-sigma**2*Si**2/dS**2 - r)
    A_plus[i]  = -(dt/2)*(0.5*sigma**2*Si**2/dS**2 + r*Si/(2*dS))
    B_minus[i] =  (dt/2)*(0.5*sigma**2*Si**2/dS**2 - r*Si/(2*dS))
    B_0[i]     = 1 + (dt/2)*(-sigma**2*Si**2/dS**2 - r)
    B_plus[i]  =  (dt/2)*(0.5*sigma**2*Si**2/dS**2 + r*Si/(2*dS))

    lower = A_minus[2:M]
    diag  = A_0[1:M]
    upper = A_plus[1:M-1]

    V = np.maximum(K - S, 0.0)

    for n_idx in range(N-1, -1, -1):
        t_n   = t[n_idx]
        t_np1 = t[n_idx+1]
        V_0_n   = K*np.exp(-r*(T-t_n))
        V_0_np1 = K*np.exp(-r*(T-t_np1))

        V_next = V[1:M].copy()
        rhs = B_0[1:M]*V_next
        rhs[1:]  += B_minus[2:M]  * V_next[:-1]
        rhs[:-1] += B_plus[1:M-1] * V_next[1:]
        rhs[0]   += B_minus[1]*V_0_np1 - A_minus[1]*V_0_n

        V[0]   = V_0_n
        V[1:M] = thomas_solver(lower, diag, upper, rhs)
        V[M]   = 0.0

    return S, V, np.interp(S_0, S, V)


def _build_theta_mats(sigma, r, dt_step, theta, dS, M, i, Si):
    #theta=1 -> backward euler, theta=0.5 -> crank-nicolson
    A_minus = np.zeros(M+1)
    A_0     = np.zeros(M+1)
    A_plus  = np.zeros(M+1)
    B_minus = np.zeros(M+1)
    B_0     = np.zeros(M+1)
    B_plus  = np.zeros(M+1)

    A_minus[i] = -theta*dt_step*(0.5*sigma**2*Si**2/dS**2 - r*Si/(2*dS))
    A_0[i]     = 1 - theta*dt_step*(-sigma**2*Si**2/dS**2 - r)
    A_plus[i]  = -theta*dt_step*(0.5*sigma**2*Si**2/dS**2 + r*Si/(2*dS))
    B_minus[i] = (1-theta)*dt_step*(0.5*sigma**2*Si**2/dS**2 - r*Si/(2*dS))
    B_0[i]     = 1 + (1-theta)*dt_step*(-sigma**2*Si**2/dS**2 - r)
    B_plus[i]  = (1-theta)*dt_step*(0.5*sigma**2*Si**2/dS**2 + r*Si/(2*dS))

    lower = A_minus[2:M]
    diag  = A_0[1:M]
    upper = A_plus[1:M-1]
    return A_minus, A_0, A_plus, B_minus, B_0, B_plus, lower, diag, upper


def price_euro_call_cn_rannacher(S_0, K, sigma, r, T, S_max, M, N):
    dt = T/N
    dS = S_max/M
    S  = np.linspace(0, S_max, M+1)
    t  = np.linspace(0, T, N+1)
    i  = np.arange(1, M)
    Si = S[i]

    Am_CN, A0_CN, Ap_CN, Bm_CN, B0_CN, Bp_CN, lCN, dCN, uCN = _build_theta_mats(sigma, r, dt,   0.5, dS, M, i, Si)
    Am_BE, A0_BE, Ap_BE, Bm_BE, B0_BE, Bp_BE, lBE, dBE, uBE = _build_theta_mats(sigma, r, dt/2, 1.0, dS, M, i, Si)

    V = np.maximum(S - K, 0.0)

    #two BE half-steps for Rannacher smoothing
    t_np1 = T
    for _ in range(2):
        t_n     = t_np1 - dt/2
        V_M_n   = S_max - K*np.exp(-r*(T-t_n))
        V_M_np1 = S_max - K*np.exp(-r*(T-t_np1))

        V_next = V[1:M].copy()
        rhs = B0_BE[1:M]*V_next
        rhs[1:]  += Bm_BE[2:M]   * V_next[:-1]
        rhs[:-1] += Bp_BE[1:M-1] * V_next[1:]
        rhs[-1]  += Bp_BE[M-1]*V_M_np1 - Ap_BE[M-1]*V_M_n

        V[0]   = 0.0
        V[1:M] = thomas_solver(lBE, dBE, uBE, rhs)
        V[M]   = V_M_n
        t_np1  = t_n

    #remaining steps with CN
    for n_idx in range(N-2, -1, -1):
        t_n   = t[n_idx]
        t_np1 = t[n_idx+1]
        V_M_n   = S_max - K*np.exp(-r*(T-t_n))
        V_M_np1 = S_max - K*np.exp(-r*(T-t_np1))

        V_next = V[1:M].copy()
        rhs = B0_CN[1:M]*V_next
        rhs[1:]  += Bm_CN[2:M]   * V_next[:-1]
        rhs[:-1] += Bp_CN[1:M-1] * V_next[1:]
        rhs[-1]  += Bp_CN[M-1]*V_M_np1 - Ap_CN[M-1]*V_M_n

        V[0]   = 0.0
        V[1:M] = thomas_solver(lCN, dCN, uCN, rhs)
        V[M]   = V_M_n

    return S, V, np.interp(S_0, S, V)


def price_euro_put_cn_rannacher(S_0, K, sigma, r, T, S_max, M, N):
    dt = T/N
    dS = S_max/M
    S  = np.linspace(0, S_max, M+1)
    t  = np.linspace(0, T, N+1)
    i  = np.arange(1, M)
    Si = S[i]

    Am_CN, A0_CN, Ap_CN, Bm_CN, B0_CN, Bp_CN, lCN, dCN, uCN = _build_theta_mats(sigma, r, dt,   0.5, dS, M, i, Si)
    Am_BE, A0_BE, Ap_BE, Bm_BE, B0_BE, Bp_BE, lBE, dBE, uBE = _build_theta_mats(sigma, r, dt/2, 1.0, dS, M, i, Si)

    V = np.maximum(K - S, 0.0)

    #two BE half-steps for Rannacher smoothing
    t_np1 = T
    for _ in range(2):
        t_n     = t_np1 - dt/2
        V_0_n   = K*np.exp(-r*(T-t_n))
        V_0_np1 = K*np.exp(-r*(T-t_np1))

        V_next = V[1:M].copy()
        rhs = B0_BE[1:M]*V_next
        rhs[1:]  += Bm_BE[2:M]   * V_next[:-1]
        rhs[:-1] += Bp_BE[1:M-1] * V_next[1:]
        rhs[0]   += Bm_BE[1]*V_0_np1 - Am_BE[1]*V_0_n

        V[0]   = V_0_n
        V[1:M] = thomas_solver(lBE, dBE, uBE, rhs)
        V[M]   = 0.0
        t_np1  = t_n

    #remaining steps with CN
    for n_idx in range(N-2, -1, -1):
        t_n   = t[n_idx]
        t_np1 = t[n_idx+1]
        V_0_n   = K*np.exp(-r*(T-t_n))
        V_0_np1 = K*np.exp(-r*(T-t_np1))

        V_next = V[1:M].copy()
        rhs = B0_CN[1:M]*V_next
        rhs[1:]  += Bm_CN[2:M]   * V_next[:-1]
        rhs[:-1] += Bp_CN[1:M-1] * V_next[1:]
        rhs[0]   += Bm_CN[1]*V_0_np1 - Am_CN[1]*V_0_n

        V[0]   = V_0_n
        V[1:M] = thomas_solver(lCN, dCN, uCN, rhs)
        V[M]   = 0.0

    return S, V, np.interp(S_0, S, V)


def price_american_put_psor_be(S_0, K, sigma, r, T, S_max, M, N,
                                omega=1.2, tol=1e-10, max_iter=20000):
    dt = T/N
    dS = S_max/M
    S  = np.linspace(0, S_max, M+1)
    t  = np.linspace(0, T, N+1)

    i  = np.arange(1, M)
    Si = S[i]

    a = np.zeros(M+1)
    b = np.zeros(M+1)
    c = np.zeros(M+1)

    a[i] = (r*Si*dt)/(2*dS) - 0.5*sigma**2*Si**2*dt/dS**2
    b[i] = 1 + sigma**2*Si**2*dt/dS**2 + r*dt
    c[i] = -0.5*sigma**2*Si**2*dt/dS**2 - (r*Si*dt)/(2*dS)

    payoff = np.maximum(K - S, 0.0)
    V = payoff.copy()

    for n_idx in range(N-1, -1, -1):
        rhs = V[1:M].copy()
        rhs[0] -= a[1] * K

        x = V[1:M].copy()
        for _ in range(max_iter):
            x_old = x.copy()
            for j in range(M-1):
                i_idx = j + 1
                left  = x[j-1] if j > 0 else K
                right = x[j+1] if j < M-2 else 0.0
                gs    = (rhs[j] - a[i_idx]*left - c[i_idx]*right) / b[i_idx]
                x[j]  = max((1-omega)*x[j] + omega*gs, payoff[i_idx])
            if np.max(np.abs(x - x_old)) < tol:
                break

        V[0]   = K
        V[1:M] = x
        V[M]   = 0.0

    return S, V, np.interp(S_0, S, V)


def price_american_put_psor_cn(S_0, K, sigma, r, T, S_max, M, N,
                                omega=1.2, tol=1e-10, max_iter=20000):
    dt = T/N
    dS = S_max/M
    S  = np.linspace(0, S_max, M+1)
    t  = np.linspace(0, T, N+1)

    i  = np.arange(1, M)
    Si = S[i]

    A_minus = np.zeros(M+1)
    A_0     = np.zeros(M+1)
    A_plus  = np.zeros(M+1)
    B_minus = np.zeros(M+1)
    B_0     = np.zeros(M+1)
    B_plus  = np.zeros(M+1)

    A_minus[i] = -(dt/2)*(0.5*sigma**2*Si**2/dS**2 - r*Si/(2*dS))
    A_0[i]     = 1 - (dt/2)*(-sigma**2*Si**2/dS**2 - r)
    A_plus[i]  = -(dt/2)*(0.5*sigma**2*Si**2/dS**2 + r*Si/(2*dS))
    B_minus[i] =  (dt/2)*(0.5*sigma**2*Si**2/dS**2 - r*Si/(2*dS))
    B_0[i]     = 1 + (dt/2)*(-sigma**2*Si**2/dS**2 - r)
    B_plus[i]  =  (dt/2)*(0.5*sigma**2*Si**2/dS**2 + r*Si/(2*dS))

    payoff = np.maximum(K - S, 0.0)
    V = payoff.copy()

    for n_idx in range(N-1, -1, -1):
        V_next = V[1:M].copy()
        rhs = B_0[1:M]*V_next
        rhs[1:]  += B_minus[2:M]  * V_next[:-1]
        rhs[:-1] += B_plus[1:M-1] * V_next[1:]
        rhs[0]   += (B_minus[1] - A_minus[1]) * K  # V(0)=K at both time levels

        x = V[1:M].copy()
        for _ in range(max_iter):
            x_old = x.copy()
            for j in range(M-1):
                i_idx = j + 1
                left  = x[j-1] if j > 0 else K
                right = x[j+1] if j < M-2 else 0.0
                gs    = (rhs[j] - A_minus[i_idx]*left - A_plus[i_idx]*right) / A_0[i_idx]
                x[j]  = max((1-omega)*x[j] + omega*gs, payoff[i_idx])
            if np.max(np.abs(x - x_old)) < tol:
                break

        V[0]   = K
        V[1:M] = x
        V[M]   = 0.0

    return S, V, np.interp(S_0, S, V)
