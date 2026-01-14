# pdes.py
#
# COMPLETE PDE DATASET + SOLVER MODULE
# Fully implements all stressors promised in:
# "Forcing and Diagnosing Failure Modes of Neural Operators Across Diverse PDE Classes"
#
# Covers:
#   - Parameter shifts
#   - Boundary / terminal condition shifts
#   - Resolution extrapolation
#   - Long-horizon rollout
#   - Perturbation sensitivity
#
# Dependencies: numpy only

from __future__ import annotations
import numpy as np

# ============================================================
# Utilities
# ============================================================

PDE_NAMES = (
    "poisson",
    "schrodinger",
    "kuramoto_sivashinsky",
    "navier_stokes",
    "black_scholes",
)

def _rng(rng=None):
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def override_params(base: dict, overrides: dict) -> dict:
    out = dict(base)
    for k, v in overrides.items():
        if v is not None:
            out[k] = v
    return out


# ============================================================
# Grids
# ============================================================

def grid_1d(n, domain=(0.0, 1.0), periodic=False):
    a, b = domain
    if periodic:
        x = np.linspace(a, b, n, endpoint=False)
        h = (b - a) / n
    else:
        x = np.linspace(a, b, n)
        h = (b - a) / (n - 1)
    return x, h


def grid_2d(n, domain=(0.0, 2.0 * np.pi), periodic=True):
    a, b = domain
    if periodic:
        x = np.linspace(a, b, n, endpoint=False)
    else:
        x = np.linspace(a, b, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    h = (b - a) / (n if periodic else (n - 1))
    return X, Y, h


def fftfreq_1d(n, L):
    return 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)


def fftfreq_2d(n, L):
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
    return np.meshgrid(k, k, indexing="ij")


# ============================================================
# 1) Poisson (1D, elliptic)
# ============================================================

def poisson_params(rng):
    return dict(
        a_scale=rng.uniform(0.3, 2.5),
        f_scale=rng.uniform(0.3, 2.5),
        a_freq=int(rng.integers(1, 6)),
        f_freq=int(rng.integers(1, 6)),
        rough=bool(rng.integers(0, 2)),
        bc0=rng.uniform(-1.0, 1.0),
        bc1=rng.uniform(-1.0, 1.0),
    )


def solve_poisson_1d(a, f, h, bc0, bc1, max_iter=20000, tol=1e-8):
    n = len(f)
    u = np.zeros(n)
    u[0], u[-1] = bc0, bc1
    a_mid = 0.5 * (a[:-1] + a[1:])

    for _ in range(max_iter):
        u_new = u.copy()
        denom = a_mid[1:] + a_mid[:-1]
        u_new[1:-1] = (
            a_mid[1:] * u[2:] +
            a_mid[:-1] * u[:-2] +
            h * h * f[1:-1]
        ) / denom
        u_new[0], u_new[-1] = bc0, bc1
        if np.linalg.norm(u_new - u) < tol:
            break
        u = u_new
    return u


def sample_poisson(
    rng, n=128, domain=(0.0, 1.0),
    a_scale=None, f_scale=None,
    a_freq=None, f_freq=None,
    rough=None, bc0=None, bc1=None
):
    rng = _rng(rng)
    params = override_params(
        poisson_params(rng),
        dict(
            a_scale=a_scale, f_scale=f_scale,
            a_freq=a_freq, f_freq=f_freq,
            rough=rough, bc0=bc0, bc1=bc1,
        )
    )

    x, h = grid_1d(n, domain, periodic=False)

    if not params["rough"]:
        a = 1.0 + params["a_scale"] * (0.5 + 0.5 * np.sin(2*np.pi*params["a_freq"]*x))
        f = params["f_scale"] * np.sin(2*np.pi*params["f_freq"]*x)
    else:
        a = 0.5 + params["a_scale"] * (np.sign(np.sin(2*np.pi*params["a_freq"]*x)) + 1.5)
        f = params["f_scale"] * np.sign(np.sin(2*np.pi*params["f_freq"]*x))

    a = np.clip(a, 0.05, None)
    u = solve_poisson_1d(a, f, h, params["bc0"], params["bc1"])

    bc_mask = np.zeros_like(x)
    bc_mask[0], bc_mask[-1] = params["bc0"], params["bc1"]

    inp = np.stack([a, f, bc_mask], axis=-1).astype(np.float32)
    out = u.astype(np.float32)[:, None]

    meta = dict(
        pde="poisson",
        type="elliptic",
        grid=dict(x=x, h=h, domain=domain),
        params=params,
    )
    return inp, out, meta


# ============================================================
# 2) Nonlinear Schrödinger (1D, dispersive)
# ============================================================

def schrodinger_params(rng):
    return dict(
        kappa=rng.uniform(0.5, 2.0),
        L=rng.uniform(10.0, 20.0),
        T=rng.uniform(0.5, 2.0),
        amp=rng.uniform(0.5, 2.0),
        width=rng.uniform(0.5, 2.5),
        k0=rng.uniform(-3.0, 3.0),
        x0=rng.uniform(-2.0, 2.0),
    )


def solve_schrodinger(u0, kappa, L, T, nt):
    n = len(u0)
    dt = T / nt
    k = fftfreq_1d(n, L)
    lin = np.exp(1j * (-(k**2)) * dt / 2)

    u = u0.copy()
    traj = np.zeros((nt + 1, n), dtype=np.complex128)
    traj[0] = u

    for t in range(1, nt + 1):
        u = np.fft.ifft(np.fft.fft(u) * lin)
        u *= np.exp(1j * kappa * np.abs(u)**2 * dt)
        u = np.fft.ifft(np.fft.fft(u) * lin)
        traj[t] = u
    return traj


def sample_schrodinger(
    rng, n=256, nt=50,
    kappa=None, L=None, T=None,
    amp=None, width=None, k0=None, x0=None
):
    rng = _rng(rng)
    params = override_params(
        schrodinger_params(rng),
        dict(kappa=kappa, L=L, T=T, amp=amp, width=width, k0=k0, x0=x0)
    )

    x, _ = grid_1d(n, (-params["L"]/2, params["L"]/2), periodic=True)
    u0 = params["amp"] * np.exp(-0.5*((x - params["x0"])/params["width"])**2) * np.exp(1j*params["k0"]*x)
    traj = solve_schrodinger(u0, params["kappa"], params["L"], params["T"], nt)

    inp = np.stack([u0.real, u0.imag, np.full_like(x, params["kappa"])], axis=-1).astype(np.float32)
    out = np.stack([traj.real, traj.imag], axis=-1).astype(np.float32)

    meta = dict(
        pde="schrodinger",
        type="dispersive",
        grid=dict(x=x, L=params["L"], periodic=True),
        params=params,
        time=dict(T=params["T"], nt=nt, dt=params["T"]/nt),
    )
    return inp, out, meta


# ============================================================
# 3) Kuramoto–Sivashinsky (1D, chaotic)
# ============================================================

def ks_params(rng):
    return dict(
        L=rng.uniform(22.0, 32.0),
        T=rng.uniform(20.0, 60.0),
        ic_amp=rng.uniform(0.5, 2.0),
        ic_freq_max=int(rng.integers(2, 9)),
    )


def solve_ks(u0, L, T, nt):
    n = len(u0)
    dt = T / nt
    k = fftfreq_1d(n, L)
    Lk = k**2 - k**4
    u_hat = np.fft.fft(u0)
    traj = np.zeros((nt+1, n))
    traj[0] = u0

    for t in range(1, nt+1):
        nonlinear = -0.5j * k * np.fft.fft(np.fft.ifft(u_hat).real**2)
        u_hat = (u_hat + dt * nonlinear) * np.exp(dt * Lk)
        traj[t] = np.fft.ifft(u_hat).real
    return traj


def sample_kuramoto_sivashinsky(
    rng, n=256, nt=200,
    L=None, T=None, ic_amp=None, ic_freq_max=None
):
    rng = _rng(rng)
    params = override_params(
        ks_params(rng),
        dict(L=L, T=T, ic_amp=ic_amp, ic_freq_max=ic_freq_max)
    )

    x, _ = grid_1d(n, (0.0, params["L"]), periodic=True)
    u0 = np.zeros(n)
    for m in range(1, params["ic_freq_max"] + 1):
        u0 += params["ic_amp"] * np.sin(2*np.pi*m*x/params["L"]) / m**1.5

    traj = solve_ks(u0, params["L"], params["T"], nt)

    inp = np.stack([u0, np.full_like(x, params["L"])], axis=-1).astype(np.float32)
    out = traj[:, :, None].astype(np.float32)

    meta = dict(
        pde="kuramoto_sivashinsky",
        type="chaotic",
        grid=dict(x=x, L=params["L"], periodic=True),
        params=params,
        time=dict(T=params["T"], nt=nt, dt=params["T"]/nt),
    )
    return inp, out, meta


# ============================================================
# 4) Navier–Stokes (2D incompressible, vorticity form; advective / multi-scale)
#     ω_t + u·∇ω = νΔω + s
#     u = ∇^⊥ ψ,  Δψ = ω
#
# Override-safe (proposal-complete):
#   - Parameter shifts: nu, forcing_amp, forcing_k, ic_amp, ic_kmax, T, L
#   - Resolution extrapolation: arbitrary n
#   - Long-horizon rollout: full trajectory length nt+1
#   - Structural perturbations: forcing toggles and IC spectrum control
# ============================================================

def navier_stokes_params(rng):
    return dict(
        L=float(2.0 * np.pi),
        nu=float(rng.uniform(1e-3, 2e-2)),
        T=float(rng.uniform(1.0, 4.0)),
        forcing_amp=float(rng.uniform(0.0, 1.0)),
        forcing_k=int(rng.integers(1, 5)),
        ic_amp=float(rng.uniform(0.5, 2.0)),
        ic_kmax=int(rng.integers(2, 9)),
    )


def ns_forcing_vorticity(X, Y, params):
    # Steady vorticity forcing: s(x,y) = A sin(kx) sin(ky)
    A = float(params["forcing_amp"])
    k = int(params["forcing_k"])
    if A == 0.0:
        return np.zeros_like(X, dtype=np.float64)
    return (A * np.sin(k * X) * np.sin(k * Y)).astype(np.float64)


def ns_ic_vorticity(X, Y, params, rng):
    """
    Random low-frequency vorticity field with controlled bandwidth ic_kmax.
    Enforces real-field Hermitian symmetry in Fourier space.
    """
    n = X.shape[0]
    L = float(params["L"])
    kmax = int(params["ic_kmax"])
    amp0 = float(params["ic_amp"])

    # mode indices in "integer-ish" form
    k_int = np.fft.fftfreq(n) * n
    kkx, kky = np.meshgrid(k_int, k_int, indexing="ij")

    w_hat = np.zeros((n, n), dtype=np.complex128)

    # populate modes in a square band [-kmax,kmax]^2 excluding (0,0)
    for i in range(-kmax, kmax + 1):
        for j in range(-kmax, kmax + 1):
            if i == 0 and j == 0:
                continue
            # decay with radius (keeps IC smooth-ish)
            denom = 1.0 + i * i + j * j
            amp = amp0 / denom
            phase = rng.uniform(0.0, 2.0 * np.pi)
            ii = i % n
            jj = j % n
            w_hat[ii, jj] = amp * np.exp(1j * phase)
            # enforce Hermitian symmetry for real ifft2
            w_hat[(-i) % n, (-j) % n] = np.conj(w_hat[ii, jj])

    w = np.fft.ifft2(w_hat).real
    return w.astype(np.float64)


def _dealias_mask_2_3_rule(n: int):
    """
    Classic 2/3 dealiasing mask for pseudo-spectral nonlinear products.
    Keeps modes with |k| <= n/3 along each axis.
    """
    cutoff = n // 3
    k = np.fft.fftfreq(n) * n  # integer-ish
    kkx, kky = np.meshgrid(k, k, indexing="ij")
    mask = (np.abs(kkx) <= cutoff) & (np.abs(kky) <= cutoff)
    return mask.astype(np.float64)


def solve_navier_stokes_vorticity(w0, s, nu, L, T, nt):
    """
    2D incompressible NS in vorticity form on periodic [0,L)^2.

    Time stepping: RK2 on ω_hat with pseudo-spectral evaluation of u·∇ω.
    Dealiasing: 2/3 rule on nonlinear term.

    Returns:
        traj: (nt+1, n, n) real vorticity field over time
    """
    n = w0.shape[0]
    dt = T / nt

    # Fourier frequencies (angular)
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0  # avoid divide-by-zero; will pin psi_hat[0,0]=0 anyway

    dealias = _dealias_mask_2_3_rule(n)

    w_hat = np.fft.fft2(w0).astype(np.complex128)
    s_hat = np.fft.fft2(s).astype(np.complex128)

    def velocity_from_vorticity_hat(wh):
        # Δψ = ω  ->  (-k^2) ψ_hat = ω_hat  -> ψ_hat = - ω_hat / k^2
        psi_hat = -wh / k2
        psi_hat[0, 0] = 0.0 + 0.0j
        # u = (∂y ψ, -∂x ψ)
        u_hat_x = 1j * ky * psi_hat
        u_hat_y = -1j * kx * psi_hat
        u_x = np.fft.ifft2(u_hat_x).real
        u_y = np.fft.ifft2(u_hat_y).real
        return u_x, u_y

    def advective_term_hat(wh):
        # compute u·∇ω in physical space, return its FFT (dealiased)
        w = np.fft.ifft2(wh).real
        u_x, u_y = velocity_from_vorticity_hat(wh)
        w_x = np.fft.ifft2(1j * kx * wh).real
        w_y = np.fft.ifft2(1j * ky * wh).real
        adv = u_x * w_x + u_y * w_y
        adv_hat = np.fft.fft2(adv)
        adv_hat *= dealias
        return adv_hat

    def rhs(wh):
        adv_hat = advective_term_hat(wh)
        # ω_t = -u·∇ω + νΔω + s ;  Δω -> -k^2 ω_hat
        return (-adv_hat) + (-nu * k2 * wh) + s_hat

    traj = np.zeros((nt + 1, n, n), dtype=np.float64)
    traj[0] = w0.copy()

    for t in range(1, nt + 1):
        k1 = rhs(w_hat)
        w_hat_mid = w_hat + dt * k1
        k2_rhs = rhs(w_hat_mid)
        w_hat = w_hat + 0.5 * dt * (k1 + k2_rhs)
        traj[t] = np.fft.ifft2(w_hat).real

    return traj


def sample_navier_stokes(
    rng,
    n=64,
    nt=40,
    # override-able parameters (proposal stress tests)
    L=None,
    nu=None,
    T=None,
    forcing_amp=None,
    forcing_k=None,
    ic_amp=None,
    ic_kmax=None,
):
    rng = _rng(rng)

    # sample then override
    params = override_params(
        navier_stokes_params(rng),
        dict(
            L=L,
            nu=nu,
            T=T,
            forcing_amp=forcing_amp,
            forcing_k=forcing_k,
            ic_amp=ic_amp,
            ic_kmax=ic_kmax,
        ),
    )

    # grid on [0,L)^2 (periodic)
    Lval = float(params["L"])
    X, Y, h = grid_2d(n, domain=(0.0, Lval), periodic=True)

    # initial vorticity + forcing
    w0 = ns_ic_vorticity(X, Y, params, rng)
    s = ns_forcing_vorticity(X, Y, params)

    traj = solve_navier_stokes_vorticity(
        w0=w0,
        s=s,
        nu=float(params["nu"]),
        L=Lval,
        T=float(params["T"]),
        nt=int(nt),
    )

    # Input channels: w0, forcing s, parameter channel nu
    nu_chan = np.full((n, n), float(params["nu"]), dtype=np.float64)
    inp = np.stack([w0, s, nu_chan], axis=-1).astype(np.float32)  # (n, n, 3)

    # Output trajectory: (nt+1, n, n, 1)
    out = traj.astype(np.float32)[:, :, :, None]

    meta = dict(
        pde="navier_stokes",
        type="advective",
        grid=dict(L=Lval, n=n, h=h, periodic=True, domain=(0.0, Lval)),
        params=params,
        time=dict(T=float(params["T"]), nt=int(nt), dt=float(params["T"]) / int(nt)),
    )

    return inp, out, meta


# ============================================================
# 5) Black–Scholes (1D, parabolic)
#     V_t + 0.5 σ^2 S^2 V_SS + r S V_S - r V = 0
#     terminal: V(S,T) = payoff(S)
#
# Override-safe (proposal-complete):
#   - Parameter shifts: sigma, r, T, Smax, K
#   - Terminal condition shifts: payoff_type (incl. discontinuous digital)
#   - Boundary condition shifts: bc_type ∈ {standard, discounted_payoff, dirichlet_zero, linear_extrapolation}
#   - Resolution extrapolation: arbitrary n, nt
#   - Outputs: by default returns V(S,0) (static PDE), optional full surface for diagnostics
# ============================================================

def black_scholes_params(rng: np.random.Generator):
    payoff_type = rng.choice(["call", "put", "digital_call", "smooth_call"])
    bc_type = rng.choice(["standard", "discounted_payoff"])  # defaults; can be overridden
    return dict(
        sigma=float(rng.uniform(0.05, 0.8)),
        r=float(rng.uniform(0.0, 0.1)),
        T=float(rng.uniform(0.25, 2.0)),
        Smax=float(rng.uniform(200.0, 400.0)),
        K=float(rng.uniform(50.0, 200.0)),
        payoff_type=str(payoff_type),
        bc_type=str(bc_type),
    )


def bs_payoff(S: np.ndarray, params: dict) -> np.ndarray:
    K = float(params["K"])
    pt = str(params["payoff_type"])

    if pt == "call":
        return np.maximum(S - K, 0.0)
    if pt == "put":
        return np.maximum(K - S, 0.0)
    if pt == "digital_call":
        return (S > K).astype(np.float64)
    if pt == "smooth_call":
        # smooth approx to max(S-K,0): softplus-ish
        beta = 0.1 * max(K, 1.0)
        return beta * np.log1p(np.exp((S - K) / beta))

    raise ValueError(f"Unknown payoff_type: {pt}")


def solve_tridiag(a, b, c, d):
    """
    Thomas algorithm for tridiagonal systems.
      a: subdiag (n-1)
      b: diag (n)
      c: superdiag (n-1)
      d: rhs (n)
    """
    n = len(b)
    cp = np.zeros(n - 1, dtype=np.float64)
    dp = np.zeros(n, dtype=np.float64)
    bp = b.astype(np.float64).copy()

    cp[0] = c[0] / bp[0]
    dp[0] = d[0] / bp[0]

    for i in range(1, n - 1):
        denom = bp[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom

    dp[n - 1] = (d[n - 1] - a[n - 2] * dp[n - 2]) / (bp[n - 1] - a[n - 2] * cp[n - 2])

    x = np.zeros(n, dtype=np.float64)
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def _bs_boundary_values(
    S: np.ndarray,
    payoff_T: np.ndarray,
    params: dict,
    tau: float,          # remaining time-to-maturity at current step
    tau_next: float,     # remaining time at next step (i.e., tau - dt)
) -> tuple[float, float]:
    """
    Produce boundary values V(0, t) and V(Smax, t) at the "next" time level.

    bc_type modes:
      - "standard": finance-asymptotic call/put boundaries when payoff_type is call/put; else fallback discounted_payoff
      - "discounted_payoff": V(boundary,t) = payoff(boundary)*exp(-r*(T-t)) approximation
      - "dirichlet_zero": V(0,t)=V(Smax,t)=0 (stress test / intentionally wrong BC)
      - "linear_extrapolation": match slope at boundary using payoff slope proxy (stress test)
    """
    bc_type = str(params.get("bc_type", "discounted_payoff"))
    pt = str(params["payoff_type"])
    r = float(params["r"])
    K = float(params["K"])
    Smax = float(S[-1])

    # convenient discount factor from terminal to next time
    disc = np.exp(-r * tau_next)

    if bc_type == "dirichlet_zero":
        return 0.0, 0.0

    if bc_type == "discounted_payoff":
        return float(payoff_T[0] * disc), float(payoff_T[-1] * disc)

    if bc_type == "standard":
        # Standard BS boundaries for vanilla call/put; fallback to discounted payoff for other payoffs.
        if pt == "call":
            # V(0,t)=0 ; V(Smax,t) ~ Smax - K e^{-r*(T-t)}
            V0 = 0.0
            VN = Smax - K * np.exp(-r * tau_next)
            return float(V0), float(VN)
        if pt == "put":
            # V(0,t) ~ K e^{-r*(T-t)} ; V(Smax,t)=0
            V0 = K * np.exp(-r * tau_next)
            VN = 0.0
            return float(V0), float(VN)
        # digital/smooth: no clean asymptotics -> use discounted payoff
        return float(payoff_T[0] * disc), float(payoff_T[-1] * disc)

    if bc_type == "linear_extrapolation":
        # Stress-test BC: approximate boundary values by linear extrapolation of payoff at boundaries,
        # then discount. (This is intentionally "structurally different" and can be wrong.)
        # Use first two / last two payoff points to extrapolate.
        if len(S) < 3:
            return float(payoff_T[0] * disc), float(payoff_T[-1] * disc)

        # left extrapolate to S=0 (already is), but adjust by local slope
        slope_left = (payoff_T[1] - payoff_T[0]) / (S[1] - S[0])
        V0 = payoff_T[0] - slope_left * (S[0] - 0.0)

        # right extrapolate to S=Smax (already), adjust by local slope
        slope_right = (payoff_T[-1] - payoff_T[-2]) / (S[-1] - S[-2])
        VN = payoff_T[-1] + slope_right * (Smax - S[-1])

        return float(V0 * disc), float(VN * disc)

    raise ValueError(f"Unknown bc_type: {bc_type}")


def solve_black_scholes_cn(
    S: np.ndarray,
    payoff_T: np.ndarray,
    sigma: float,
    r: float,
    T: float,
    nt: int,
    params: dict,
    return_surface: bool = False,
):
    """
    Crank–Nicolson backward in time for BS on S in [0, Smax].

    Returns:
      - if return_surface=False: V0 (n,)
      - if return_surface=True: surface (nt+1, n) with surface[0]=V(S,T), surface[-1]=V(S,0)
        (time stored in reverse, terminal-to-initial, for convenience)
    """
    n = len(S)
    dS = float(S[1] - S[0])
    dt = float(T / nt)

    # start at terminal condition
    V = payoff_T.astype(np.float64).copy()

    surface = None
    if return_surface:
        surface = np.zeros((nt + 1, n), dtype=np.float64)
        surface[0] = V.copy()  # at tau=T (terminal)

    # Interior indices i=1..n-2
    Si = S[1:-1]
    A = 0.5 * sigma * sigma * (Si ** 2)
    B = r * Si

    # FD coefficients for L operator:
    alpha = (A / dS**2) - (B / (2.0 * dS))
    beta  = (-2.0 * A / dS**2) - r
    gamma = (A / dS**2) + (B / (2.0 * dS))

    # CN matrices:
    aL = -0.5 * dt * alpha
    bL = 1.0 - 0.5 * dt * beta
    cL = -0.5 * dt * gamma

    aR = 0.5 * dt * alpha
    bR = 1.0 + 0.5 * dt * beta
    cR = 0.5 * dt * gamma

    # Step backward in time: tau = remaining time-to-maturity
    # At step s (0-based), current V corresponds to tau = T - s*dt
    for s in range(nt):
        tau = T - s * dt
        tau_next = tau - dt

        # boundary values at next time level
        V0, VN = _bs_boundary_values(S, payoff_T, params, tau=tau, tau_next=tau_next)

        # RHS for interior
        rhs = aR * V[:-2] + bR * V[1:-1] + cR * V[2:]
        rhs = rhs.astype(np.float64)

        # add boundary contributions (CN)
        rhs[0]  += (0.5 * dt * alpha[0]) * V0
        rhs[-1] += (0.5 * dt * gamma[-1]) * VN

        # solve tridiagonal for interior
        V_interior = solve_tridiag(aL, bL, cL, rhs)

        V_new = V.copy()
        V_new[0] = V0
        V_new[-1] = VN
        V_new[1:-1] = V_interior
        V = V_new

        if return_surface:
            surface[s + 1] = V.copy()

    if return_surface:
        return surface
    return V


def sample_black_scholes(
    rng,
    n=256,
    nt=80,
    # override-able parameters (proposal stress tests)
    sigma=None,
    r=None,
    T=None,
    Smax=None,
    K=None,
    payoff_type=None,
    bc_type=None,
    # optional diagnostic output
    return_surface: bool = False,
):
    rng = _rng(rng)

    params = override_params(
        black_scholes_params(rng),
        dict(
            sigma=sigma,
            r=r,
            T=T,
            Smax=Smax,
            K=K,
            payoff_type=payoff_type,
            bc_type=bc_type,
        ),
    )

    Smax_val = float(params["Smax"])
    S = np.linspace(0.0, Smax_val, int(n), dtype=np.float64)

    payoff = bs_payoff(S, params).astype(np.float64)

    sol = solve_black_scholes_cn(
        S=S,
        payoff_T=payoff,
        sigma=float(params["sigma"]),
        r=float(params["r"]),
        T=float(params["T"]),
        nt=int(nt),
        params=params,
        return_surface=bool(return_surface),
    )

    # Input channels: payoff(S), sigma channel, r channel
    sigma_chan = np.full_like(S, float(params["sigma"]), dtype=np.float64)
    r_chan = np.full_like(S, float(params["r"]), dtype=np.float64)
    inp = np.stack([payoff, sigma_chan, r_chan], axis=-1).astype(np.float32)  # (n, 3)

    # Output:
    # - default: V(S,0) as (n,1) for "static PDE" interface
    # - optionally: full surface as (nt+1, n, 1) (still "static" to the training loop unless you use it)
    if not return_surface:
        V0 = sol
        out = V0.astype(np.float32)[:, None]  # (n, 1)
    else:
        surface = sol  # (nt+1, n), where surface[0]=terminal, surface[-1]=initial
        out = surface.astype(np.float32)[:, :, None]  # (nt+1, n, 1)

    meta = dict(
        pde="black_scholes",
        type="parabolic",
        grid=dict(S=S, Smax=Smax_val, n=int(n), dS=float(S[1] - S[0])),
        params=params,
        time=dict(T=float(params["T"]), nt=int(nt), dt=float(params["T"]) / int(nt)),
        terminal=dict(payoff_type=str(params["payoff_type"]), K=float(params["K"])),
        boundary=dict(bc_type=str(params["bc_type"])),
        output=dict(return_surface=bool(return_surface)),
    )

    return inp, out, meta

# ============================================================
# Dispatcher
# ============================================================

def sample_pde(pde_name: str, rng=None, **kwargs):
    pde = pde_name.lower()
    if pde == "poisson":
        return sample_poisson(rng, **kwargs)
    if pde == "schrodinger":
        return sample_schrodinger(rng, **kwargs)
    if pde in ("kuramoto_sivashinsky", "ks"):
        return sample_kuramoto_sivashinsky(rng, **kwargs)
    if pde in ("navier_stokes", "ns"):
        return sample_navier_stokes(rng, **kwargs)
    if pde in ("black_scholes", "bs"):
        return sample_black_scholes(rng, **kwargs)
    raise ValueError(f"Unknown PDE {pde_name}")

