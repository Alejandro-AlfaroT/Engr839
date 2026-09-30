#!/usr/bin/env python3
"""
Eigenvalue model of the full 126 m pier, and the joint-opening check.

Cantilever, rigid base, deck mass lumped at the top. Euler-Bernoulli beam
elements, self-mass lumped at the nodes. Solves for periods and mode shapes,
builds the modal moment and shear profiles, scales them by a design spectrum,
combines SRSS, and compares the moment demand up the height against the
decompression moment M_dec(z) = (N(z) + P_PT) * S/A.

    pip install numpy scipy matplotlib
    python ps_modal.py

Outputs
    ps_modes.png / .pdf      first three mode shapes
    ps_joints.png / .pdf     moment demand vs M_dec up PS, joint by joint

Everything in PARAMS is an input you must be able to defend. The three marked
PLACEHOLDER are not from your prototype and need replacing.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# PARAMETERS
# --------------------------------------------------------------------------


@dataclass
class Params:
    # pier
    H: float = 126.0            # m, full pier height
    H_PS: float = 31.2          # m, PS height
    seg_h: float = 5.2          # m, segment height (joints at multiples)
    A: float = 38.52            # m^2, net section area
    I: float = 650.0            # m^4, about the section's vertical axis
                                #      -> longitudinal sway. Use 513 for transverse.
    c: float = 6.522            # m, centroid to the far extreme fibre (12.0 - 5.478)
    E: float = 30.0e3           # MPa, C40 concrete  (30 GPa)
    gamma: float = 25.0         # kN/m^3, reinforced concrete unit weight
    n_el: int = 26              # elements

    # loads on the joints
    W_deck: float = 45.0e3      # kN, tributary deck weight at pier top   PLACEHOLDER
    P_PT: float = 40.0e3        # kN, post-tensioning force through the joints   PLACEHOLDER

    # design spectrum (ASCE 7 shape, 5% damping)                         PLACEHOLDER
    S_DS: float = 1.00          # g
    S_D1: float = 0.60          # g·s
    T_L: float = 8.0            # s

    n_modes: int = 3

    # post-decompression cap: base moment rises to alpha * M_dec as the joint
    # rocks (tendon stretch + ED bars). Governs what the upper joints see.
    alpha_cap: float = 1.3      # PLACEHOLDER — from the joint's M-theta model


P = Params()
g = 9.81

C_MODE = "#0E7FA0"
C_INK = "#12171B"
C_DEC = "#B8481B"
C_MUTED = "#5A6670"
C_GRID = "#CFD7DC"


# --------------------------------------------------------------------------
# finite element cantilever
# --------------------------------------------------------------------------

def beam_matrices(p: Params):
    """Global K and lumped M for a vertical cantilever. DOFs per node:
    lateral translation u, rotation θ. Node 0 at the base is fixed."""
    n = p.n_el
    L = p.H / n
    EI = p.E * 1e3 * p.I                       # kN·m^2  (MPa -> kPa)
    m_line = p.gamma * p.A / g                 # t/m   (kN/m^3 * m^2 / (m/s^2))

    ndof = 2 * (n + 1)
    K = np.zeros((ndof, ndof))
    M = np.zeros((ndof, ndof))

    k = EI / L**3 * np.array([
        [12,      6*L,   -12,     6*L],
        [6*L,   4*L*L,  -6*L,   2*L*L],
        [-12,    -6*L,    12,    -6*L],
        [6*L,   2*L*L,  -6*L,   4*L*L],
    ])
    for e in range(n):
        idx = [2*e, 2*e+1, 2*e+2, 2*e+3]
        K[np.ix_(idx, idx)] += k
        # lumped translational mass, half the element to each end
        M[2*e, 2*e] += m_line * L / 2
        M[2*e+2, 2*e+2] += m_line * L / 2

    # deck mass at the top node
    M[2*n, 2*n] += p.W_deck / g

    # tiny rotational inertia so M stays non-singular for eigh
    for i in range(n + 1):
        M[2*i+1, 2*i+1] += 1e-6 * m_line * L**3

    free = list(range(2, ndof))                 # fix u and θ at node 0
    return K, M, free, L


def modal(p: Params):
    K, M, free, L = beam_matrices(p)
    Kf, Mf = K[np.ix_(free, free)], M[np.ix_(free, free)]
    w2, phi = eigh(Kf, Mf)
    order = np.argsort(w2)
    w2, phi = w2[order], phi[:, order]

    n = p.n_el
    z = np.linspace(0, p.H, n + 1)

    # translational components only, normalised to unit tip displacement
    Phi = np.zeros((n + 1, p.n_modes))
    for j in range(p.n_modes):
        v = np.zeros(2 * (n + 1))
        v[free] = phi[:, j]
        u = v[0::2]
        Phi[:, j] = u / u[-1]

    # lumped translational masses per node
    m = np.zeros(n + 1)
    m_line = p.gamma * p.A / g
    m[:-1] += m_line * L / 2
    m[1:] += m_line * L / 2
    m[-1] += p.W_deck / g

    T = 2 * np.pi / np.sqrt(w2[:p.n_modes])
    Gamma = (Phi.T @ m) / np.einsum("ij,i,ij->j", Phi, m, Phi)   # participation
    Meff = (Phi.T @ m) ** 2 / np.einsum("ij,i,ij->j", Phi, m, Phi)
    return z, m, Phi, T, Gamma, Meff


# --------------------------------------------------------------------------
# spectrum, modal forces, internal-force profiles
# --------------------------------------------------------------------------

def Sa(T, p: Params):
    """ASCE 7 design spectrum, g."""
    T0, Ts = 0.2 * p.S_D1 / p.S_DS, p.S_D1 / p.S_DS
    if T < T0:
        return p.S_DS * (0.4 + 0.6 * T / T0)
    if T <= Ts:
        return p.S_DS
    if T <= p.T_L:
        return p.S_D1 / T
    return p.S_D1 * p.T_L / T**2


def profiles(z, m, Phi, Gamma, T, p: Params):
    """Per-mode shear V_n(z) and moment M_n(z) from the modal inertia forces
    s_n = Γ_n m φ_n Sa_n. Returns arrays [node, mode] in kN and kN·m."""
    nn, nm = Phi.shape
    V = np.zeros((nn, nm))
    Mo = np.zeros((nn, nm))
    for j in range(nm):
        f = Gamma[j] * m * Phi[:, j] * Sa(T[j], p) * g     # kN at each node
        for i in range(nn):
            above = np.arange(i, nn)
            V[i, j] = f[above].sum()
            Mo[i, j] = (f[above] * (z[above] - z[i])).sum()
    return V, Mo


def m_dec(z, p: Params):
    """Decompression moment at height z, kN·m."""
    N_self = p.gamma * p.A * (p.H - z)          # weight of pier above z
    N = N_self + p.W_deck + p.P_PT
    S = p.I / p.c
    return N * S / p.A, N


# --------------------------------------------------------------------------

def main():
    p = P
    z, m, Phi, T, Gamma, Meff = modal(p)
    V, Mo = profiles(z, m, Phi, Gamma, T, p)
    Msrss = np.sqrt((Mo**2).sum(axis=1))
    Vsrss = np.sqrt((V**2).sum(axis=1))
    Mdec, N = m_dec(z, p)

    # capped demand: base rocks at alpha*M_dec; the pier above sees the
    # first-mode moment shape scaled to that cap
    Mcap = p.alpha_cap * Mdec[0] * Mo[:, 0] / Mo[0, 0]

    mtot = m.sum()

    # ---------------- console ----------------
    print("=== Modal properties (full 126 m pier, rigid base) ===")
    for j in range(p.n_modes):
        print(f"  mode {j+1}:  T = {T[j]:6.2f} s   f = {1/T[j]:5.2f} Hz   "
              f"Sa = {Sa(T[j], p):.3f} g   "
              f"eff. mass = {100*Meff[j]/mtot:5.1f} %   "
              f"base M = {Mo[0,j]/1e3:8.0f} MN·m   base V = {V[0,j]/1e3:7.0f} MN")
    print(f"  first three modes carry {100*Meff.sum()/mtot:.1f} % of the mass")
    print(f"  SRSS base moment {Msrss[0]/1e3:.0f} MN·m,  base shear {Vsrss[0]/1e3:.0f} MN")
    print(f"  mode-2 / mode-1 at the base:  moment {Mo[0,1]/Mo[0,0]:.2f}   shear {V[0,1]/V[0,0]:.2f}")

    print("\n=== Joint check inside PS ===")
    print(f"  N at base {N[0]/1e3:.0f} MN (self {p.gamma*p.A*p.H/1e3:.0f} + deck "
          f"{p.W_deck/1e3:.0f} + PT {p.P_PT/1e3:.0f}),  S/A = {p.I/p.c/p.A:.2f} m")
    print(f"  base cap alpha = {p.alpha_cap}  ->  base moment held near {p.alpha_cap*Mdec[0]/1e3:.0f} MN·m once rocking")
    print(f"  {'joint':>6} {'z (m)':>7} {'M_dec':>9} {'elastic':>9} {'capped':>9} {'cap/Mdec':>9}  opens under cap?")
    joints = np.arange(0, p.H_PS + 1e-9, p.seg_h)
    for k, zj in enumerate(joints):
        md = np.interp(zj, z, Mdec)
        m3 = np.interp(zj, z, Msrss)
        mc = np.interp(zj, z, Mcap)
        print(f"  {k+1:>6} {zj:7.1f} {md/1e3:9.0f} {m3/1e3:9.0f} {mc/1e3:9.0f} "
              f"{mc/md:9.2f}  {'YES' if mc > md else 'no'}")
    print("\n  elastic demand >> M_dec everywhere: the base WILL rock (by design).")
    print("  the capped column is what the upper joints actually see; cap/Mdec > 1 -> that joint opens too.")
    print("  PLACEHOLDERS in use: W_deck, P_PT, S_DS/S_D1 — replace before quoting numbers.")

    # ---------------- figure 1: mode shapes ----------------
    fig, ax = plt.subplots(figsize=(4.8, 7.2))
    styles = ["-", "--", ":"]
    for j in range(p.n_modes):
        ax.plot(Phi[:, j], z, ls=styles[j], color=C_MODE, lw=2.0,
                label=f"mode {j+1}   T = {T[j]:.2f} s")
    ax.axhline(p.H_PS, color=C_DEC, lw=1.0, ls="--", alpha=0.8)
    ax.text(-0.98, p.H_PS + 2, "top of PS", fontsize=8.5, color=C_DEC)
    ax.axvline(0, color=C_GRID, lw=0.8)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(0, p.H)
    ax.set_xlabel("mode shape, normalised to tip", fontsize=9.5, color=C_INK)
    ax.set_ylabel("height (m)", fontsize=9.5, color=C_INK)
    ax.grid(True, color=C_GRID, lw=0.6, alpha=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=C_MUTED, labelsize=8.5)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("First three modes, 126 m pier", fontsize=11.5,
                 fontweight="bold", color=C_INK, loc="left")
    fig.tight_layout()
    fig.savefig("ps_modes.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig("ps_modes.pdf", bbox_inches="tight", facecolor="white")

    # ---------------- figure 2: demand vs M_dec ----------------
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    zz = np.linspace(0, p.H_PS, 200)
    f_el = np.interp(zz, z, Msrss); f_cap = np.interp(zz, z, Mcap); f_dec = np.interp(zz, z, Mdec)
    ax.plot(f_el / 1e3, zz, ls=":", color=C_MODE, lw=1.8, label="elastic demand, SRSS 3 modes")
    ax.plot(f_cap / 1e3, zz, ls="-", color=C_INK, lw=2.2,
            label=f"demand once the base rocks (cap = {p.alpha_cap}·M_dec)")
    ax.plot(f_dec / 1e3, zz, ls="--", color=C_DEC, lw=2.2, label="M_dec = (N + P_PT)·S/A")

    # joints
    for k, zj in enumerate(joints):
        md = np.interp(zj, z, Mdec); mc = np.interp(zj, z, Mcap)
        ax.axhline(zj, color=C_GRID, lw=0.7)
        ax.plot([md/1e3], [zj], "o", ms=6, color=C_DEC, mec="white", mew=1.2, zorder=5)
        ax.text(2, zj + 0.5, f"joint {k+1}" + ("  opens" if mc > md else "  stays closed"),
                fontsize=8, color=C_DEC if mc > md else C_MUTED)

    ax.set_xlim(0, max(Msrss[0], Mdec[0]) / 1e3 * 1.12)
    ax.set_ylim(0, p.H_PS + 1)
    ax.set_xlabel("bending moment (MN·m)", fontsize=10, color=C_INK)
    ax.set_ylabel("height above base (m)", fontsize=10, color=C_INK)
    ax.grid(True, color=C_GRID, lw=0.6, alpha=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=C_MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("Which joints in PS open: demand vs decompression moment",
                 fontsize=12, fontweight="bold", color=C_INK, loc="left", pad=10)
    fig.text(0.01, -0.01,
             f"Longitudinal sway. E = {p.E/1e3:.0f} GPa, A = {p.A} m², I = {p.I:.0f} m⁴, "
             f"deck {p.W_deck/1e3:.0f} MN, PT {p.P_PT/1e3:.0f} MN, "
             f"spectrum S_DS = {p.S_DS} g, S_D1 = {p.S_D1} g·s, cap α = {p.alpha_cap}. "
             "Deck, PT, spectrum and α are placeholders. Elastic modes; the capped "
             "curve is a bound, not a nonlinear analysis.",
             fontsize=7.5, color=C_MUTED, va="top")
    fig.tight_layout()
    fig.savefig("ps_joints.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig("ps_joints.pdf", bbox_inches="tight", facecolor="white")
    print("\nwrote ps_modes.png/.pdf and ps_joints.png/.pdf")


if __name__ == "__main__":
    main()