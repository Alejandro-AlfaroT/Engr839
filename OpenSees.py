#!/usr/bin/env python3
"""
OpenSees modal analysis of the 126 m pier, with similitude scaling.

Builds the pier as an elastic beam-column cantilever, rigid base, self-mass
lumped at the nodes, deck mass on top. Reports periods, mode shapes and
effective modal mass. Same model at any geometric scale, under three mass
similitudes, so the scaling decisions in the test plan can be checked.

    pip install openseespy numpy matplotlib
    python ps_opensees_modal.py                          # prototype
    python ps_opensees_modal.py --scale 8                # 1/8, geometry only
    python ps_opensees_modal.py --scale 8 --mass artificial
    python ps_opensees_modal.py --scale 8 --mass under --Sa 2    # Shi et al. 2025
    python ps_opensees_modal.py --scale 8 --ps-only      # PS alone on a table
    python ps_opensees_modal.py --transverse             # sway the other way

Similitude (same material, S_E = S_sigma = 1, length scale S_L = 1/scale):
    none        geometry scaled, nothing added. Gravity stress is S_L of the
                prototype's. Periods scale by S_L. Rocking is wrong.
    artificial  density scaled by 1/S_L with added mass so Cauchy and Froude
                both hold: S_a = 1, S_T = sqrt(S_L). Heavy; often exceeds table
                capacity.
    under       underartificial mass with acceleration scale S_a > 1
                (Shi et al. 2025 used S_a = 2): S_rho = 1/(S_a*S_L),
                S_T = sqrt(S_L/S_a). Lighter model, input motion scaled up.

For the hybrid simulation the numerical substructure runs at prototype scale
and the interface converts by S_L (displacement) and S_L^2 (force); gravity on
the specimen is supplied by the axial rig, not by similitude. The scaled runs
here are for the methods comparison and for checking a standalone PS test.

Units: kN, m, s, tonnes.
"""

import argparse
from dataclasses import dataclass, replace

import numpy as np
import openseespy.opensees as ops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

g = 9.81
C_MODE = "#0E7FA0"; C_PS = "#B8481B"; C_INK = "#12171B"
C_MUTED = "#5A6670"; C_GRID = "#CFD7DC"; C_CONC = "#C8D0D4"


# --------------------------------------------------------------------------
# prototype
# --------------------------------------------------------------------------

@dataclass
class Pier:
    H: float = 126.0        # m
    H_PS: float = 31.2      # m
    seg_h: float = 5.2      # m
    A: float = 38.52        # m^2
    I: float = 650.0        # m^4, vertical section axis -> longitudinal sway
    D: float = 12.0         # m, section depth in the sway direction
    E: float = 30.0e6       # kPa (30 GPa, C40)
    gamma: float = 25.0     # kN/m^3
    W_deck: float = 59.4e3    # kN, deck-only tributary weight: Mei et al. (2019) Fig. 14 node loads
                              # (37881 + 36847 + 91208 = 165936 kN) minus their pier self-weight
                              # (126.06 m x 33.8 m2 x 25 kN/m3 = 106.5 MN). Pier mass is distributed
                              # along the elements here, so only the deck share goes on the tip.
    top: str = "free"         # "free" (cantilever) | "fixed" (girder prevents pier-top rotation) | "spring"
    k_top: float = 0.0        # kN·m/rad rotational spring at the pier top, used when top == "spring"
    n_el: int = 26
    n_modes: int = 7          # 7 modes reach 95 % cumulative effective mass
    label: str = "prototype"


TRANSVERSE = dict(I=513.0, D=10.7)


# --------------------------------------------------------------------------
# similitude
# --------------------------------------------------------------------------

@dataclass
class Similitude:
    S_L: float          # length, model/prototype
    S_a: float          # acceleration
    S_rho: float        # mass density
    S_T: float          # time
    S_E: float = 1.0
    S_sigma: float = 1.0
    S_eps: float = 1.0

    @property
    def S_F(self):      # force = stress * area
        return self.S_sigma * self.S_L**2

    @property
    def S_m(self):      # mass = density * volume
        return self.S_rho * self.S_L**3

    @property
    def S_f(self):
        return 1.0 / self.S_T

    def table(self):
        rows = [
            ("length",         "m",     self.S_L),
            ("elastic modulus","kN/m²", self.S_E),
            ("stress",         "kN/m²", self.S_sigma),
            ("strain",         "—",     self.S_eps),
            ("force",          "kN",    self.S_F),
            ("acceleration",   "m/s²",  self.S_a),
            ("mass density",   "t/m³",  self.S_rho),
            ("mass",           "t",     self.S_m),
            ("time",           "s",     self.S_T),
            ("frequency",      "Hz",    self.S_f),
            ("gravity",        "m/s²",  1.0),
        ]
        out = ["  quantity          unit     model / prototype"]
        for n, u, v in rows:
            out.append(f"  {n:<17} {u:<8} {v:>12.5g}")
        return "\n".join(out)


def similitude(scale: float, mass: str, Sa: float) -> Similitude:
    S_L = 1.0 / scale
    if mass == "none":
        # same material, nothing added: Cauchy holds (S_T = S_L), Froude does not
        return Similitude(S_L=S_L, S_a=1.0 / S_L, S_rho=1.0, S_T=S_L)
    if mass == "artificial":
        # Cauchy + Froude with S_a = 1  ->  S_rho = 1/S_L, S_T = sqrt(S_L)
        return Similitude(S_L=S_L, S_a=1.0, S_rho=1.0 / S_L, S_T=np.sqrt(S_L))
    if mass == "under":
        # underartificial mass with chosen S_a  ->  S_rho = 1/(S_a S_L), S_T = sqrt(S_L/S_a)
        return Similitude(S_L=S_L, S_a=Sa, S_rho=1.0 / (Sa * S_L), S_T=np.sqrt(S_L / Sa))
    raise ValueError(mass)


def scaled_pier(p: Pier, s: Similitude, label: str) -> Pier:
    """Geometry scaled by S_L; density scaled by S_rho via gamma."""
    return replace(
        p,
        H=p.H * s.S_L, H_PS=p.H_PS * s.S_L, seg_h=p.seg_h * s.S_L,
        A=p.A * s.S_L**2, I=p.I * s.S_L**4, D=p.D * s.S_L,
        E=p.E * s.S_E,
        gamma=p.gamma * s.S_rho,
        W_deck=p.W_deck * s.S_m,
        label=label,
    )


# --------------------------------------------------------------------------
# OpenSees model
# --------------------------------------------------------------------------

def modal(p: Pier):
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    n = p.n_el
    L = p.H / n
    m_line = p.gamma * p.A / g
    z = np.linspace(0.0, p.H, n + 1)

    for i, zi in enumerate(z):
        ops.node(i + 1, 0.0, float(zi))
    ops.fix(1, 1, 1, 1)
    if p.top == "fixed":
        ops.fix(n + 1, 0, 0, 1)                      # girder clamps rotation, translation free

    masses = np.zeros(n + 1)
    for i in range(n + 1):
        m = 0.0
        if i > 0:
            m += m_line * L / 2
        if i < n:
            m += m_line * L / 2
        if i == n:
            m += p.W_deck / g
        masses[i] = m
        ops.mass(i + 1, m, 0.0, 0.0)          # lateral only

    ops.geomTransf("Linear", 1)
    for e in range(n):
        ops.element("elasticBeamColumn", e + 1, e + 1, e + 2, p.A, p.E, p.I, 1)
    if p.top == "spring" and p.k_top > 0:
        ops.node(n + 2, 0.0, float(p.H))
        ops.fix(n + 2, 1, 1, 1)
        ops.uniaxialMaterial("Elastic", 99, p.k_top)
        ops.element("zeroLength", 999, n + 2, n + 1, "-mat", 99, "-dir", 6)

    # Arpack needs a system comfortably larger than the number of modes asked for;
    # fall back to the dense solver on small meshes (e.g. --ps-only)
    solver = "-genBandArpack" if n >= 3 * p.n_modes else "-fullGenLapack"
    lam = ops.eigen(solver, p.n_modes)
    T = 2 * np.pi / np.sqrt(np.array(lam))

    Phi = np.zeros((n + 1, p.n_modes))
    meff = np.zeros(p.n_modes)
    for j in range(p.n_modes):
        phi = np.array([ops.nodeEigenvector(i + 1, j + 1, 1) for i in range(n + 1)])
        phi = phi / phi[-1]
        Phi[:, j] = phi
        Ln = (phi * masses).sum()
        Mn = (phi * masses * phi).sum()
        meff[j] = Ln**2 / Mn
    return z, Phi, T, meff / masses.sum(), masses.sum()


def report(p: Pier, z, Phi, T, meff_frac, mtot, title):
    print(f"\n=== {title} ===")
    print(f"  H = {p.H:.2f} m   A = {p.A:.4g} m²   I = {p.I:.4g} m⁴   "
          f"γ = {p.gamma:.3g} kN/m³   deck = {p.W_deck/g:.4g} t   total mass = {mtot:.4g} t   top = {p.top}"
          + (f" (k = {p.k_top:.3g} kN·m/rad)" if p.top == "spring" else ""))
    cum = np.cumsum(meff_frac)
    for j in range(p.n_modes):
        flag = "   <- 95 %" if (cum[j] >= 0.95 and (j == 0 or cum[j-1] < 0.95)) else ""
        print(f"  mode {j+1}:  T = {T[j]:7.4f} s   f = {1/T[j]:7.2f} Hz   eff. mass = {100*meff_frac[j]:5.1f} %"
              f"   cumulative = {100*cum[j]:5.1f} %{flag}")
    if cum[-1] < 0.95:
        print(f"  cumulative effective mass {100*cum[-1]:.1f} % < 95 % -- raise --modes")
    if p.H_PS < p.H - 1e-9:
        i_ps = int(np.argmin(np.abs(z - p.H_PS)))
        print(f"  mode-shape ordinates at z = {z[i_ps]:.2f} m (top of PS, tip = 1): "
              + "  ".join(f"m{j+1} {Phi[i_ps, j]:+.3f}" for j in range(p.n_modes)))


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def figures(p: Pier, z, Phi, T, tag):
    zf = np.linspace(0, p.H, 400)
    joints = np.arange(0, p.H_PS + 1e-9, p.seg_h)
    has_ps = p.H_PS < p.H - 1e-9

    fig, ax = plt.subplots(figsize=(4.6, 7.0))
    styles = ["-", "--", ":", "-.", (0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2))]
    cmap = plt.get_cmap("viridis")
    for j in range(p.n_modes):
        ax.plot(Phi[:, j], z, ls=styles[j % len(styles)], color=cmap(0.85 * j / max(p.n_modes - 1, 1)),
                lw=1.8, label=f"mode {j+1}   T = {T[j]:.3g} s")
    if has_ps:
        ax.axhspan(0, p.H_PS, color=C_PS, alpha=0.08, lw=0)
        ax.axhline(p.H_PS, color=C_PS, lw=1.0, ls="--", alpha=0.9)
        ax.text(0.98, p.H_PS + 0.012 * p.H, "PS", fontsize=9, color=C_PS, ha="right", fontweight="bold")
    ax.axvline(0, color=C_GRID, lw=0.8)
    lim = max(1.05, np.abs(Phi).max() * 1.05)
    ax.set_xlim(-lim, lim); ax.set_ylim(0, p.H)
    ax.set_xlabel("mode-shape ordinate (tip = 1)", fontsize=9.5, color=C_INK)
    ax.set_ylabel("height above base (m)", fontsize=9.5, color=C_INK)
    ax.grid(True, color=C_GRID, lw=0.6, alpha=0.8)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(colors=C_MUTED, labelsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.set_title(f"OpenSees mode shapes — {p.label}", fontsize=11.5, fontweight="bold", color=C_INK, loc="left")
    fig.tight_layout()
    fig.savefig(f"ps_modes_{tag}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(f"ps_modes_{tag}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, p.n_modes, figsize=(1.95 * p.n_modes + 0.5, 4.9), sharey=True)
    for j, ax in enumerate(np.atleast_1d(axes)):
        amp = 0.22 * p.H / np.abs(Phi[:, j]).max()
        ux = amp * np.interp(zf, z, Phi[:, j])
        ax.add_patch(Rectangle((-p.D/2, 0), p.D, p.H, facecolor="none", edgecolor=C_GRID, lw=1.0, ls="--"))
        left = np.column_stack([ux - p.D/2, zf]); right = np.column_stack([ux + p.D/2, zf])
        ax.add_patch(Polygon(np.vstack([left, right[::-1]]), closed=True,
                             facecolor=C_CONC, edgecolor=C_INK, lw=1.2, zorder=3))
        if has_ps:
            mask = zf <= p.H_PS
            ax.add_patch(Polygon(np.vstack([left[mask], right[mask][::-1]]), closed=True,
                                 facecolor=C_PS, alpha=0.35, edgecolor="none", zorder=4))
        for zj in joints:
            xj = amp * np.interp(zj, z, Phi[:, j])
            ax.plot([xj - p.D/2, xj + p.D/2], [zj, zj], color=C_PS, lw=0.9, zorder=5)
        ax.plot(ux, zf, color=C_MODE, lw=1.6, zorder=6)
        fd = 0.032 * p.H
        ax.add_patch(Rectangle((-p.D/2 - 0.33*p.D, -fd), p.D + 0.66*p.D, fd,
                               facecolor="#6E7276", edgecolor=C_INK, lw=1, zorder=2))
        ax.set_xlim(-0.30*p.H, 0.30*p.H); ax.set_ylim(-1.2*fd, p.H * 1.05)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"mode {j+1}\nT = {T[j]:.3g} s", fontsize=10.5, color=C_INK, fontweight="bold")
    fig.suptitle(f"Deformed shapes — {p.label}", fontsize=12.5, fontweight="bold", color=C_INK, y=1.0)
    fig.text(0.5, -0.01, "PS shaded, joints marked, amplitude exaggerated for visibility",
             fontsize=8.5, color=C_MUTED, ha="center", va="top")
    fig.tight_layout()
    fig.savefig(f"ps_deformed_{tag}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(f"ps_deformed_{tag}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote ps_modes_{tag}.png/.pdf and ps_deformed_{tag}.png/.pdf")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="OpenSees modal analysis with similitude scaling")
    ap.add_argument("--scale", type=float, default=1.0, help="geometric scale, e.g. 8 for 1/8")
    ap.add_argument("--mass", choices=["none", "artificial", "under"], default="none",
                    help="mass similitude for the scaled model")
    ap.add_argument("--Sa", type=float, default=2.0, help="acceleration scale for --mass under")
    ap.add_argument("--ps-only", action="store_true",
                    help="model PS alone as a cantilever with everything above lumped on top")
    ap.add_argument("--transverse", action="store_true", help="sway in the transverse direction")
    ap.add_argument("--top", choices=["free", "fixed", "spring"], default="free",
                    help="pier-top rotation: free cantilever, fixed by the girder, or a rotational spring")
    ap.add_argument("--k-top", type=float, default=0.0, help="rotational spring stiffness, kN·m/rad (with --top spring)")
    ap.add_argument("--deck", type=float, default=Pier.W_deck, help="deck-only tributary weight on the pier top, kN (default 59400 from Mei Fig. 14)")
    ap.add_argument("--modes", type=int, default=Pier.n_modes,
                    help="number of modes to extract (default 7, which reaches 95 %% cumulative mass)")
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()

    proto = Pier(**TRANSVERSE, label="prototype, transverse") if a.transverse else Pier()
    proto = replace(proto, n_modes=a.modes, top=a.top, k_top=a.k_top, W_deck=a.deck,
                    label=proto.label + f", top {a.top}")

    z, Phi, T, mf, mt = modal(proto)
    report(proto, z, Phi, T, mf, mt, proto.label)
    T_proto = T.copy()
    if not a.no_fig:
        figures(proto, z, Phi, T, "prototype")

    if a.scale != 1.0:
        s = similitude(a.scale, a.mass, a.Sa)
        print(f"\n=== similitude: 1/{a.scale:g} scale, mass = {a.mass}"
              + (f", S_a = {a.Sa:g}" if a.mass == "under" else "") + " ===")
        print(s.table())
        label = f"1/{a.scale:g} model, {a.mass} mass"
        mdl = scaled_pier(proto, s, label)
        z, Phi, T, mf, mt = modal(mdl)
        report(mdl, z, Phi, T, mf, mt, label)

        print("\n  period check  (model T should equal prototype T × S_T if the similitude holds)")
        for j in range(proto.n_modes):
            print(f"    mode {j+1}: model {T[j]:.4f} s   prototype×S_T {T_proto[j]*s.S_T:.4f} s   "
                  f"ratio {T[j]/(T_proto[j]*s.S_T):.3f}")
        self_m = proto.gamma * proto.A * proto.H / g * s.S_L**3
        need_m = proto.gamma * proto.A * proto.H / g * s.S_m
        print(f"\n  pier self-mass at this scale, same material   {self_m:8.2f} t")
        print(f"  pier mass the similitude requires              {need_m:8.2f} t")
        print(f"  added mass to distribute along the pier        {need_m - self_m:8.2f} t")
        if not a.no_fig:
            figures(mdl, z, Phi, T, f"scale{a.scale:g}_{a.mass}")

    if a.ps_only:
        if a.scale != 1.0:
            s = similitude(a.scale, a.mass, a.Sa)
            base = scaled_pier(proto, s, "")
        else:
            base = proto
        above = base.gamma * base.A * (base.H - base.H_PS)      # weight of pier above PS
        ps = replace(base, H=base.H_PS, n_el=24, W_deck=base.W_deck + above,
                     label=f"PS alone, 1/{a.scale:g}")
        z, Phi, T, mf, mt = modal(ps)
        report(ps, z, Phi, T, mf, mt, ps.label + "  (pier above + deck lumped on top)")
        print("  a lumped mass cannot carry modes 2 and 3 of the flexible continuation — "
              "this is the standalone-shake-table problem in numbers.")
        if not a.no_fig:
            figures(ps, z, Phi, T, f"ps_only_scale{a.scale:g}")


if __name__ == "__main__":
    main()