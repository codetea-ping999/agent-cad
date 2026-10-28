"""Bolted joint calculation after VDI 2230 Part 1 (2015), simplified, with the torque
relation of JIS B 1083 (= VDI 2230 eq. R7/1).

    j = Joint("pinch", BOLTS["M5-12.9"], n=2, torque_Nm=(5.0, 6.0), plates=[Plate(3.75, 71_700, 13.0)],
              gap_mm=2.5, tapped=True, thread_mat=MATERIALS["A7075-T6"], engagement_mm=7.5, ...)
    r = calc(j)          # every intermediate value and the checks with their limits

What is calculated, per bolt (VDI 2230 step in brackets):
  - compliance of the bolt (R3: head, free shank in the clamp and in any gap, engaged
    thread and the tapped hole or nut) and of the clamped plates (deformation cone or
    cone + sleeve, R3), plus any extra series compliance of a slotted clamp ring;
  - load factor Phi_n = n * delta_P / (delta_S + delta_P) (R3);
  - assembly preload from the tightening torque at the friction extremes:
    F_M = M_A / (0.16 P + 0.58 d2 mu_G + D_Km / 2 mu_K)  (R7/1, JIS B 1083 K-formula);
    the scatter of torque and friction gives F_M,min .. F_M,max and the tightening factor
    alpha_A = F_M,max / F_M,min (R1);
  - preload loss by embedding F_Z = f_Z / (delta_S + delta_P) (R5);
  - residual clamp force F_M,min - F_Z - (1 - Phi_n) F_A,max against what the joint
    has to hold (R2, R10): friction against slip for a clamp, separation for a flange;
  - bolt stress at assembly (R8: von Mises of tension and full thread torque, against
    nu * Rp0.2) and in service (R8: torsion relaxed to k_tau = 0.5);
  - alternating stress against the endurance limit of a rolled thread (R9);
  - surface pressure under the head (R10) and stripping of the internal thread against
    the bolt's tensile capacity (R11: the bolt must break before the thread strips).

Material data are handbook values; friction coefficients are the VDI 2230 Table A5
class B range (0.10 .. 0.16) for black-oxide or zinc-plated steel, oiled or greased.
"""
import math
from dataclasses import dataclass, field, asdict

E_STEEL = 205_000.0


@dataclass(frozen=True)
class Bolt:
    name: str
    d: float                   # nominal diameter
    P: float                   # pitch (coarse, JIS B 0205)
    grade: str                 # JIS B 1051 strength class, or the alloy
    Rp02_MPa: float            # minimum lower yield / 0.2 % proof stress
    Rm_MPa: float              # minimum tensile strength
    E_MPa: float
    head_d: float              # head bearing outer diameter d_W (cap screw: d_k, JIS B 1176)
    head_kind: str = "socket"  # "socket" (VDI: 0.4 d head compliance length) or "hex" (0.5 d)
    rolled_after_ht: bool = False

    @property
    def d2(self):
        return self.d - 0.649519 * self.P

    @property
    def d3(self):
        return self.d - 1.226869 * self.P

    @property
    def As(self):              # tensile stress area (JIS B 1082)
        return math.pi / 4 * ((self.d2 + self.d3) / 2) ** 2

    @property
    def A3(self):
        return math.pi / 4 * self.d3 ** 2

    @property
    def AN(self):
        return math.pi / 4 * self.d ** 2


# JIS B 1176 hexagon socket head cap screws (d_k max) and JIS B 1180 hexagon bolts with a
# JIS B 1256 washer (washer outer diameter as the bearing diameter)
BOLTS = {
    "M5-12.9": Bolt("M5×0.8 12.9", 5.0, 0.8, "12.9", 1100, 1220, E_STEEL, 8.5),
    "M5-10.9": Bolt("M5×0.8 10.9", 5.0, 0.8, "10.9", 940, 1040, E_STEEL, 8.5),
    "M5-Ti64": Bolt("M5×0.8 Ti-6Al-4V", 5.0, 0.8, "Ti-6Al-4V", 830, 900, 113_800, 8.5),
    "M8-8.8": Bolt("M8×1.25 8.8 + 座金", 8.0, 1.25, "8.8", 640, 800, E_STEEL, 16.0, "hex"),
    "M8-10.9": Bolt("M8×1.25 10.9 + 座金", 8.0, 1.25, "10.9", 940, 1040, E_STEEL, 16.0, "hex"),
}


@dataclass(frozen=True)
class Plate:
    """One clamped part in the bolt's grip: thickness, modulus and the outer diameter of
    the material around the hole that the deformation cone may use (D_A)."""
    t: float
    E_MPa: float
    DA: float
    name: str = ""


@dataclass(frozen=True)
class Joint:
    name: str
    bolt: Bolt
    n: int                                   # bolts sharing the clamp requirement
    torque_Nm: tuple                         # (min, max) tightening torque per bolt
    plates: tuple                            # Plate, head side first
    hole_d: float                            # clearance hole d_h
    tapped: bool                             # True: into a tapped hole (ESV); False: nut (DSV)
    engagement_mm: float                     # usable thread engagement (tapped hole or nut height)
    thread_Rm_MPa: float                     # tensile strength of the internal-thread material
    bearing_Rm_MPa: float                    # tensile strength of the material under the head
    thread_E_MPa: float = 71_000.0           # modulus of the tapped part (tapped joints)
    gap_mm: float = 0.0                      # open gap the bolt spans (slotted / split clamp)
    extra_compliance: float = 0.0            # series compliance of the clamped side (mm/N), e.g. a clamp ring
    interfaces: int = 0                      # inner interfaces in the clamp path (VDI Table 5)
    embed_extra_um: float = 0.0              # embedding of clamp contacts, mapped to the bolt axis (um)
    mu_G: tuple = (0.10, 0.16)
    mu_K: tuple = (0.10, 0.16)
    load_intro_n: float = 0.5                # VDI load introduction factor (assumed mid-height)
    FA_static_N: float = 0.0                 # axial working load per bolt, static maximum
    FA_alt_N: float = 0.0                    # axial working load per bolt that cycles 0 .. FA_alt
    F_req_N: float = 0.0                     # residual clamp force per bolt the function needs
    load_with_req: bool = True               # the working loads act while F_req is needed (same load case)
    F_req_basis: str = ""
    nu: float = 0.9                          # utilisation of Rp0.2 allowed at assembly
    SF_fatigue: float = 1.2
    SF_strip: float = 1.0                    # thread stripping force / bolt breaking force
    SF_bearing: float = 1.0
    rz_um: str = "10-40"                     # surface roughness band for embedding


# VDI 2230 Table 5: embedding per surface under axial load (um), by roughness Rz
EMBED_UM = {"<10": dict(thread=3.0, bearing=2.5, inner=1.5),
            "10-40": dict(thread=3.0, bearing=3.0, inner=2.0)}


def _rule(name, ok, value, limit, basis, **kw):
    return {"check": name, "pass": bool(ok), "value": value, "limit": limit, "basis": basis, **kw}


def bolt_compliance(j: Joint):
    """delta_S (mm/N) after VDI 2230 R3: head, unthreaded shank, free loaded thread,
    engaged thread, tapped hole / nut. Cap screws are assumed fully threaded in the grip
    (short screws are), so the free length uses A3."""
    b = j.bolt
    lK = sum(p.t for p in j.plates)
    l_free = lK + j.gap_mm
    head = (0.4 if b.head_kind == "socket" else 0.5) * b.d / (b.E_MPa * b.AN)
    free = l_free / (b.E_MPa * b.A3)
    thread = 0.5 * b.d / (b.E_MPa * b.A3)
    if j.tapped:
        nut = 0.33 * b.d / (j.thread_E_MPa * b.AN)
    else:
        nut = 0.4 * b.d / (E_STEEL * b.AN)
    return head + free + thread + nut, dict(head=head, free=free, thread=thread, nut=nut, l_free=l_free)


def plate_compliance(j: Joint):
    """delta_P (mm/N) of the clamped plates, VDI 2230 R3: deformation cone (tan phi from
    the plate geometry), or cone + sleeve where the part is narrower than the cone."""
    b = j.bolt
    dw, dh = b.head_d, j.hole_d
    lK = sum(p.t for p in j.plates)
    if lK <= 0:
        return 0.0, {}
    # one equivalent plate: modulus weighted by thickness, D_A the smallest around the hole
    E = lK / sum(p.t / p.E_MPa for p in j.plates)
    DA = min(p.DA for p in j.plates)
    w = 2 if j.tapped else 1
    bl, y = lK / dw, DA / dw
    tphi = (0.348 + 0.013 * math.log(bl) + 0.193 * math.log(y)) if j.tapped else \
        (0.362 + 0.032 * math.log(bl / 2) + 0.153 * math.log(y))
    DA_gr = dw + w * lK * tphi
    if DA >= DA_gr:
        dP = 2 * math.log((dw + dh) * (dw + w * lK * tphi - dh) / ((dw - dh) * (dw + w * lK * tphi + dh))) \
            / (w * E * math.pi * dh * tphi)
        kind = "cone"
    else:
        cone = 2 / (w * dh * tphi) * math.log((dw + dh) * (DA - dh) / ((dw - dh) * (DA + dh)))
        sleeve = 4 / (DA ** 2 - dh ** 2) * (lK - (DA - dw) / (w * tphi))
        dP = (cone + sleeve) / (E * math.pi)
        kind = "cone+sleeve"
    return dP + j.extra_compliance, dict(tan_phi=tphi, DA=DA, DA_limit=DA_gr, model=kind, E_eq=E,
                                         cone=dP, extra=j.extra_compliance)


def torque_factor(b: Bolt, dKm, mu_G, mu_K):
    """M_A / F_M in mm (JIS B 1083: K d = 0.16 P + 0.58 d2 mu_G + D_Km / 2 mu_K)."""
    return 0.16 * b.P + 0.58 * b.d2 * mu_G + dKm / 2 * mu_K


def assembly_stress(b: Bolt, FM, mu_G, plastic=True):
    """von Mises stress at assembly (VDI R7/R8): tension on d0 = (d2 + d3) / 2 and the thread
    torque M_G = F_M (0.16 P + 0.58 d2 mu_G) on the same section. VDI rates the assembly
    state with the plastic torsion modulus pi d0^3 / 12 (the 3/2 in its F_M,zul formula);
    plastic=False gives the elastic shear stress (pi d0^3 / 16) used in service."""
    d0 = (b.d2 + b.d3) / 2
    A0, Wp = math.pi / 4 * d0 ** 2, math.pi / (12 if plastic else 16) * d0 ** 3
    MG = FM * (0.16 * b.P + 0.58 * b.d2 * mu_G)
    s, t = FM / A0, MG / Wp
    return math.sqrt(s ** 2 + 3 * t ** 2), s, t


def FM_allowed(b: Bolt, mu_G, nu):
    """Largest assembly preload with the von Mises stress at nu * Rp0.2 (VDI R7)."""
    d0 = (b.d2 + b.d3) / 2
    A0 = math.pi / 4 * d0 ** 2
    # tau / sigma with the plastic torsion modulus: VDI's 3/2 d2/d0 (P / (pi d2) + 1.155 mu_G),
    # written with the same thread-torque coefficients as the torque formula
    k = 3 * (0.16 * b.P + 0.58 * b.d2 * mu_G) / d0
    return A0 * nu * b.Rp02_MPa / math.sqrt(1 + 3 * k ** 2)


def endurance_limit(b: Bolt, FSm=None):
    """Endurance limit of the thread, stress amplitude on As (VDI R9): rolled before heat
    treatment sigma_ASV = 0.85 (150 / d + 45); rolled after heat treatment it rises with the
    mean preload (factor 2 - F_Sm / F_0.2). Steel bolts only; a titanium bolt is given
    the same value scaled by its strength (estimate)."""
    s = 0.85 * (150 / b.d + 45)
    if b.rolled_after_ht and FSm is not None:
        s *= 2 - FSm / (b.As * b.Rp02_MPa)
    if b.E_MPa < 150_000:
        s *= b.Rm_MPa / 1220
    return s


def strip_area(b: Bolt, m):
    """Shear area of the internal thread over engagement m (basic profile, ISO 68-1):
    pi d m (1/2 + (d - D2) tan 30 / P), with C3 = 0.9 for thread bending (VDI R11)."""
    D2 = b.d2
    return 0.9 * math.pi * b.d * m * (0.5 + (b.d - D2) * math.tan(math.radians(30)) / b.P)


def _FA_with_req(j: Joint):
    """Working load that acts while the required clamp force F_req is needed."""
    return max(j.FA_static_N, j.FA_alt_N) if j.load_with_req else 0.0


def calc(j: Joint):
    b = j.bolt
    dS, dS_parts = bolt_compliance(j)
    dP, dP_info = plate_compliance(j)
    phi_K = dP / (dS + dP)
    phi_n = j.load_intro_n * phi_K
    dKm = (b.head_d + j.hole_d) / 2
    Mmin, Mmax = (t * 1000 for t in j.torque_Nm)
    FM_min = Mmin / torque_factor(b, dKm, j.mu_G[1], j.mu_K[1])
    FM_max = Mmax / torque_factor(b, dKm, j.mu_G[0], j.mu_K[0])
    alpha_A = FM_max / FM_min
    e = EMBED_UM[j.rz_um]
    fZ_um = e["thread"] + e["bearing"] * (1 if j.tapped else 2) + e["inner"] * j.interfaces + j.embed_extra_um
    FZ = fZ_um / 1000 / (dS + dP)
    FA_max = max(j.FA_static_N, j.FA_alt_N)          # peak working load: static or the cyclic peak
    F_res = FM_min - FZ - (1 - phi_n) * _FA_with_req(j)
    F_res_alt = FM_min - FZ - (1 - phi_n) * FA_max    # the joint must not open at the peak load either
    SG = F_res / j.F_req_N if j.F_req_N > 0 else float("inf")
    # bolt at assembly (worst: highest preload at the lowest thread friction... and the highest
    # friction at that torque for the torsion part: VDI uses mu_G,min with F_M,max)
    s_red_M, s_M, t_M = assembly_stress(b, FM_max, j.mu_G[0])
    FMzul = FM_allowed(b, j.mu_G[0], j.nu)
    nu_used = s_red_M / b.Rp02_MPa
    # in service: torsion relaxed to half (k_tau = 0.5), full working load added
    FS_max = FM_max + phi_n * FA_max
    d0 = (b.d2 + b.d3) / 2
    s_z = FS_max / (math.pi / 4 * d0 ** 2)
    t_el = assembly_stress(b, FM_max, j.mu_G[0], plastic=False)[2]
    s_red_B = math.sqrt(s_z ** 2 + 3 * (0.5 * t_el) ** 2)
    # fatigue
    sigma_a = phi_n * j.FA_alt_N / (2 * b.As)
    sigma_ASV = endurance_limit(b, FM_max + phi_n * j.FA_alt_N / 2)
    SD = sigma_ASV / sigma_a if sigma_a > 0 else float("inf")
    # head bearing
    Ap = math.pi / 4 * (b.head_d ** 2 - j.hole_d ** 2)
    p_max = (FM_max + phi_n * FA_max) / Ap
    pG = 0.9 * j.bearing_Rm_MPa
    # thread stripping (internal thread)
    A_strip = strip_area(b, j.engagement_mm)
    F_strip = A_strip * 0.6 * j.thread_Rm_MPa if j.tapped else float("inf")
    F_break = b.As * b.Rm_MPa
    rules = [
        _rule("assembly_stress", nu_used <= j.nu + 1e-9, round(nu_used, 3), j.nu,
              f"von Mises at F_M,max / Rp0.2 ({b.Rp02_MPa:g} MPa), VDI 2230 R8"),
        _rule("service_stress", s_red_B <= b.Rp02_MPa, round(s_red_B, 0), b.Rp02_MPa,
              "von Mises in service (k_tau = 0.5) <= Rp0.2, VDI 2230 R8"),
        _rule("clamp_force", F_res >= j.F_req_N and F_res_alt >= 0, round(F_res, 0), round(j.F_req_N, 0),
              "F_M,min - F_Z - (1 - Phi_n) F_A >= required per bolt (" + j.F_req_basis + ")"
              + ("" if j.load_with_req else "; at the peak working load (separate load case) > 0"),
              safety=round(SG, 2), residual_at_peak_load=round(F_res_alt, 0)),
        _rule("fatigue", SD >= j.SF_fatigue, round(SD, 2) if SD != float("inf") else "inf", j.SF_fatigue,
              f"sigma_ASV {sigma_ASV:.0f} MPa / sigma_a, VDI 2230 R9"),
        _rule("head_bearing", p_max * j.SF_bearing <= pG, round(p_max, 0), round(pG, 0),
              "surface pressure under the head <= p_G (0.9 Rm of the clamped material), VDI 2230 R10"),
    ]
    if j.tapped:
        rules.append(_rule("thread_strip", F_strip >= j.SF_strip * F_break, round(F_strip, 0), round(F_break, 0),
                           f"internal thread shear (m = {j.engagement_mm:g} mm, tau_B = 0.6 Rm) >= bolt breaking "
                           "force As Rm: the bolt breaks first, VDI 2230 R11"))
    return {
        "joint": j.name, "bolt": b.name, "n_bolts": j.n, "torque_Nm": list(j.torque_Nm),
        "mu_G": list(j.mu_G), "mu_K": list(j.mu_K),
        "geometry": dict(d2=round(b.d2, 3), d3=round(b.d3, 3), As=round(b.As, 2), dw=b.head_d, dh=j.hole_d,
                         D_Km=round(dKm, 2), l_K=round(sum(p.t for p in j.plates), 2), gap=j.gap_mm,
                         plates=[asdict(p) for p in j.plates], engagement=j.engagement_mm),
        "compliance": dict(delta_S=dS, delta_P=dP, parts_S=dS_parts, plate=dP_info,
                           k_S_kN_mm=round(1 / dS / 1000, 1), k_P_kN_mm=round(1 / dP / 1000, 1) if dP else None),
        "load_factor": dict(Phi_K=round(phi_K, 3), n=j.load_intro_n, Phi_n=round(phi_n, 3)),
        "preload": dict(FM_min=round(FM_min, 0), FM_max=round(FM_max, 0), alpha_A=round(alpha_A, 2),
                        FM_zul=round(FMzul, 0), fZ_um=fZ_um, FZ=round(FZ, 0), F_residual_min=round(F_res, 0),
                        K_min=round(torque_factor(b, dKm, j.mu_G[0], j.mu_K[0]) / b.d, 3),
                        K_max=round(torque_factor(b, dKm, j.mu_G[1], j.mu_K[1]) / b.d, 3)),
        "loads": dict(FA_static=j.FA_static_N, FA_alt=j.FA_alt_N, F_req=round(j.F_req_N, 0), F_req_basis=j.F_req_basis),
        "stress": dict(sigma_M=round(s_M, 0), tau_M=round(t_M, 0), sigma_red_M=round(s_red_M, 0), nu=round(nu_used, 3),
                       sigma_red_B=round(s_red_B, 0), sigma_a=round(sigma_a, 2), sigma_ASV=round(sigma_ASV, 1),
                       p_head=round(p_max, 0), p_G=round(pG, 0), F_strip=round(F_strip, 0) if j.tapped else None,
                       F_break=round(F_break, 0)),
        "torque_window_Nm": torque_window(j, dS, dP, phi_n, fZ_um),
        "rules": rules,
        "pass": all(r["pass"] for r in rules),
        "failed": [r["check"] for r in rules if not r["pass"]],
    }


def torque_window(j: Joint, dS=None, dP=None, phi_n=None, fZ_um=None):
    """Tightening torque range that satisfies both ends with the friction scatter:
    lowest torque so that F_M,min (mu max) still holds the clamp after embedding, highest
    torque so that F_M,max (mu min) stays within nu Rp0.2. None when the window is empty."""
    b = j.bolt
    if dS is None:
        dS, _ = bolt_compliance(j); dP, _ = plate_compliance(j)
        phi_n = j.load_intro_n * dP / (dS + dP)
        e = EMBED_UM[j.rz_um]
        fZ_um = e["thread"] + e["bearing"] * (1 if j.tapped else 2) + e["inner"] * j.interfaces + j.embed_extra_um
    dKm = (b.head_d + j.hole_d) / 2
    FZ = fZ_um / 1000 / (dS + dP)
    F_need = max(j.F_req_N + FZ + (1 - phi_n) * _FA_with_req(j), FZ + (1 - phi_n) * max(j.FA_static_N, j.FA_alt_N))
    M_lo = F_need * torque_factor(b, dKm, j.mu_G[1], j.mu_K[1]) / 1000
    M_hi = FM_allowed(b, j.mu_G[0], j.nu) * torque_factor(b, dKm, j.mu_G[0], j.mu_K[0]) / 1000
    return dict(min=round(M_lo, 2), max=round(M_hi, 2), feasible=M_lo <= M_hi)
