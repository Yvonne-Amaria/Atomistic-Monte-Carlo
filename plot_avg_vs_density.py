import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Settings ----------------
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

FINAL_RE = re.compile(
    r"FINAL\s+rho\*=(\S+)\s+E\*/N=(\S+)\s+P\*=(\S+)\s+Mu\*_xs=(\S+)\s+Cv\*/N_xs=(\S+)"
)

RHO_DIR_RE = re.compile(r"rho_([0-9.]+)")
RUN_DIR_RE = re.compile(r"run_(\d+)")

# Prefer numeric-only if you have it; otherwise use literature.csv
LIT_CANDIDATES = ["literature_numeric.csv", "literature.csv"]

# columns we will use from literature (ignore any extras like T_star, source)
LIT_COLS = ["rho_star", "E_overN", "P_star", "mu_xs_star", "cv_xs_overN",
            "err_E", "err_P", "err_mu", "err_cv"]


# ---------------- Helpers ----------------
def rho_and_run_from_path(p: Path):
    rho = None
    run = None
    for part in p.parts:
        m = RHO_DIR_RE.fullmatch(part)
        if m:
            rho = float(m.group(1))
        m = RUN_DIR_RE.fullmatch(part)
        if m:
            run = int(m.group(1))
    return rho, run


def parse_final_line(log_path: Path):
    with log_path.open("r") as f:
        for line in f:
            m = FINAL_RE.search(line)
            if m:
                rho = float(m.group(1))
                E = float(m.group(2))
                P = float(m.group(3))
                mu = float(m.group(4))
                cv = float(m.group(5))
                return rho, E, P, mu, cv
    return None


def sem(x):
    x = np.asarray(x, dtype=float)
    if x.size <= 1:
        return np.nan
    return x.std(ddof=1) / np.sqrt(x.size)


def _read_header_names(csv_path: Path):
    header = csv_path.read_text().splitlines()[0].strip()
    return [h.strip() for h in header.split(",")]


def try_load_literature():
    """
    Robust loader for old numpy/matplotlib:
    - Reads only the numeric columns we care about (ignores extra columns like 'source')
    - Forces dtype=float so blanks become np.nan
    - Returns a structured numpy array or None
    """
    lit_path = None
    for name in LIT_CANDIDATES:
        p = Path(name)
        if p.exists():
            lit_path = p
            break
    if lit_path is None:
        return None

    names = _read_header_names(lit_path)
    name_to_idx = {n: i for i, n in enumerate(names)}

    # Only keep columns that exist
    use_names = [c for c in LIT_COLS if c in name_to_idx]
    use_idx = [name_to_idx[c] for c in use_names]

    if "rho_star" not in use_names:
        print("WARNING: literature file missing rho_star; ignoring literature overlay.")
        return None

    lit = np.genfromtxt(
        lit_path,
        delimiter=",",
        names=use_names,
        dtype=float,      # crucial: blank -> nan, no masked/object types
        encoding=None,
        usecols=use_idx
    )
    return lit


def add_literature_overlay(lit, yfield, efield):
    """Overlay literature points if available and finite."""
    if lit is None:
        return False
    if yfield not in lit.dtype.names:
        return False

    # Force float arrays (avoids weird masked/bool behavior on older stacks)
    x = np.asarray(lit["rho_star"], dtype=float)
    y = np.asarray(lit[yfield], dtype=float)

    e = None
    if efield and (efield in lit.dtype.names):
        e = np.asarray(lit[efield], dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    if e is not None:
        mask &= np.isfinite(e)

    if not np.any(mask):
        print(f"INFO: Literature has no finite values for {yfield}; skipping overlay.")
        return False

    plt.errorbar(
        x[mask], y[mask],
        yerr=(e[mask] if e is not None else None),
        marker="s", linestyle="none", capsize=3, label="Literature"
    )
    return True


# ---------------- Main ----------------
def main():
    logs = sorted(Path(".").glob("results/rho_*/run_*/Monte_Carlo.log"))
    if not logs:
        raise SystemExit("No logs found. Expected: results/rho_*/run_*/Monte_Carlo.log")

    by_rho = {}

    for lp in logs:
        rho_path, run = rho_and_run_from_path(lp)
        parsed = parse_final_line(lp)
        if parsed is None:
            print(f"WARNING: No FINAL line found in {lp}")
            continue

        rho_final, E, P, mu, cv = parsed
        rho_use = rho_path if rho_path is not None else rho_final

        if rho_path is not None and abs(rho_path - rho_final) > 1e-6:
            print(f"WARNING: rho mismatch in {lp}: folder rho={rho_path} vs FINAL rho={rho_final}. Using folder rho.")

        by_rho.setdefault(rho_use, {"E": [], "P": [], "mu": [], "cv": [], "files": []})
        by_rho[rho_use]["E"].append(E)
        by_rho[rho_use]["P"].append(P)
        by_rho[rho_use]["mu"].append(mu)
        by_rho[rho_use]["cv"].append(cv)
        by_rho[rho_use]["files"].append(str(lp))

    if not by_rho:
        raise SystemExit("Parsed 0 usable logs (no FINAL lines).")

    rhos = np.array(sorted(by_rho.keys()), dtype=float)

    Emean = np.array([np.mean(by_rho[r]["E"]) for r in rhos], dtype=float)
    Pmean = np.array([np.mean(by_rho[r]["P"]) for r in rhos], dtype=float)
    mumean = np.array([np.mean(by_rho[r]["mu"]) for r in rhos], dtype=float)
    cvmean = np.array([np.mean(by_rho[r]["cv"]) for r in rhos], dtype=float)

    Eerr = np.array([sem(by_rho[r]["E"]) for r in rhos], dtype=float)
    Perr = np.array([sem(by_rho[r]["P"]) for r in rhos], dtype=float)
    muerr = np.array([sem(by_rho[r]["mu"]) for r in rhos], dtype=float)
    cverr = np.array([sem(by_rho[r]["cv"]) for r in rhos], dtype=float)

    lit = try_load_literature()

    def oneplot(y, yerr, ylabel, outname, lit_field=None, lit_err=None):
        plt.figure()
        plt.errorbar(rhos, y, yerr=yerr, marker="o", linestyle="none", capsize=3,
                     label="MC (mean ± SEM)")

        if lit_field is not None:
            add_literature_overlay(lit, lit_field, lit_err)

        plt.xlabel("Reduced density ρ*")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        out = PLOT_DIR / outname
        plt.savefig(out, dpi=200)
        plt.close()
        print("Wrote", out)

    oneplot(Emean, Eerr, "⟨E*⟩/N", "avg_E_vs_rho.png", lit_field="E_overN", lit_err="err_E")
    oneplot(Pmean, Perr, "⟨P*⟩", "avg_P_vs_rho.png", lit_field="P_star", lit_err="err_P")
    oneplot(cvmean, cverr, "⟨Cv,XS*⟩/N", "avg_Cv_vs_rho.png", lit_field="cv_xs_overN", lit_err="err_cv")
    oneplot(mumean, muerr, "⟨μXS*⟩", "avg_mu_vs_rho.png", lit_field="mu_xs_star", lit_err="err_mu")

    # Summary CSV
    out = PLOT_DIR / "avg_summary.csv"
    header = (
        "rho_star,E_overN_mean,E_overN_SEM,"
        "P_star_mean,P_star_SEM,"
        "mu_xs_mean,mu_xs_SEM,"
        "cv_xs_overN_mean,cv_xs_overN_SEM\n"
    )
    with out.open("w") as f:
        f.write(header)
        for i, r in enumerate(rhos):
            f.write(
                f"{r},{Emean[i]},{Eerr[i]},"
                f"{Pmean[i]},{Perr[i]},"
                f"{mumean[i]},{muerr[i]},"
                f"{cvmean[i]},{cverr[i]}\n"
            )
    print("Wrote", out)


if __name__ == "__main__":
    main()
