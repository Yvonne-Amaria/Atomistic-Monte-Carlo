import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

RHO_DIR_RE = re.compile(r"rho_([0-9.]+)")
RUN_DIR_RE = re.compile(r"run_(\d+)")

def rho_and_run_from_path(p: Path):
    """Extract rho* and run index from .../results/rho_0.1/run_1/rdf.dat"""
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

def load_rdf(path: Path):
    """Load rdf.dat with columns: r, nint, g(r)."""
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"{path} does not have at least 3 columns.")
    r = data[:, 0]
    nint = data[:, 1]
    gr = data[:, 2]
    return r, nint, gr

def main():
    # Expected output from your updated C++: results/rho_*/run_*/rdf.dat
    files = sorted(Path(".").glob("results/rho_*/run_*/rdf.dat"))
    if not files:
        raise SystemExit(
            "No RDF files found.\n"
            "Expected: results/rho_*/run_*/rdf.dat\n"
            "Make sure your RDF code writes rdf.dat into each run folder."
        )

    # Group by density, store run files
    by_rho = {}
    for fp in files:
        rho, run = rho_and_run_from_path(fp)
        if rho is None:
            print(f"WARNING: Could not parse rho from path: {fp}")
            continue
        by_rho.setdefault(rho, []).append(fp)

    if not by_rho:
        raise SystemExit("Found rdf.dat files, but parsed 0 usable densities from folder names.")

    # For each density: average g(r) and N(r) across runs.
    # Assumes all runs at the same rho use the same r-grid (same binw, same box length).
    rho_list = sorted(by_rho.keys())

    # --- Plot averaged g(r) for all densities ---
    plt.figure()
    for rho in rho_list:
        run_files = sorted(by_rho[rho])

        r0, n0, g0 = load_rdf(run_files[0])
        g_stack = [g0]
        n_stack = [n0]

        # Load additional runs and check r grid compatibility
        for rf in run_files[1:]:
            r, nint, gr = load_rdf(rf)

            if len(r) != len(r0) or np.max(np.abs(r - r0)) > 1e-9:
                print(f"WARNING: r-grid mismatch at rho={rho} for file {rf}. Skipping this run.")
                continue

            g_stack.append(gr)
            n_stack.append(nint)

        g_stack = np.vstack(g_stack)
        g_mean = np.mean(g_stack, axis=0)

        plt.plot(r0, g_mean, linewidth=1.5, label=f"rho*={rho:g} (avg over {g_stack.shape[0]} runs)")

    plt.xlabel("r (Å)")
    plt.ylabel("g(r)")
    plt.title("Radial distribution function (averaged over runs)")
    plt.legend()
    plt.tight_layout()
    out1 = PLOT_DIR / "rdf_all_densities.png"
    plt.savefig(out1, dpi=200)
    plt.close()
    print("Wrote", out1)

    # --- Plot averaged number integral for all densities ---
    plt.figure()
    for rho in rho_list:
        run_files = sorted(by_rho[rho])

        r0, n0, g0 = load_rdf(run_files[0])
        n_stack = [n0]

        for rf in run_files[1:]:
            r, nint, gr = load_rdf(rf)
            if len(r) != len(r0) or np.max(np.abs(r - r0)) > 1e-9:
                continue
            n_stack.append(nint)

        n_stack = np.vstack(n_stack)
        n_mean = np.mean(n_stack, axis=0)

        plt.plot(r0, n_mean, linewidth=1.5, label=f"rho*={rho:g} (avg over {n_stack.shape[0]} runs)")

    plt.xlabel("r (Å)")
    plt.ylabel("Number integral N(r)")
    plt.title("Number integral (averaged over runs)")
    plt.legend()
    plt.tight_layout()
    out2 = PLOT_DIR / "nint_all_densities.png"
    plt.savefig(out2, dpi=200)
    plt.close()
    print("Wrote", out2)

if __name__ == "__main__":
    main()
