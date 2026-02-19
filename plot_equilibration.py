import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

# --------- Parsing helpers ---------
STEP_RE = re.compile(r"^\s*Step:\s*(\d+)\s+.*?E\*/N:\s*([-\d\.Ee+]+)\s+P\*:\s*([-\d\.Ee+]+)")

RHO_DIR_RE = re.compile(r"rho_([0-9.]+)")
RUN_DIR_RE = re.compile(r"run_(\d+)")

def parse_log_steps(log_path: Path):
    steps, estar, pstar = [], [], []
    with log_path.open("r") as f:
        for line in f:
            m = STEP_RE.match(line)
            if m:
                steps.append(int(m.group(1)))
                estar.append(float(m.group(2)))
                pstar.append(float(m.group(3)))
    return np.array(steps), np.array(estar), np.array(pstar)

def rho_and_run_from_path(p: Path):
    """Extract rho* and run index from .../results/rho_0.1/run_1/Monte_Carlo.log"""
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

# --------- Main ---------
def main():
    # Run from Atomistic-Monte-Carlo (repo root)
    logs = sorted(Path(".").glob("results/rho_*/run_*/Monte_Carlo.log"))
    if not logs:
        raise SystemExit("No logs found. Expected: results/rho_*/run_*/Monte_Carlo.log")

    # group by density
    by_rho = {}
    for lp in logs:
        rho, run = rho_and_run_from_path(lp)
        if rho is None:
            print(f"WARNING: Could not parse rho from path: {lp}")
            continue
        by_rho.setdefault(rho, []).append(lp)

    # Make two zoom levels per property
    for prop in ["E", "P"]:
        for zoom in ["first1e6", "last1e4"]:
            plt.figure()

            for rho in sorted(by_rho.keys()):
                for lp in sorted(by_rho[rho]):
                    steps, estar, pstar = parse_log_steps(lp)
                    if steps.size == 0:
                        print(f"WARNING: No Step lines parsed in {lp}")
                        continue

                    if zoom == "first1e6":
                        mask = steps <= 1_000_000
                    else:
                        maxs = steps.max()
                        mask = steps >= (maxs - 10_000)

                    y = estar if prop == "E" else pstar
                    _, run = rho_and_run_from_path(lp)
                    run_label = f"run_{run}" if run is not None else lp.parent.name

                    plt.plot(
                        steps[mask], y[mask], linewidth=1.0,
                        label=f"rho*={rho:g} ({run_label})"
                    )

            plt.xlabel("MC step")
            plt.ylabel("E*/N" if prop == "E" else "P*")
            plt.title(f"Equilibration: {'E*/N' if prop=='E' else 'P*'} ({zoom})")

            # Legend outside (can get huge)
            plt.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()

            out = f"equil_{'E' if prop=='E' else 'P'}_{zoom}.png"
            plt.savefig(PLOT_DIR / out, dpi=200)
            plt.close()
            print("Wrote", PLOT_DIR / out)

if __name__ == "__main__":
    main()
