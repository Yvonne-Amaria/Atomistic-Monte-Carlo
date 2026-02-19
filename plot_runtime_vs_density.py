import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

RHO_DIR_RE = re.compile(r"rho_([0-9.]+)")
RUN_DIR_RE = re.compile(r"run_(\d+)")

def rho_and_run_from_path(p: Path):
    """Extract rho* and run index from .../results/rho_0.1/run_1/<file>"""
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

def parse_time_file(path: Path):
    """
    Parse output from /usr/bin/time -o <file>.
    Supports:
      - default verbose format containing '... 0:00.13elapsed ...'
      - GNU time -p format with 'real 0.13'
    """
    txt = path.read_text()

    # Look for "... M:SS.xxxelapsed"
    m = re.search(r"(\d+):(\d+)\.(\d+)elapsed", txt)
    if m:
        minutes = int(m.group(1))
        seconds = int(m.group(2))
        frac = int(m.group(3)) / (10 ** len(m.group(3)))
        return 60 * minutes + seconds + frac

    # GNU time -p: "real X.Y"
    m = re.search(r"real\s+([0-9.]+)", txt)
    if m:
        return float(m.group(1))

    raise ValueError(f"Could not parse elapsed time from {path}")

def sem(x):
    x = np.asarray(x, float)
    if x.size <= 1:
        return np.nan
    return x.std(ddof=1) / np.sqrt(x.size)

def main():
    # Search timing files under each run directory.
    # This matches: results/rho_*/run_*/<anything with 'timing' in name>.dat
    candidates = sorted(Path(".").glob("results/rho_*/run_*/*timing*.dat"))
    if not candidates:
        # also try case-insensitive by scanning .dat files in run folders
        fallback = []
        for p in Path(".").glob("results/rho_*/run_*/*.dat"):
            if "timing" in p.name.lower():
                fallback.append(p)
        candidates = sorted(fallback)

    if not candidates:
        raise SystemExit(
            "No timing files found.\n"
            "Expected something like: results/rho_0.1/run_1/*timing*.dat\n"
            "If your timing files have a different name, tell me the exact filename."
        )

    by_rho = {}
    for fp in candidates:
        rho, run = rho_and_run_from_path(fp)
        if rho is None:
            print(f"WARNING: Could not parse rho from path: {fp}")
            continue
        try:
            t = parse_time_file(fp)
        except Exception as e:
            print(f"WARNING: Skipping {fp} ({e})")
            continue
        by_rho.setdefault(rho, []).append(t)

    if not by_rho:
        raise SystemExit("Found timing files, but parsed 0 usable runtimes.")

    rhos = np.array(sorted(by_rho.keys()))
    tmean = np.array([np.mean(by_rho[r]) for r in rhos])
    terr  = np.array([sem(by_rho[r]) for r in rhos])

    plt.figure()
    plt.errorbar(rhos, tmean, yerr=terr, marker="o", linestyle="none", capsize=3)
    plt.xlabel("Reduced density ρ*")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs density (mean ± SEM across runs)")
    plt.tight_layout()

    out = PLOT_DIR / "runtime_vs_rho.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("Wrote", out)

if __name__ == "__main__":
    main()
