"""
plot_scaling.py  –  Project 2, Part 3
=======================================
Produces three plots from timing data collected on Great Lakes:

  1. System-size scaling: seconds per 1M steps vs number of atoms (serial)
  2. Strong scaling vs density: Msteps/hr for 1, 2, 4 procs, all densities
  3. Strong scaling vs nproc:  Msteps/hr vs nproc for nrepl=8

Input files (auto-discovered from results/ tree):
  results/syssize/nrepl_N/wall_time_sec.txt
  results/part2a/rho_R/nproc_P/LJ.out
  results/part2b/nrepl_8/nproc_P/wall_time_sec.txt

"""

import os, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

NSTEPS = 1_000_000   # steps used for Part 3 runs


def read_wall_time(path):
    """Read wall time in seconds from wall_time_sec.txt"""
    with open(path) as f:
        return float(f.read().strip())


def parse_walltime_from_lammps(log_path):
    """Parse 'Total wall time' line from a LAMMPS .out file."""
    with open(log_path) as f:
        for line in f:
            if 'Total wall time' in line:
                parts = line.split(':')
                # format: 0:hh:mm:ss  or  hh:mm:ss
                try:
                    h, m, s = int(parts[-3]), int(parts[-2]), int(parts[-1])
                    return h*3600 + m*60 + s
                except Exception:
                    pass
    return None


def msteps_per_hour(wall_s, nsteps=NSTEPS):
    """Convert wall time in seconds to million steps per hour."""
    if wall_s is None or wall_s == 0:
        return None
    return nsteps / wall_s * 3600 / 1e6


# ─────────────────────────────────────────────────────────────────
#  1.  System-size scaling
# ─────────────────────────────────────────────────────────────────

def plot_syssize(results_dir):
    pattern = os.path.join(results_dir, 'syssize', 'nrepl_*', 'wall_time_sec.txt')
    files = sorted(glob.glob(pattern))
    if not files:
        print("  [syssize] No data found – skipping plot 1")
        return

    natoms_list, secs_list = [], []
    for fpath in files:
        nrepl = int(os.path.basename(os.path.dirname(fpath)).replace('nrepl_', ''))
        natoms = 4 * nrepl**3
        wall_s = read_wall_time(fpath)
        secs_per_Mstep = wall_s   # run was 1M steps
        natoms_list.append(natoms)
        secs_list.append(secs_per_Mstep)
        print(f"  syssize: natoms={natoms}, wall={wall_s:.1f}s, {secs_per_Mstep:.1f} s/Mstep")

    natoms = np.array(natoms_list)
    secs   = np.array(secs_list)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(natoms, secs, 'o-', color='royalblue', lw=1.8, ms=7,
            label='Measured')

    # Fit power law: log(t) = a*log(N) + b
    coeffs = np.polyfit(np.log(natoms), np.log(secs), 1)
    exp = coeffs[0]
    N_fit = np.linspace(natoms[0]*0.9, natoms[-1]*1.1, 100)
    t_fit = np.exp(np.polyval(coeffs, np.log(N_fit)))
    ax.plot(N_fit, t_fit, '--', color='gray', lw=1.2,
            label=f'Power-law fit  (t ∝ N^{exp:.2f})')

    ax.set_xlabel('Number of atoms', fontsize=12)
    ax.set_ylabel('Wall time (s) per 10⁶ steps', fontsize=12)
    ax.set_title('System-size Scaling  (1 processor, ρ*=0.9)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    fig.tight_layout()
    fig.savefig('plots/scaling_syssize.png', dpi=150)
    plt.close(fig)
    print("  Saved plots/scaling_syssize.png")


# ─────────────────────────────────────────────────────────────────
#  2.  Strong scaling vs density (1, 2, 4 procs)
# ─────────────────────────────────────────────────────────────────

def plot_scaling_density(results_dir):
    nproc_list  = [1, 2, 4]
    colors      = ['royalblue', 'seagreen', 'tomato']
    markers     = ['o', 's', '^']
    densities   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    fig, ax = plt.subplots(figsize=(8, 5))
    any_data = False

    for nproc, col, mk in zip(nproc_list, colors, markers):
        rho_vals, perf_vals = [], []
        for rho in densities:
            rho_str = f'{rho:.1f}'
            if nproc == 1:
                log = os.path.join(results_dir, 'part1',
                                   f'rho_{rho_str}', 'nproc_1', 'LJ.out')
            else:
                log = os.path.join(results_dir, 'part2a',
                                   f'rho_{rho_str}', f'nproc_{nproc}', 'LJ.out')

            if not os.path.exists(log):
                # try wall_time_sec.txt
                wt = log.replace('LJ.out', 'wall_time_sec.txt')
                if os.path.exists(wt):
                    wall_s = read_wall_time(wt)
                else:
                    continue
            else:
                wall_s = parse_walltime_from_lammps(log)
                if wall_s is None:
                    continue

            perf = msteps_per_hour(wall_s)
            if perf:
                rho_vals.append(rho)
                perf_vals.append(perf)

        if rho_vals:
            ax.plot(rho_vals, perf_vals, marker=mk, color=col,
                    lw=1.5, ms=7, label=f'{nproc} processor(s)')
            any_data = True

    if not any_data:
        print("  [density scaling] No data found – skipping plot 2")
        plt.close(fig)
        return

    ax.set_xlabel('Reduced density ρ*', fontsize=12)
    ax.set_ylabel('Performance (Msteps/hr)', fontsize=12)
    ax.set_title('Strong Scaling vs Density  (nrepl=5, 500 atoms)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('plots/scaling_vs_density.png', dpi=150)
    plt.close(fig)
    print("  Saved plots/scaling_vs_density.png")


# ─────────────────────────────────────────────────────────────────
#  3.  Strong scaling vs nproc  (nrepl=8)
# ─────────────────────────────────────────────────────────────────

def plot_strong_scaling(results_dir):
    nproc_list = [1, 2, 4, 8, 16, 36, 72]
    nproc_vals, perf_vals = [], []

    for nproc in nproc_list:
        base = os.path.join(results_dir, 'part2b', 'nrepl_8', f'nproc_{nproc}')
        wt_file = os.path.join(base, 'wall_time_sec.txt')
        log     = os.path.join(base, 'LJ.out')

        if os.path.exists(wt_file):
            wall_s = read_wall_time(wt_file)
        elif os.path.exists(log):
            wall_s = parse_walltime_from_lammps(log)
        else:
            continue

        if wall_s:
            perf = msteps_per_hour(wall_s)
            if perf:
                nproc_vals.append(nproc)
                perf_vals.append(perf)
                print(f"  nproc={nproc}: {wall_s:.1f}s → {perf:.3f} Msteps/hr")

    if not nproc_vals:
        print("  [strong scaling] No data found – skipping plot 3")
        return

    nproc_arr = np.array(nproc_vals)
    perf_arr  = np.array(perf_vals)

    # Ideal scaling from 1-proc measurement (or first available)
    perf_1 = perf_arr[0]
    nproc_1 = nproc_arr[0]
    ideal   = perf_1 * nproc_arr / nproc_1

    # Efficiency
    efficiency = perf_arr / ideal

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Performance
    ax1.loglog(nproc_arr, perf_arr, 'o-', color='royalblue', lw=1.8, ms=7,
               label='Measured')
    ax1.loglog(nproc_arr, ideal, '--', color='gray', lw=1.2, label='Ideal')
    ax1.set_xlabel('Number of processors', fontsize=12)
    ax1.set_ylabel('Performance (Msteps/hr)', fontsize=12)
    ax1.set_title('Strong Scaling  (nrepl=8, 2048 atoms, ρ*=0.9)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')

    # Efficiency
    ax2.semilogx(nproc_arr, efficiency, 'o-', color='seagreen', lw=1.8, ms=7)
    ax2.axhline(0.80, color='royalblue', ls='--', lw=1.2, label='80% threshold')
    ax2.set_xlabel('Number of processors', fontsize=12)
    ax2.set_ylabel('Parallel efficiency', fontsize=12)
    ax2.set_title('Parallel Efficiency', fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig('plots/scaling_strong_nproc.png', dpi=150)
    plt.close(fig)
    print("  Saved plots/scaling_strong_nproc.png")

    # Print efficiency table
    print("\n  Strong Scaling Summary (nrepl=8):")
    print(f"  {'nproc':>8}  {'Measured':>12}  {'Ideal':>12}  {'Efficiency':>12}")
    for n, p, i, e in zip(nproc_arr, perf_arr, ideal, efficiency):
        print(f"  {int(n):>8}  {p:>12.3f}  {i:>12.3f}  {e:>12.3f}")


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='results',
                        help='Root results directory')
    args = parser.parse_args()

    os.makedirs('plots', exist_ok=True)

    print("=== Plot 1: System-size scaling ===")
    plot_syssize(args.results_dir)

    print("\n=== Plot 2: Scaling vs density ===")
    plot_scaling_density(args.results_dir)

    print("\n=== Plot 3: Strong scaling vs nproc ===")
    plot_strong_scaling(args.results_dir)

    print("\nAll scaling plots done.")


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    main()
