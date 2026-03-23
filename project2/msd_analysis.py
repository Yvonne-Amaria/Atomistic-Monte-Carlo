"""
msd_analysis.py  –  Project 2, Part 2
========================================
Computes MSD and diffusion coefficients from LAMMPS lammpstrj files.
Supports both:
  (A) Simple analysis (no moving window)   →  standard credit
  (B) Moving-window analysis               →  +5 extra credit

The trajectory must contain UNWRAPPED coordinates (xu, yu, zu).

Usage:
  # Simple:
  python3 msd_analysis.py --traj traj.lammpstrj --dt_frame 0.025 --out simple
  
  # Moving window:
  python3 msd_analysis.py --traj traj.lammpstrj --dt_frame 0.025 --out mw --moving_window

Arguments:
  --traj          Path to lammpstrj file
  --dt_frame      Time between frames in ps  (= timestep_fs * iofrq / 1000)
                  For Part 1: 0.5 fs * 50 steps = 25 fs = 0.025 ps
  --moving_window Use moving window (extra credit)
  --out           Output prefix

Output:
  <out>_msd.dat        columns: time(ps)  MSD(Å²)
  <out>_Dcoeff.txt     diffusion coefficient in m²/s
"""

import os, sys, argparse
import numpy as np
from scipy.stats import linregress


# ────────────────────────────────────────────────────────────
#  Trajectory reader  (adapts the get_dist logic from RDF code)
# ────────────────────────────────────────────────────────────

def read_trajectory(traj_path, max_frames=None):
    """
    Read a LAMMPS lammpstrj with fields: id type element xu yu zu [vx vy vz]
    Returns:
        frames  : list of (natoms, 3) arrays  [Å, UNWRAPPED]
        box_arr : (nframes, 3) box dimensions
        atom_ids: (natoms,) sorted atom ids from frame 0
    """
    frames   = []
    box_arr  = []
    atom_ids = None

    with open(traj_path) as f:
        while True:
            # ITEM: TIMESTEP
            line = f.readline()
            if not line:
                break
            if 'TIMESTEP' not in line:
                continue
            f.readline()           # timestep value

            f.readline()           # ITEM: NUMBER OF ATOMS
            natoms = int(f.readline())

            f.readline()           # ITEM: BOX BOUNDS
            bx = float(f.readline().split()[1])
            by = float(f.readline().split()[1])
            bz = float(f.readline().split()[1])
            box_arr.append([bx, by, bz])

            f.readline()           # ITEM: ATOMS ...
            coords = np.zeros((natoms, 3))
            ids    = np.zeros(natoms, dtype=int)
            for i in range(natoms):
                tok = f.readline().split()
                ids[i]      = int(tok[0])
                coords[i, 0] = float(tok[3])   # xu
                coords[i, 1] = float(tok[4])   # yu
                coords[i, 2] = float(tok[5])   # zu

            # Sort by atom id so indexing is consistent across frames
            sort_idx = np.argsort(ids)
            coords   = coords[sort_idx]
            ids      = ids[sort_idx]

            if atom_ids is None:
                atom_ids = ids

            frames.append(coords)

            if max_frames and len(frames) >= max_frames:
                break

    return np.array(frames), np.array(box_arr), atom_ids


# ────────────────────────────────────────────────────────────
#  MSD  (simple, from frame 0)
# ────────────────────────────────────────────────────────────

def compute_msd_simple(frames):
    """
    MSD(t) = <|r(t) - r(0)|²>  averaged over all atoms.
    Frames must use UNWRAPPED coords.
    Returns: msd array of length nframes.
    """
    r0   = frames[0]                    # (natoms, 3)
    msd  = np.zeros(len(frames))
    for t, r in enumerate(frames):
        dr      = r - r0                # (natoms, 3)
        msd[t]  = np.mean(np.sum(dr**2, axis=1))
    return msd


# ────────────────────────────────────────────────────────────
#  MSD  (moving window)
# ────────────────────────────────────────────────────────────

def compute_msd_moving_window(frames):
    """
    Moving-window (time-averaged) MSD:
      MSD(Δt) = < <|r(t+Δt) - r(t)|²>_atoms >_t

    Better statistics, but answers will differ from simple MSD.
    Returns: msd array of length nframes.
    """
    nframes, natoms, _ = frames.shape
    msd = np.zeros(nframes)
    count = np.zeros(nframes, dtype=int)

    for t0 in range(nframes):
        for dt in range(nframes - t0):
            dr = frames[t0 + dt] - frames[t0]   # (natoms, 3)
            msd[dt]   += np.mean(np.sum(dr**2, axis=1))
            count[dt] += 1

    # avoid division by zero at dt=0 (count should always be >0)
    valid = count > 0
    msd[valid] /= count[valid]
    return msd


# ────────────────────────────────────────────────────────────
#  Diffusion coefficient from Einstein relation
# ────────────────────────────────────────────────────────────

def fit_diffusion(times_ps, msd_Asq, fit_start_frac=0.2, fit_end_frac=0.8):
    """
    D = MSD / (6 t)  in the linear regime.
    Fit MSD = 6 D t + b  over the specified fraction of the data.

    Returns D in m²/s.
    """
    n = len(times_ps)
    i0 = int(n * fit_start_frac)
    i1 = int(n * fit_end_frac)
    if i1 <= i0 + 2:
        i0, i1 = 1, n - 1          # fallback: use all but first point

    slope, intercept, r, p, se = linregress(times_ps[i0:i1], msd_Asq[i0:i1])

    # slope has units Å²/ps
    # D = slope/6  [Å²/ps]  → convert to m²/s
    # 1 Å² = 1e-20 m²;  1 ps = 1e-12 s  →  1 Å²/ps = 1e-8 m²/s
    D_m2s = (slope / 6.0) * 1e-8
    return D_m2s, slope, intercept, r**2, times_ps[i0:i1], msd_Asq[i0:i1]


# ────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj',         required=True, help='Path to lammpstrj file')
    parser.add_argument('--dt_frame',     type=float, default=0.025,
                        help='Time between trajectory frames in ps (default: 0.025 ps = 0.5fs*50)')
    parser.add_argument('--moving_window', action='store_true',
                        help='Use moving-window MSD (extra credit)')
    parser.add_argument('--max_frames',   type=int, default=None,
                        help='Limit frames read (for testing)')
    parser.add_argument('--out',          default='msd', help='Output file prefix')
    parser.add_argument('--fit_start',    type=float, default=0.20,
                        help='Fraction of trajectory to start linear fit (default 0.2)')
    parser.add_argument('--fit_end',      type=float, default=0.80,
                        help='Fraction of trajectory to end linear fit (default 0.8)')
    args = parser.parse_args()

    print(f"Reading trajectory: {args.traj}")
    frames, boxes, _ = read_trajectory(args.traj, max_frames=args.max_frames)
    nframes = len(frames)
    print(f"  Loaded {nframes} frames, {frames.shape[1]} atoms")

    times = np.arange(nframes) * args.dt_frame   # ps

    if args.moving_window:
        print("  Computing MSD with moving-window averaging...")
        msd = compute_msd_moving_window(frames)
        method = 'moving_window'
    else:
        print("  Computing MSD (simple, from frame 0)...")
        msd = compute_msd_simple(frames)
        method = 'simple'

    # Save MSD data
    msd_file = f'{args.out}_msd.dat'
    np.savetxt(msd_file, np.column_stack([times, msd]),
               header='time(ps)  MSD(Ang^2)',
               fmt='%.6e')
    print(f"  Saved {msd_file}")

    # Fit diffusion coefficient
    D, slope, intercept, r2, t_fit, msd_fit = fit_diffusion(
        times, msd, args.fit_start, args.fit_end)

    d_file = f'{args.out}_Dcoeff.txt'
    with open(d_file, 'w') as f:
        f.write(f"# Diffusion coefficient analysis\n")
        f.write(f"# Method: {method}\n")
        f.write(f"# Trajectory: {args.traj}\n")
        f.write(f"# dt_frame: {args.dt_frame} ps\n")
        f.write(f"# Linear fit range: {args.fit_start*100:.0f}%–{args.fit_end*100:.0f}% of trajectory\n")
        f.write(f"# Slope (Ang^2/ps): {slope:.6f}\n")
        f.write(f"# R²: {r2:.6f}\n")
        f.write(f"Diffusion coefficient D = {D:.6e} m^2/s\n")
    print(f"  D = {D:.4e} m²/s   (R² = {r2:.4f})")
    print(f"  Saved {d_file}")

    # Plot MSD
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(times, msd, color='steelblue', lw=1.2, label='MSD')
    ax.plot(t_fit, slope * t_fit + intercept, 'r--', lw=1.5,
            label=f'Linear fit  D={D:.2e} m²/s')
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('MSD (Å²)', fontsize=12)
    ax.set_title(f'Mean Square Displacement  [{method}]', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{args.out}_msd_plot.png', dpi=150)
    plt.close(fig)
    print(f"  Saved {args.out}_msd_plot.png")


# ── Batch convenience: run over all densities ─────────────────────────────

def batch_all_densities(results_dir='results/part1', dt_frame=0.025,
                        moving_window=False, out_dir='plots/msd'):
    """
    Convenience function: loop over all rho_* directories and compute D for each.
    """
    import glob
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    SIGMA = 3.4

    densities, D_vals = [], []

    fig_msd, ax_msd = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, 9))
    ci = 0

    for rho_dir in sorted(glob.glob(os.path.join(results_dir, 'rho_*'))):
        rho_str = os.path.basename(rho_dir).replace('rho_', '')
        try:
            rho = float(rho_str)
        except ValueError:
            continue

        traj = os.path.join(rho_dir, 'nproc_1', 'traj.lammpstrj')
        if not os.path.exists(traj):
            print(f"  Skipping rho*={rho}: no trajectory found")
            continue

        print(f"\n  Processing rho*={rho}")
        frames, boxes, _ = read_trajectory(traj)
        times = np.arange(len(frames)) * dt_frame

        if moving_window:
            msd = compute_msd_moving_window(frames)
        else:
            msd = compute_msd_simple(frames)

        D, slope, intercept, r2, t_fit, _ = fit_diffusion(times, msd)
        densities.append(rho)
        D_vals.append(D)

        ax_msd.plot(times, msd, color=colors[ci], lw=1.0, label=f'ρ*={rho}')
        ci += 1
        print(f"    D = {D:.4e} m²/s")

    ax_msd.set_xlabel('Time (ps)', fontsize=12)
    ax_msd.set_ylabel('MSD (Å²)', fontsize=12)
    ax_msd.set_title('MSD for all densities', fontsize=13)
    ax_msd.legend(fontsize=7, ncol=3)
    ax_msd.grid(True, alpha=0.3)
    fig_msd.tight_layout()
    fig_msd.savefig(os.path.join(out_dir, 'msd_all_densities.png'), dpi=150)
    plt.close(fig_msd)

    # Diffusion vs density plot
    fig_d, ax_d = plt.subplots(figsize=(7, 4.5))
    ax_d.semilogy(densities, D_vals, 'o-', color='royalblue', lw=1.5, ms=7)
    ax_d.set_xlabel('Reduced density ρ*', fontsize=12)
    ax_d.set_ylabel('D  (m²/s)', fontsize=12)
    ax_d.set_title('Diffusion Coefficient vs Density', fontsize=13)
    ax_d.grid(True, alpha=0.3, which='both')
    fig_d.tight_layout()
    fig_d.savefig(os.path.join(out_dir, 'diffusion_vs_density.png'), dpi=150)
    plt.close(fig_d)

    print(f"\nDone. Plots saved to {out_dir}/")
    return densities, D_vals


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    main()
