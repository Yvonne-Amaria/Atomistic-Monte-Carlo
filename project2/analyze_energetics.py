"""
analyze_energetics.py  –  Project 2, Part 1 (items 1 & 2)
=============================================================
Parses LAMMPS thermo output from LJ.out files and produces:
  - 4 energy plots  (pe, ke, etotal, econserve)  vs timestep
  - 1 pressure plot  (avg P*)  vs density

Directory structure expected:
  results/part1/rho_<rho>/nproc_1/LJ.out

Usage:
  python analyze_energetics.py [--results_dir results/part1]
"""

import os, re, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Argon LJ parameters ────────────────────────────────────────────────────
SIGMA   = 3.4        # Angstrom
EPS_KB  = 120.0      # K  (epsilon/kB)
kB_kcal = 1.987204e-3  # kcal/(mol*K)
EPS_kcal = EPS_KB * kB_kcal   # kcal/mol  (~0.2385)
MASS    = 39.948     # g/mol
T_K     = 144.0      # K

# Reduced temperature
T_STAR  = T_K / EPS_KB  # = 1.2


def parse_thermo(log_path):
    """
    Parse the LAMMPS log file and return a dict of column arrays.
    Handles multiple 'thermo' blocks (takes the last production run).
    """
    header = None
    blocks = []
    current = []

    with open(log_path) as f:
        for line in f:
            line = line.rstrip()

            # detect header line (starts with 'Step', possibly with leading whitespace)
            if re.match(r'^\s*Step\s', line):
                if current:
                    blocks.append((header, current))
                    current = []
                header = line.split()
                continue

            # numeric data row
            if header is not None:
                parts = line.split()
                if len(parts) == len(header):
                    try:
                        current.append([float(x) for x in parts])
                    except ValueError:
                        pass

    if current:
        blocks.append((header, current))

    if not blocks:
        raise ValueError(f"No thermo data found in {log_path}")

    # use the longest block (the production run)
    header, rows = max(blocks, key=lambda b: len(b[1]))
    data = np.array(rows)
    return {col: data[:, i] for i, col in enumerate(header)}


def reduced_energy(e_kcal_per_atom, eps=EPS_kcal):
    """Convert kcal/mol/atom → E* = E/(N ε)"""
    return e_kcal_per_atom / eps


def reduced_pressure(p_atm, rho_star):
    """
    LAMMPS 'real' units: pressure in atm.
    Convert to P* = P σ³ / ε using:
      P [atm] * (1 atm = 101325 Pa)
      σ [Å] * (1 Å = 1e-10 m)  →  σ³ [m³]
      ε [kcal/mol] * (4184 J/kcal / 6.022e23) [J/molecule]
    """
    Pa_per_atm = 101325.0
    m3_per_A3  = 1e-30
    J_per_kcal_mol = 4184.0 / 6.022e23

    eps_J = EPS_kcal * J_per_kcal_mol
    sig3_m3 = (SIGMA * 1e-10)**3

    P_Pa = p_atm * Pa_per_atm
    return P_Pa * sig3_m3 / eps_J


def find_equilibration_step(steps, energies, window=500):
    """
    Estimate equilibration: point where the running mean first
    stays within ±2% of the final mean for one window length.
    Returns the step index (not step number).
    """
    final_mean = np.mean(energies[-window:])
    if final_mean == 0:
        return 0
    for i in range(len(energies) - window):
        chunk = energies[i:i+window]
        if np.all(np.abs(chunk - final_mean) / abs(final_mean) < 0.02):
            return i
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='results/part1',
                        help='Root dir with rho_*/nproc_1/LJ.out files')
    parser.add_argument('--nequil_frac', type=float, default=0.2,
                        help='Fraction of run used for equilibration (for avg pressure)')
    args = parser.parse_args()

    # ── Collect all densities ───────────────────────────────────────────────
    densities = []
    data_by_rho = {}

    for entry in sorted(os.listdir(args.results_dir)):
        if not entry.startswith('rho_'):
            continue
        rho_str = entry.replace('rho_', '')
        try:
            rho = float(rho_str)
        except ValueError:
            continue

        log = os.path.join(args.results_dir, entry, 'nproc_1', 'LJ.out')
        if not os.path.exists(log):
            print(f"  WARNING: {log} not found – skipping")
            continue

        print(f"  Parsing rho*={rho}  ({log})")
        thermo = parse_thermo(log)
        data_by_rho[rho] = thermo
        densities.append(rho)

    if not densities:
        print("ERROR: No LJ.out files found. Check --results_dir path.")
        return

    densities = sorted(densities)
    nrho = len(densities)
    colors = cm.viridis(np.linspace(0.1, 0.9, nrho))

    os.makedirs('plots', exist_ok=True)

    # ── Item 1: Four energy plots ───────────────────────────────────────────
    energy_cols = {
        'PotEng': ('Potential Energy',     'E_{pot}'),
        'KinEng': ('Kinetic Energy',       'E_{kin}'),
        'TotEng': ('Total Energy',         'E_{tot}'),
        'Econserve': ('Conserved Quantity','E_{conserve}'),
    }

    for col, (title, label) in energy_cols.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        for rho, c in zip(densities, colors):
            thermo = data_by_rho[rho]
            # find the right column name (LAMMPS capitalises variably)
            ckey = next((k for k in thermo if k.lower() == col.lower()), None)
            if ckey is None:
                print(f"  Column '{col}' not found for rho*={rho}; available: {list(thermo.keys())}")
                continue
            steps = thermo['Step']
            e_star = reduced_energy(thermo[ckey])
            ax.plot(steps, e_star, color=c, lw=0.8, label=f'ρ*={rho}')

        ax.set_xlabel('Timestep', fontsize=12)
        ax.set_ylabel(f'${label}^*$ / atom', fontsize=12)
        ax.set_title(f'{title} vs Timestep  (T*={T_STAR:.2f})', fontsize=13)
        ax.legend(fontsize=7, ncol=3, loc='upper right')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = f'plots/energy_{col.lower()}.png'
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}")

    # ── Item 2: Pressure vs density ─────────────────────────────────────────
    rho_vals, pstar_md = [], []
    for rho in densities:
        thermo = data_by_rho[rho]
        steps  = thermo['Step']
        n_equil = int(len(steps) * args.nequil_frac)

        pcol = next((k for k in thermo if k.lower() == 'press'), None)
        if pcol is None:
            print(f"  WARNING: 'Press' column not found for rho*={rho}")
            continue

        p_equil = thermo[pcol][n_equil:]   # post-equil
        p_avg   = np.mean(p_equil)
        pstar   = reduced_pressure(p_avg, rho)
        rho_vals.append(rho)
        pstar_md.append(pstar)
        print(f"  rho*={rho:.1f}  <P*>_MD = {pstar:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rho_vals, pstar_md, 'o-', color='royalblue', lw=1.5,
            ms=6, label='MD (this work)')

    # ---- Optional: overlay MC data if available ----------------------------
    mc_file = 'mc_pressure.dat'   # columns: rho* P*
    if os.path.exists(mc_file):
        mc = np.loadtxt(mc_file)
        ax.plot(mc[:, 0], mc[:, 1], 's--', color='tomato', lw=1.5,
                ms=6, label='MC (Project 1)')

    ax.axhline(0, color='gray', lw=0.7, ls=':')
    ax.set_xlabel('Reduced density ρ*', fontsize=12)
    ax.set_ylabel('Reduced pressure P*', fontsize=12)
    ax.set_title(f'P* vs ρ*  (T* = {T_STAR:.2f},  NVT MD)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('plots/pressure_vs_density.png', dpi=150)
    plt.close(fig)
    print("  Saved plots/pressure_vs_density.png")

    # ── Equilibration estimate ──────────────────────────────────────────────
    print("\n── Equilibration estimates ──")
    for rho in densities:
        thermo = data_by_rho[rho]
        ecol = next((k for k in thermo if k.lower() == 'toteng'), None)
        if ecol is None:
            continue
        idx = find_equilibration_step(thermo['Step'], thermo[ecol])
        step = int(thermo['Step'][idx])
        print(f"  rho*={rho:.1f}  ≈ equilibrated by step {step}")

    print("\nDone. Plots written to ./plots/")


if __name__ == '__main__':
    main()
