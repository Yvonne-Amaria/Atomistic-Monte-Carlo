"""
plot_rdf.py  –  Project 2, Part 1, Item 3
===========================================
Plots RDF for each density, overlaying MD and MC 

Expects:
  MD RDF files: results/part1/rho_<rho>/nproc_1/rdf.dat
                OR Travis output:  msd_Ar_#2.csv  (converted below)
  MC RDF files: mc_data/rho_<rho>/rdf.dat

Both files should be 3-column: distance(Å)  number_integral  g(r)
(Travis outputs pm – set --travis_pm flag to auto-convert)

"""

import os, argparse
import numpy as np
import matplotlib.pyplot as plt

def load_rdf(path, pm_to_ang=False):
    """Load a 3-col RDF file. Skips comment lines starting with '#'."""
    data = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    data.append([float(x) for x in parts[:3]])
                except ValueError:
                    pass
    arr = np.array(data)
    if pm_to_ang:
        arr[:, 0] /= 100.0   # pm → Å
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--md_dir',   default='results/part1')
    parser.add_argument('--mc_dir',   default='mc_data')
    parser.add_argument('--travis_pm', action='store_true',
                        help='If set, convert MD RDF distances from pm to Å')
    parser.add_argument('--out_dir',  default='plots/rdf')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    SIGMA = 3.4   # Å – for x-axis in reduced units

    for rho in densities:
        rho_str = f'{rho:.1f}'

        md_path = os.path.join(args.md_dir, f'rho_{rho_str}', 'nproc_1', 'rdf.dat')
        mc_path = os.path.join(args.mc_dir, f'rho_{rho_str}', 'rdf.dat')

        has_md = os.path.exists(md_path)
        has_mc = os.path.exists(mc_path)

        if not has_md and not has_mc:
            print(f"  rho*={rho_str}: no RDF files found – skipping")
            continue

        fig, ax = plt.subplots(figsize=(7, 4.5))

        if has_md:
            md = load_rdf(md_path, pm_to_ang=args.travis_pm)
            r_red = md[:, 0] / SIGMA
            ax.plot(r_red, md[:, 2], color='royalblue', lw=1.5, label='MD')
        else:
            print(f"  rho*={rho_str}: MD RDF not found ({md_path})")

        if has_mc:
            mc = load_rdf(mc_path)
            r_red = mc[:, 0] / SIGMA
            ax.plot(r_red, mc[:, 2], color='tomato', lw=1.5,
                    ls='--', label='MC (Project 1)')
        else:
            print(f"  rho*={rho_str}: MC RDF not found ({mc_path})")

        ax.axhline(1, color='gray', lw=0.7, ls=':')
        ax.set_xlabel('r / σ', fontsize=12)
        ax.set_ylabel('g(r)', fontsize=12)
        ax.set_title(f'Radial Distribution Function   ρ* = {rho_str}', fontsize=13)
        ax.set_xlim(0, 4)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fname = os.path.join(args.out_dir, f'rdf_rho{rho_str.replace(".","")}.png')
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}")

    print(f"\nDone. RDF plots in {args.out_dir}/")


if __name__ == '__main__':
    main()
