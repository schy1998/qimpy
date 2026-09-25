import argparse
import time

import h5py
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

import qimpy
from qimpy import log, rc, io
from qimpy.mpi import ProcessGrid
from qimpy.profiler import StopWatch
from . import AbInitio


def generate_low_rank(
    *,
    ab_initio: AbInitio | dict,
    low_rank_file: str,
    max_rank: int,
) -> None:
    """Calculate singular-value spectra of ab_initio.lindblad.P."""

    process_grid = ProcessGrid("rk", (1, -1))
    ab_initio = AbInitio(**ab_initio, process_grid=process_grid)
    assert not ab_initio.lindblad.low_rank_file

    # Full Lindblad operators P and Pbar are stacked along dimension 0:
    # P.shape = (2, N_out, N_in)
    P = ab_initio.lindblad.P.detach().to(rc.cpu).numpy()  # to CPU/NumPy for SVD

    log.info(f"Lindblad tensor shape: {P.shape}")

    U, S, Vdag = np.linalg.svd(P, full_matrices=False)

    sqrtS = np.sqrt(S[..., :max_rank])

    U = U[..., :max_rank] * sqrtS[:, None, :]
    Vdag = Vdag[..., :max_rank, :] * sqrtS[:, :, None]

    # Write to file:
    with h5py.File(low_rank_file, "w") as fp:
        fp["S"] = S
        fp["U"] = U
        fp["Vdag"] = Vdag
    
    return
    
    # -----------------------------------
    
    singular_values = []
    

    for i, name in enumerate(("P", "Pbar")):
        log.info(f"Computing singular values of {name}")
        log.info(f"{name} matrix shape: {P_np[i].shape}")

        start = time.perf_counter()

        # Singular values only: no U or Vdag.
        S = scipy.linalg.svdvals(P_np[i])

        elapsed = time.perf_counter() - start
        
        singular_values.append(S)

        # Save immediately so an expensive SVD is not lost.
        with h5py.File(low_rank_file, "a") as fp:
            dataset_name = f"S_{name}"
            if dataset_name in fp:
                del fp[dataset_name]
            fp[dataset_name] = S

        log.info(f"{name} SVD time: {elapsed:.3f} s")
        log.info(f"{name} number of singular values: {len(S)}")
        log.info(f"{name} largest singular value: {S[0]:.6e}")
        log.info(f"{name} smallest singular value: {S[-1]:.6e}")

        # Cumulative fraction of sum of singular values.
        cumulative = np.cumsum(S**2) / np.sum(S**2)

        for fraction in (0.50, 0.90, 0.99):
            n_required = np.searchsorted(cumulative, fraction) + 1
            log.info(
                f"{name}: {100*fraction:.0f}% cumulative squared singular-value "
                f"weight requires {n_required} / {len(S)} modes "
                f"({100*n_required/len(S):.2f}%)"
            )

    # ---------------------------------------------------------
    # Plot 1: singular-value spectrum
    # ---------------------------------------------------------

    plt.figure(figsize=(6, 4))

    for S, name in zip(singular_values, ("P", "Pbar")):
        index = np.arange(1, len(S) + 1)
        plt.semilogy(index, S, label=name)

    plt.xlabel("Singular-value index")
    plt.ylabel("Singular value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("singular_value_spectrum.pdf")
    plt.close()

    # ---------------------------------------------------------
    # Plot 2: cumulative singular-value weight
    # ---------------------------------------------------------

    plt.figure(figsize=(6, 4))

    for S, name in zip(singular_values, ("P", "Pbar")):
        index = np.arange(1, len(S) + 1)
        cumulative = np.cumsum(S**2) / np.sum(S**2)
        plt.plot(index, cumulative, label=name)

    plt.axhline(0.50, linestyle="--", linewidth=0.8)
    plt.axhline(0.90, linestyle="--", linewidth=0.8)
    plt.axhline(0.99, linestyle="--", linewidth=0.8)

    plt.xlabel("Number of singular values retained")
    plt.ylabel("Cumulative fraction")
    plt.ylim(0.0, 1.01)
    plt.legend()
    plt.tight_layout()
    plt.savefig("singular_value_cumulative.pdf")
    plt.close()
    
    

def main():
    parser = argparse.ArgumentParser(
        prog="python -m qimpy.transport.material.ab_initio.make_low_rank",
        description="Analyze singular-value spectrum of Lindblad scattering",
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="YAML input file",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        metavar="FILE",
        help="output file (stdout if unspecified)",
    )

    args = parser.parse_args()
    io.log_config(output_file=args.output_file)

    # Print version header
    log.info("*" * 15 + " QimPy " + qimpy.__version__ + " " + "*" * 15)

    # Configure hardware resources
    rc.init()

    # Load input parameters from YAML file:
    input_dict = io.dict.key_cleanup(io.yaml.load(args.input_file))
    log.info(f"\n# Processed input:\n{io.yaml.dump(input_dict)}")
    input_dict = io.dict.remove_units(input_dict)  # Remove units

    # Analyze singular-value spectrum:
    generate_low_rank(**input_dict)

    # Cleanup and report timings:
    rc.free()
    rc.report_end()
    StopWatch.print_stats()


if __name__ == "__main__":
    main()
