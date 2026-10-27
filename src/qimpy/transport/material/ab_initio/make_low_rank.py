import argparse
import h5py
import numpy as np
import torch
import qimpy
from qimpy import log, rc, io
from qimpy.mpi import ProcessGrid
from qimpy.profiler import StopWatch
from . import AbInitio

def orthonormalize_observables(M, tol=1e-10):
    """
    Construct an orthonormal basis for the columns of M.

    M.shape = (N, n_observables)
    """
    U, S, _ = np.linalg.svd(M, full_matrices=False)

    if not len(S) or S[0] == 0:
        raise ValueError("No nonzero observable vectors found.")

    keep = S > tol * S[0]
    Q = U[:, keep]

    log.info(
        f"Protected observable subspace dimension: {Q.shape[1]}"
    )

    return Q



def construct_observable_basis(ab_initio, N):
    """
    Construct protected charge and momentum observable vectors
    in the packed-Hermitian representation.

    Protected observables: I, Px, Py, Pz.
    The flattened ordering must match Lindblad.P.
    """
    nk = ab_initio.k_division.n_tot
    nb = ab_initio.n_bands
    ph = ab_initio.packed_hermitian

    if N != nk * nb**2:
        raise ValueError(
            f"Unexpected Lindblad dimension {N}; "
            f"expected {nk * nb**2}."
        )

    # Identity operator for charge
    momentum = ab_initio.P

    identity = torch.eye(
        nb,
        dtype=momentum.dtype,
        device=momentum.device,
    )
    identity = identity[None].expand(nk, -1, -1)

    # Momentum operators: Px, Py, Pz
    # Expected shape: (nk, 3, nb, nb)
    if momentum.shape != (nk, 3, nb, nb):
        raise ValueError(
            f"Unexpected momentum tensor shape: {momentum.shape}. "
            f"Expected ({nk}, 3, {nb}, {nb})."
        )

    # Shape: (4, nk, nb, nb)
    operators = torch.cat(
        (
            identity[None],
            momentum.permute(1, 0, 2, 3),
        ),
        dim=0,
    )

    # Pack into the same representation used by Lindblad
    packed = ph.pack(operators)

    M = (
        packed.reshape(packed.shape[0], -1)
        .T.detach()
        .to(rc.cpu)
        .numpy()
    )

    return orthonormalize_observables(M)


def protected_low_rank(P, Q_left, Q_right, max_rank):
    """
    Observable-preserving low-rank approximation
    following the professor's compact factorization.

    Preserve the three protected blocks exactly.
    Apply truncated SVD only to the fourth block.

    max_rank is the total output factor rank.
    """

    N_out, N_in = P.shape

    ml = Q_left.shape[1]
    mr = Q_right.shape[1]

    if Q_left.shape[0] != N_out:
        raise ValueError("Left basis dimension mismatch.")

    if Q_right.shape[0] != N_in:
        raise ValueError("Right basis dimension mismatch.")

    protected_rank = ml + mr

    if max_rank < protected_rank:
        raise ValueError(
            f"max_rank={max_rank} is too small. "
            f"Need at least {protected_rank}."
        )

    # -----------------------------------------
    # Step 1: Construct protected contributions
    # -----------------------------------------

    # P Q_right
    B = P @ Q_right

    # Q_left^dagger P
    C = Q_left.conj().T @ P

    # Q_left^dagger P Q_right
    A = Q_left.conj().T @ B

    # -----------------------------------------
    # Step 2: Construct the fourth block
    #
    # R_left P R_right
    # where R = I - Q Q^dagger
    # -----------------------------------------

    residual = (
        P
        - B @ Q_right.conj().T
        - Q_left @ C
        + Q_left @ A @ Q_right.conj().T
    )

    # -----------------------------------------
    # Step 3: Truncated SVD of the residual
    # -----------------------------------------


    residual_rank = max_rank - protected_rank

    if residual_rank == 0:
        # All 8 factor columns are used by protected blocks.
        # No residual SVD is necessary.
        Ur = np.empty((N_out, 0), dtype=P.dtype)
        S = np.empty((0,), dtype=float)
        Vhr = np.empty((0, N_in), dtype=P.dtype)
    else:
        Ur, S, Vhr = np.linalg.svd(
            residual, full_matrices=False
        )
        residual_rank = min(residual_rank, len(S))

    # Keep only the selected singular modes
    Ur = Ur[:, :residual_rank]
    Sr = S[:residual_rank]
    Vhr = Vhr[:residual_rank, :]

    # Check that residual SVD modes lie
    # outside the protected subspaces
    left_orthogonality = np.linalg.norm(
        Q_left.conj().T @ Ur
    )

    right_orthogonality = np.linalg.norm(
        Vhr @ Q_right
    )

    # Absorb sqrt(Sigma) into both factors
    sqrtS = np.sqrt(Sr)

    Ur_scaled = Ur * sqrtS[None, :]
    Vhr_scaled = sqrtS[:, None] * Vhr

    # -----------------------------------------
    # Step 4: Professor's compact factorization
    #
    # U_final = [P Q - Q(Q^dagger P Q),
    #            Q,
    #            U_r sqrt(Sigma)]
    #
    # Vdag_final = [Q^dagger;
    #               Q^dagger P;
    #               sqrt(Sigma) V_r^dagger]
    # -----------------------------------------

    U_final = np.concatenate(
        (
            B - Q_left @ A,
            Q_left,
            Ur_scaled,
        ),
        axis=1,
    )

    Vdag_final = np.concatenate(
        (
            Q_right.conj().T,
            C,
            Vhr_scaled,
        ),
        axis=0,
    )

    # -----------------------------------------
    # Step 5: Validate the approximation
    # -----------------------------------------

    P_approx = U_final @ Vdag_final

    normP = np.linalg.norm(P)

    relative_error = (
        np.linalg.norm(P - P_approx) / normP
        if normP > 0 else 0.0
    )

    right_error = np.linalg.norm(
        (P - P_approx) @ Q_right
    )

    left_error = np.linalg.norm(
        Q_left.conj().T @ (P - P_approx)
    )

    log.info(
        f"Total factor rank: {U_final.shape[1]}"
    )
    log.info(
        f"Residual SVD rank: {residual_rank}"
    )
    log.info(
        f"Relative reconstruction error: "
        f"{relative_error:.6e}"
    )
    log.info(
        f"Protected right error: {right_error:.6e}"
    )
    log.info(
        f"Protected left error: {left_error:.6e}"
    )
    log.info(
        f"Residual left orthogonality: "
        f"{left_orthogonality:.6e}"
    )
    log.info(
        f"Residual right orthogonality: "
        f"{right_orthogonality:.6e}"
    )

    return U_final, S, Vdag_final


def generate_low_rank(
    *,
    ab_initio: AbInitio | dict,
    low_rank_file: str,
    max_rank: int,
) -> None:
    """
    Generate observable-preserving low-rank
    approximations to Lindblad P and Pbar.
    """

    process_grid = ProcessGrid("rk", (1, -1))

    ab_initio = AbInitio(
        **ab_initio,
        process_grid=process_grid,
    )

    assert not ab_initio.lindblad.low_rank_file

    # Full Lindblad matrices:
    # P.shape = (2, N_out, N_in)
    P = (
        ab_initio.lindblad.P
        .detach()
        .to(rc.cpu)
        .numpy()
    )

    log.info(f"Lindblad tensor shape: {P.shape}")

    N_out, N_in = P.shape[1:]

    # Construct protected observable bases.
    # Current ProcessGrid is intended to
    # construct the complete operator.
    if N_out != N_in:
        raise ValueError(
            "This implementation requires a square "
            "global Lindblad matrix."
        )

    Q = construct_observable_basis(
        ab_initio, N_in
    )

    U_list = []
    S_list = []
    Vdag_list = []

    for i, name in enumerate(("P", "Pbar")):

        log.info(
            f"Constructing protected low-rank {name}"
        )

        U, S, Vdag = protected_low_rank(
            P[i],
            Q,
            Q,
            max_rank,
        )

        U_list.append(U)
        S_list.append(S)
        Vdag_list.append(Vdag)

    U = np.stack(U_list, axis=0)
    S = np.stack(S_list, axis=0)
    Vdag = np.stack(Vdag_list, axis=0)

    # Save in the same format expected by
    # the existing _lindblad.py.
    with h5py.File(low_rank_file, "w") as fp:
        fp["S"] = S
        fp["U"] = U
        fp["Vdag"] = Vdag

    log.info(
        f"Saved protected low-rank factors to {low_rank_file}"
    )


def main():
    parser = argparse.ArgumentParser(
        prog="python -m qimpy.transport.material.ab_initio.make_low_rank",
        description="Observable-preserving low-rank Lindblad scattering",
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

    log.info(
        "*" * 15
        + " QimPy "
        + qimpy.__version__
        + " " * 1
        + "*" * 15
    )

    rc.init()

    input_dict = io.dict.key_cleanup(
        io.yaml.load(args.input_file)
    )

    log.info(
        f"\n# Processed input:\n"
        f"{io.yaml.dump(input_dict)}"
    )

    input_dict = io.dict.remove_units(input_dict)

    generate_low_rank(**input_dict)

    rc.free()
    rc.report_end()
    StopWatch.print_stats()


if __name__ == "__main__":
    main()

