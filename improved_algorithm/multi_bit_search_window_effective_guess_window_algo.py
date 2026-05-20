from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AttackParams:
    K: int
    E: int
    eps_b: float
    eps_b_compl: float
    eps_b_multipliers: list[float]
    eps_b_compl_multipliers: list[float]


def _mask(width: int) -> int:
    return (1 << width) - 1


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def evaluate_probability(
        r_window_pred: int,
        d_window_pred: int,
        noisy_scalar: int,
        start_bit: int,
        window_size: int,
        params: AttackParams,
) -> float:
    """Evaluate P(observation | candidate) for one trace.

    start_bit is a 0-based bit offset of the window LSB within each K-bit half.
    """
    window_mask = _mask(window_size)

    noisy_r_window = (noisy_scalar >> (params.K + start_bit)) & window_mask
    noisy_d_window = (noisy_scalar >> start_bit) & window_mask

    h = hamming_distance(r_window_pred, noisy_r_window) + hamming_distance(d_window_pred, noisy_d_window)
    return params.eps_b_multipliers[h] * params.eps_b_compl_multipliers[2 * window_size - h]


def improved_algorithm_step1(
        iteration: int,
        window_t: int,
        noisy_scalars: list[int],
        Lr: list[list[tuple[float, int]]],
        d_mod_2_i_minus_1: int,
        params: AttackParams,
        d_add_window: int = 1,
        d_widening_window: int = 1,
) -> int:
    """Paper-style step 1: decide the next bit of d."""
    iters = 1 << d_add_window
    fixed_bits = iteration - d_widening_window

    scores = [0.0 for _ in range(iters)]

    score_width = iteration + d_add_window - d_widening_window
    width = min(window_t, score_width)
    start_bit = max(score_width - width, 0)

    d_low = d_mod_2_i_minus_1 & _mask(fixed_bits)
    mod_i = _mask(score_width)

    for d_hat in range(iters):
        d_bar = d_low | (d_hat << fixed_bits)

        for ell, noisy_scalar in enumerate(noisy_scalars):
            for _, r_tilde in Lr[ell]:
                for r_hat in range(iters):
                    r_bar = r_tilde | (r_hat << fixed_bits)
                    d_bar_ell = (r_bar * params.E + d_bar) & mod_i

                    r_window = r_bar >> start_bit
                    d_window = d_bar_ell >> start_bit
                    scores[d_hat] += evaluate_probability(
                        r_window,
                        d_window,
                        noisy_scalar,
                        start_bit,
                        width,
                        params,
                    )

    chosen_iters = 1 << d_widening_window
    chosen_mask = _mask(d_widening_window)
    scores_binary = [0.0 for _ in range(chosen_iters)]
    for d_hat, score in enumerate(scores):
        scores_binary[d_hat & chosen_mask] += score

    argmax = max(range(chosen_iters), key=lambda i: scores_binary[i])
    return d_low | (argmax << fixed_bits)


def improved_algorithm_step2(
        iteration: int,
        L: int,
        noisy_scalars: list[int],
        Lr: list[list[tuple[float, int]]],
        d_star: int,
        params: AttackParams,
        d_add_window: int = 1,
        d_widening_window: int = 1,
) -> list[list[tuple[float, int]]]:
    """Paper-style step 2: extend each per-trace candidate list and keep top-L."""
    iters = 1 << d_add_window
    fixed_bits = iteration - d_widening_window

    score_width = iteration + d_add_window - d_widening_window
    mod_i = _mask(score_width)

    updated: list[list[tuple[float, int]]] = []
    for ell, row in enumerate(Lr):
        # Keep only the integer candidates from the previous phase.
        prev_candidates = [cand for _, cand in row]
        scored: dict[int, float] = {}

        for prev in prev_candidates:
            for add_bit in range(iters):
                cand = prev | (add_bit << fixed_bits)
                d_bar_ell = (cand * params.E + d_star) & mod_i
                p = evaluate_probability(cand, d_bar_ell, noisy_scalars[ell], 0, score_width, params)

                # Merge duplicates by keeping the better score.
                old = scored.get(cand)
                if old is None or p > old:
                    scored[cand] = p

        best_sorted = sorted(((p, cand) for cand, p in scored.items()), key=lambda x: (-x[0], x[1]))

        best = best_sorted[:L]

        updated.append(best)

    return updated


def benchmark(
        noisy_scalars: list[int],
        d_true: int,
        R: int,
        L: int,
        t: int,
        params: AttackParams,
        d_add_window: int = 1,
        d_widening_window: int = 1,
) -> int:
    """Return number of correctly recovered low bits.

    Baseline mode (revisited_config is None) follows greedy paper-style progression.
    Revisited mode applies beam expansion over d hypotheses and validates either:
    - only the best beam element, or
    - whether the beam still contains the true prefix.
    """
    if d_add_window < 1:
        raise ValueError("d_add_window must be >= 1")
    if d_widening_window < 1:
        raise ValueError("d_widening_window must be >= 1")
    if d_widening_window > d_add_window:
        raise ValueError("d_widening_window must be <= d_add_window")

    Lr: list[list[tuple[float, int]]] = [[(1.0, 0)] for _ in noisy_scalars]

    d_prefix = 0
    iteration = 0
    while iteration < R:
        chunk = min(d_widening_window, R - iteration)
        iteration += chunk

        d_star = improved_algorithm_step1(
            iteration,
            t,
            noisy_scalars,
            Lr,
            d_prefix,
            params,
            d_add_window=d_add_window,
            d_widening_window=chunk,
        )
        if (d_star & _mask(iteration)) != (d_true & _mask(iteration)):
            return iteration - chunk

        Lr = improved_algorithm_step2(
            iteration,
            L,
            noisy_scalars,
            Lr,
            d_star,
            params,
            d_add_window=d_add_window,
            d_widening_window=chunk,
        )
        d_prefix = d_star
    return R


# PROCESSING PRECOMPUTED DATASET FILES
def process_binary(file: str) -> tuple[int, int, int, list[int], list[int], list[int], int, int]:
    """
    Loads and processes binary file generated by numbers_generator
    :param file: Path to a file to be processed
    :return: tuple consisting of:
        - E - elliptic curve group order
        - d - private key
        - ε - error rate of a given dataset
        - d~_l - list of blinded values with error applied (using XOR)
        - e_l - list error vectors
        - r_l - list of multipliers
        - K - curve size in bits
        - R - multiplier size in bits
    """

    data = pickle.loads(open(file, "rb").read())
    E = data["E"]
    d = data["d"]
    error_rate = data["error_rate"]
    blinded_with_errors = data["blinded_with_errors"]
    multipliers = data["multipliers"]
    error_vectors = data["error_vectors"]
    curve_size = data["curve_size"]
    multiplier_size = data["multiplier_size"]

    return E, d, error_rate, blinded_with_errors, error_vectors, multipliers, curve_size, multiplier_size


def change_error_rate(old_blnd_w_errs: list[int], old_err_vects: list[int],
                      new_err_vects: list[int], new_err_rate: int) \
        -> tuple[list[int], list[int], int]:
    """
    Removes old error vectors from a dataset and adds new error vectors to a dataset
    :param old_blnd_w_errs: Blinded values with errors
    :param old_err_vects: Error vectors of blinded values
    :param new_err_vects: Error vectors to be applied
    :param new_err_rate: New intended error rate, error rate of the new error vectors
    :return: Blinded values with new error vectors, new error vectors and the new error rate
    """
    # working on principle
    #     blinded_values_with_errors    XOR old_error_vectors == blinded_values_without_errors
    #     blinded_values_without_errors XOR new_error_vectors == blinded_values_with_new_errors
    #     THIS IS EQUIVALENT TO
    #     (old_error_vectors XOR new_error_vectors) XOR blinded_values_with_errors == blinded_values_with_new_errors
    translating_error_vectors = [error_vector ^ new_error_vector for error_vector, new_error_vector in
                                 zip(old_err_vects, new_err_vects)]

    blinded_with_errors = [blinded_with_error ^ translating_error_vector for
                           blinded_with_error, translating_error_vector in
                           zip(old_blnd_w_errs, translating_error_vectors)]

    return blinded_with_errors, new_err_vects, new_err_rate


def load_binary(file: str, replace_errors_file: str = None) \
        -> tuple[int, int, int, list[int], list[int], list[int], int, int]:
    """
    Loads and processes binary file generated by numbers_generator
    If replace_errors_file is not None, replaces error vectors of old file with error vectors of file given by the value
    :param file: Path to a file to be processed
    :param replace_errors_file: Path to a file with new error vectors
    :return: tuple consisting of:
        - E - elliptic curve group order
        - d - private key
        - ε - error rate of a given dataset
        - d~_l - list of blinded values with error applied (using XOR)
        - e_l - list error vectors
        - r_l - list of multipliers
        - K - curve size in bits
        - R - multiplier size in bits
    """
    E, d, err_rate, blinded_w_errs, err_vectors, multipliers, curve_s, multiplier_s = process_binary(file)

    if replace_errors_file:
        _, _, new_err_rate, _, new_err_vectors, _, _, _ = process_binary(replace_errors_file)
        blinded_w_errs, err_vectors, err_rate = (
            change_error_rate(blinded_w_errs, err_vectors, new_err_vectors, new_err_rate))

    return E, d, err_rate, blinded_w_errs, err_vectors, multipliers, curve_s, multiplier_s


def run_dataset(
        file_path: str | Path,
        N: int,
        L: int,
        t: int,
        d_add_window: int,
        d_widening_window: int) -> int:
    E, d, error_rate, blinded_with_errors, _, _, curve_size, R = load_binary(file_path, None)
    K = curve_size
    eps_b = error_rate / 100.0
    eps_b_multipliers = [eps_b ** i for i in range(2 * K + 1)]
    eps_b_compl_multipliers = [(1 - eps_b) ** i for i in range(2 * K + 1)]
    params = AttackParams(K=K, E=E, eps_b=eps_b, eps_b_compl=1 - eps_b, eps_b_multipliers=eps_b_multipliers,
                          eps_b_compl_multipliers=eps_b_compl_multipliers)
    return benchmark(blinded_with_errors[:N], d, R, L, t, params, d_add_window, d_widening_window)


def main():
    curve_name = "secp256k1Curve"
    try_num = 1
    file_path = Path(f"./{curve_name}_e15_r64_10000-{try_num}.pkl")
    N = 500
    L = 32
    t = 16
    # w_d for lookahead
    d_add_window = 1
    # w_{d_e} for lookahead stripping
    d_widening_window = 1
    print(run_dataset(file_path, N=N, L=L, t=t, d_add_window=d_add_window, d_widening_window=d_widening_window))


if __name__ == "__main__":
    main()
