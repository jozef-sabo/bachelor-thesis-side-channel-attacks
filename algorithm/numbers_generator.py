import argparse
import pickle
import random
import logging
import sys
import secrets
import time

import numpy as np

from elliptic_curves import EllipticCurve, curves


def generate_blinders(curve: EllipticCurve, multipliers: list[int]) -> list[int]:
    logging.debug(f"Computing {len(multipliers)} blinders (E*r), curve {curve.name}")

    output = []
    for r in multipliers:
        blinder = curve.E * r
        output.append(blinder)

    return output


def generate_multipliers(number_of_values: int, multiplier_bits: int, original: bool) -> list[int]:
    logging.info(f"Genrating {number_of_values} multipliers (r) of {multiplier_bits} bits")

    output = []
    for _ in range(number_of_values):
        if original:
            r = random.randint(3<<(multiplier_bits-2),(1<<multiplier_bits)-1)
            assert r.bit_length() == multiplier_bits
        else:
            r = secrets.randbits(multiplier_bits)
        logging.debug(f"({_ + 1}/{number_of_values}), r = {r}")
        output.append(r)

    return output


def generate_blinded_values(d: int, blinders: list[int]) -> list[int]:
    logging.info(f"Computing {len(blinders)} blinded values (d + E*r)")

    output = []
    for Er in blinders:
        blinded_d = d + Er
        output.append(blinded_d)

    return output


def generate_error_vectors(size: int, error_rate: int, number_of_values: int, original: bool) -> list[int]:
    logging.info(
        f"Genrating {number_of_values} error vectors (εi) with error rate of {error_rate}% and size {size} bits")

    output = []
    for _ in range(number_of_values):
        e = 0
        if original:
            sigma = 0.5  # for \epsilon_b = 0.15
            b = np.zeros(size, dtype=np.uint8)
            b_noisy = b + np.random.normal(0, sigma, size=b.shape)
            b_noisy = np.uint8(b_noisy > 0.5)

            pad_length = (8 - len(b) % 8) % 8
            b_padded = np.pad(b_noisy, (pad_length, 0), mode="constant")
            byte_data = np.packbits(b_padded).tobytes()
            e = int.from_bytes(byte_data, byteorder="big")
        else:
            for bit_num in range(size):
                if secrets.randbelow(100) < error_rate:
                    e ^= (1 << bit_num)
        logging.debug(f"({_ + 1}/{number_of_values}), εi = {e}")
        output.append(e)

    return output


def apply_error_vectors(blinded_ds: list[int], error_vectors: list[int]):
    output = []
    for blinded_d, error_vector in zip(blinded_ds, error_vectors):
        blinded_with_e = blinded_d ^ error_vector
        output.append(blinded_with_e)

    return output


def generate_out_binary(filename: str, mode: str, curve: EllipticCurve, d: int, blinded_with_errors: list[int],
                        multipliers: list[int], multiplier_bits: int, error_vectors: list[int], error_rate: int):
    data = {
        "curve_size": curve.size,
        "E": curve.E,
        "d": d,
        "multiplier_size": multiplier_bits,
        "error_rate": error_rate,
        "blinded_with_errors": blinded_with_errors,
        "multipliers": multipliers,
        "error_vectors": error_vectors,
    }


    with open(filename, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_values(curve: EllipticCurve, d: int, number_of_values: int, multiplier_bits: int, error_rate: int, original: bool):
    out_value_size = curve.size + multiplier_bits

    multipliers = generate_multipliers(number_of_values, multiplier_bits, original)
    blinders = generate_blinders(curve, multipliers)
    blinded_ds = generate_blinded_values(d, blinders)
    error_vectors = generate_error_vectors(out_value_size, error_rate, number_of_values, original)

    blinded_with_errors = apply_error_vectors(blinded_ds, error_vectors)

    return blinded_with_errors, multipliers, error_vectors, d


def generate(curve: EllipticCurve, number_of_values: int, multiplier_bits: int, error_rate: int,
             output_file: str, original: bool):
    logging.info(f"Generating random value (d) of {curve.size} bits")
    d = random.getrandbits(curve.d_size)
    logging.info(f"d = {d}, size {curve.size} bits")

    blinded_with_errors, multipliers, error_vectors, d = generate_values(curve, d, number_of_values, multiplier_bits,
                                                                         error_rate, original)

    generate_out_binary(output_file, "w", curve, d, blinded_with_errors, multipliers, multiplier_bits, error_vectors,
                      error_rate)


if __name__ == "__main__":
    logging.basicConfig(
        handlers=[
            logging.FileHandler("syslog.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ],
        # level=logging.ERROR,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(name)s@%(threadName)s:%(message)s'
    )
    parser = argparse.ArgumentParser(
        prog='EC Blinded values generator',
        description='This program is used for blinded values generation')

    parser.add_argument('curve', choices=curves.keys(),
                        help="Type of curve used to generate values")
    parser.add_argument('count', help="Number of generated values", type=int)
    parser.add_argument('out_file', help="Output filename", type=str)
    parser.add_argument('-e', "--error-rate", choices=range(0, 100 + 1), help="Percentage of the error rate (ε)",
                        type=int, default=15)
    parser.add_argument('-m', "--multiplier-bits", help="Number of bits of a multiplier (r)", type=int, default=64)
    parser.add_argument('-o', '--original', help="Uses original generation proposed by authors", action="store_true")

    args = parser.parse_args()

    t1 = time.perf_counter(), time.process_time()

    generate(curves[args.curve], args.count, args.multiplier_bits, args.error_rate, args.out_file, args.original)

    t2 = time.perf_counter(), time.process_time()
    print(f"Real time: {t2[0] - t1[0]:.2f} seconds", file=sys.stderr)
    print(f"CPU time: {t2[1] - t1[1]:.2f} seconds", file=sys.stderr)
    print("", file=sys.stderr)
