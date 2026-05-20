import math


class CurveStandard:
    name: str
    url: str


class NIST(CurveStandard):
    name = "NIST"
    url = "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-186.pdf"


class SEC2(CurveStandard):
    name = "SEC2"
    url = "https://www.secg.org/sec2-v2.pdf"


class EllipticCurve:
    def __init__(self, name: str, standard: type[CurveStandard], size: int, n: int, h: int, p: int, d_size: int):
        """
        :param n: order of the curve
        :param h: cofactor
        :param p: modulus of the group (specifying the field Fp)
        """
        self.name = name
        self.standard = standard
        self._n = n
        self._h = h
        self.size = size
        self.E = n * h
        self.mod = p
        self.d_size = d_size


P256Curve = EllipticCurve(
    name="P256",
    standard=NIST,
    size=256,
    n=0xffffffff_00000000_ffffffff_ffffffff_bce6faad_a7179e84_f3b9cac2_fc632551,
    h=1,
    p=0xffffffff_00000001_00000000_00000000_00000000_ffffffff_ffffffff_ffffffff,
    d_size=256)

W25519Curve = EllipticCurve(
    name="W25519",
    standard=NIST,
    size=256,
    n=(1 << 252) + 0x14def9de_a2f79cd6_5812631a_5cf5d3ed,
    h=8,
    p=0x7fffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffed,
    d_size=256)

Curve25519Curve = W25519Curve
Edwards25519Curve = W25519Curve

secp256k1Curve = EllipticCurve(
    name="secp256k1",
    standard=SEC2,
    size=256,
    n=0xffffffff_ffffffff_ffffffff_fffffffe_baaedce6_af48a03b_bfd25e8c_d0364141,
    h=1,
    p=0xffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_fffffffe_fffffc2f,
    d_size=256)

secp256r1Curve = EllipticCurve(
    name="secp256r1",
    standard=SEC2,
    size=256,
    n=0xffffffff_00000000_ffffffff_ffffffff_bce6faad_a7179e84_f3b9cac2_fc632551,
    h=1,
    p=0xFFFFFFFF_00000001_00000000_00000000_00000000_FFFFFFFF_FFFFFFFF_FFFFFFFF,
    d_size=256)

curves = {"P256": P256Curve,
          "W25519": W25519Curve,
          "Curve25519": Curve25519Curve,
          "Edwards25519": Edwards25519Curve,
          "secp256k1": secp256k1Curve,
          "secp256r1": secp256r1Curve,}


def hasses_theorem_curve_test(curve: EllipticCurve):
    """
    :param curve: curve to be tested
    :return: True if curve passes the test, False otherwise
    """
    q = curve.mod
    sqrt_q = math.sqrt(q)
    sqrt_q_2_times = int(2 * sqrt_q)
    print(q - sqrt_q_2_times, bin(q - sqrt_q_2_times))
    print(q + sqrt_q_2_times, bin(q + sqrt_q_2_times))

    return q - sqrt_q_2_times < curve.E < q + sqrt_q_2_times


if __name__ == "__main__":
    for curve in curves.values():
        print(f"{curve.name} curve test: {hasses_theorem_curve_test(curve)}")
