from symforce import symbolic as sf


def expm(mat: sf.Matrix) -> sf.Matrix:
    """matrix exponential via first 5 terms of power series expansion"""

    out = sf.Matrix.eye(mat.rows, mat.cols)
    out += mat
    out += (mat * mat) / 2.0
    out += (mat * mat * mat) / 6.0
    out += (mat * mat * mat * mat * mat) / 24.0

    return out
