"""CS 639, Fall 2026 -- Homework 0, Problem 4.

Complete the two functions marked TODO, then run

    python hw0.py

The script prints your Python, NumPy, and Matplotlib versions, checks both
functions against known answers, and saves the figure hw0_fig.png.  Paste the
printed output and include the figure in your PDF; this file itself is not
submitted.
"""
import sys

import numpy as np


# ----------------------------------------------------------------- Problem 4(a)
def standardize_columns(X):
    """Return the array Z with Z[:, j] = (X[:, j] - m_j) / s_j, where m_j and
    s_j are the mean and the standard deviation of column j of X.

    X is a 2-D array with at least two rows and no constant column.  Use the
    population standard deviation (NumPy's default, ddof=0) and no Python
    loop (hint: the `axis` argument and broadcasting).
    """
    X = np.asarray(X, dtype=float)
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    return Z


def least_squares(X, y):
    """Return the w minimizing ||X w - y||^2, for X of shape (n, d) with
    linearly independent columns and y of shape (n,).

    This is w* from Problem 3(b).  Prefer np.linalg.solve to forming an
    inverse; any other correct method (e.g. np.linalg.lstsq) is also fine.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.linalg.solve(X.T @ X, X.T @ y)


# ------------------------------------------------- figure (nothing to code here)
def make_figure(filename="hw0_fig.png"):
    """Plot the running average of repeated rolls of a fair die, for three
    independent sequences of rolls.  This only checks that Matplotlib is
    installed and can save a file."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(639)
    n = np.arange(1, 2001)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for _ in range(3):
        rolls = rng.integers(1, 7, size=n.size)
        ax.plot(n, np.cumsum(rolls) / n, linewidth=1)
    ax.axhline(3.5, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("number of rolls $n$")
    ax.set_ylabel("average of the first $n$ rolls")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    return filename


# ------------------------------------------------------------------ self-check
def _report(name, fn):
    """Run fn(), which returns (passed, detail); print one line."""
    try:
        passed, detail = fn()
    except NotImplementedError:
        print(f"[TODO] {name}: not implemented yet")
        return False
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: raised {type(e).__name__}: {e}")
        return False
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return passed


def _check_standardize_columns():
    Z = np.asarray(standardize_columns([[1, 2], [3, 6]]))
    want = [[-1.0, -1.0], [1.0, 1.0]]
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 4)) * [1, 2, 3, 4] + [5, 6, 7, 8]
    Zr = np.asarray(standardize_columns(X))
    ok = (Z.shape == (2, 2) and np.allclose(Z, want) and Zr.shape == X.shape
          and np.allclose(Zr.mean(axis=0), 0) and np.allclose(Zr.std(axis=0), 1))
    return ok, f"[[1, 2], [3, 6]] -> {np.round(Z, 6).tolist()}, expected {want}"


def _check_least_squares():
    X = [[1, 0], [1, 1], [1, 2]]
    y = [1, 3, 5]
    w = np.asarray(least_squares(X, y))
    rng = np.random.default_rng(2)
    Xr, yr = rng.normal(size=(30, 3)), rng.normal(size=30)
    ref = np.linalg.lstsq(Xr, yr, rcond=None)[0]
    ok = w.shape == (2,) and np.allclose(w, [1, 2]) and np.allclose(least_squares(Xr, yr), ref)
    return ok, f"X = {X}, y = {y} -> w = {np.round(w, 6).tolist()}, expected [1.0, 2.0]"


def main():
    print(f"Python {sys.version.split()[0]}, NumPy {np.__version__}", end="")
    try:
        import matplotlib

        print(f", Matplotlib {matplotlib.__version__}")
    except ImportError:
        print(", Matplotlib NOT INSTALLED")
    results = [
        _report("standardize_columns", _check_standardize_columns),
        _report("least_squares", _check_least_squares),
    ]
    print(f"{sum(results)} of {len(results)} checks passed.")
    try:
        print(f"Figure saved to {make_figure()}")
    except Exception as e:  # noqa: BLE001
        print(f"Could not make the figure: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
