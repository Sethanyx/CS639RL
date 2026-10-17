"""CS 639, Fall 2026 -- Homework 1, Problems 3-4.

Complete the five functions marked TODO, then run

    python hw1.py

The script prints the results of the four algorithm functions on specified
inputs and selected rows from the Gridworld builder, then runs the
experiments of Problem 4 with your functions as they are.  It prints the
tables the write-up asks for and saves one figure, hw1_fig1.png.
A block that needs a function you have not written yet is skipped.  Paste the
complete printed output into Problem 3 of your PDF, fill in the tables of
Problem 4 from it, and include the figure where the handout asks.  This file
itself is not submitted.

Conventions, the same as in the notes.  A finite MDP is two arrays:

    P[s, a, s2] = P(s2 | s, a)      shape (n_states, n_actions, n_states)
    r[s, a]     = r(s, a)           shape (n_states, n_actions)

Terminal states are absorbing with reward 0 and are ordinary rows of P and r.
A deterministic policy is an integer array pi of shape (n_states,) with
pi[s] the action taken in s; a value function is a float array V of shape
(n_states,); Q[s, a] is a state-action table.  Value iteration computes every
entry of V_{k+1} from the whole of V_k (Lecture 5's box, not in place).  Only
NumPy may be used (Matplotlib
draws the figure; nothing there is yours to write).
"""
import sys

import numpy as np


# ------------------------------------------------------- given: the chain
def induced_chain(P, r, pi):
    """Return (P_pi, r_pi), the Markov chain with rewards left by the
    deterministic policy pi: P_pi[s, s2] = P(s2 | s, pi[s]) of shape
    (n_states, n_states) and r_pi[s] = r(s, pi[s]) of shape (n_states,).
    Given: policy_evaluation and optimality_test call it.
    """
    P = np.asarray(P, dtype=float)
    r = np.asarray(r, dtype=float)
    pi = np.asarray(pi, dtype=int)
    idx = np.arange(P.shape[0])
    return P[idx, pi, :], r[idx, pi]


# ------------------------------------------------ the four algorithm functions
# Lecture 3: the exact value of a policy.
# Lecture 4: the lookahead table and the greedy policy.
# Lecture 5: value iteration.


def policy_evaluation(P, r, gamma, pi):
    """Return V^pi, the exact value of the deterministic policy pi, by solving
    the linear system (I - gamma P_pi) V = r_pi of Lecture 3 with
    np.linalg.solve (not an explicit inverse), where (P_pi, r_pi) is the
    chain induced_chain(P, r, pi).  Requires gamma < 1.
    """
    I = np.eye(P.shape[0])

    P_pi, r_pi = induced_chain(P, r, pi)

    V_pi = np.linalg.solve(I - gamma*P_pi, r_pi)

    return V_pi


def q_from_v(P, r, gamma, V):
    """Return the lookahead table Q_V[s, a] = r(s, a) + gamma * sum_s2
    P(s2 | s, a) V[s2], of shape (n_states, n_actions).  With V = V^pi this
    is Q^pi; with V = V* it is Q*.  No Python loop is needed (hint: P @ V).
    """
    P = np.asarray(P, dtype=float)
    r = np.asarray(r, dtype=float)
    V = np.asarray(V, dtype=float)

    return r + gamma * P @ V


def greedy_policy(Q, tol=1e-9):
    """Return a deterministic policy greedy with respect to the table Q:
    pi[s] maximizes Q[s, a] over a.

    Ties: call an action tied at s if Q[s, a] is within tol of max_a Q[s, a],
    and take the smallest tied action index.  Do not rely on np.argmax alone:
    floating-point ties are real (the Gridworld has symmetric cells), and the
    tolerance decides them.
    """
    Q = np.asarray(Q, dtype=float)
    n_states = Q.shape[0]
    pi = np.zeros(n_states, dtype=int)

    for s in range(n_states):
        max = np.max(Q[s,:])
        for a in range(Q.shape[1]):
            if (Q[s,a] >= max - tol):
                pi[s] = a
                break
    return pi


def value_iteration(P, r, gamma, theta=1e-8, V0=None, max_iter=10**6):
    """Value iteration, Lecture 5's box.  Start from V0 (default: all zeros)
    and iterate V_{k+1}(s) = max_a [ r(s, a) + gamma sum_s2 P(s2 | s, a) V_k(s2) ],
    every entry of V_{k+1} from the whole of V_k, until the first k with
    max_s |V_k(s) - V_{k-1}(s)| <= theta.

    Return (V, iterates) where V = V_k is the last iterate and iterates is
    the list [V_0, V_1, ..., V_k], so that len(iterates) - 1 iterations were
    made.  Raise RuntimeError if max_iter iterations do not reach theta.
    """
    n_states = np.asarray(P).shape[0]
    V = np.zeros(n_states) if V0 is None else np.array(V0, dtype=float)
    iterates = [V.copy()]
    Q = q_from_v(P, r, gamma, V)
    V1 = np.max(Q, axis = 1)
    iterates.append(V1.copy())
    k = 1
    while (np.max(np.abs(V-V1)) > theta and k < max_iter):
        V = iterates[-1]
        Q = q_from_v(P, r, gamma, V)
        V1 = np.max(Q, axis = 1)
        iterates.append(V1.copy())
        k = k + 1
    if (k > max_iter):
        raise RuntimeError
    return V1, iterates

# -------------------------------------------------------- given: the test
def optimality_test(P, r, gamma, pi):
    """Lecture 4's test: pi is optimal iff V^pi(s) = max_a Q^pi(s, a) for
    every s.  Given, built from your policy_evaluation and q_from_v: returns
    (V, gap) with V = V^pi and gap[s] = max_a Q^pi(s, a) - V^pi(s) >= 0, the
    amount by which the best one-step deviation at s beats the policy's own
    action; pi passes the test at s iff gap[s] = 0 (up to rounding).
    """
    V = policy_evaluation(P, r, gamma, pi)
    gap = q_from_v(P, r, gamma, V).max(axis=1) - V
    return V, gap


# ------------------------------------------------------------ models given
def machine():
    """The three-state machine of Lectures 3-5: states G, W, B; actions run (0)
    and repair (1); run earns 8, 4, 0 and wears the machine down one level
    with probability 1/3; repair costs 10 and restores it to G.  The
    report uses it at gamma = 0.8 for policy evaluation, Q_V, and value iteration."""
    P = np.zeros((3, 2, 3))
    P[:, 0, :] = [[2 / 3, 1 / 3, 0.0], [0.0, 2 / 3, 1 / 3], [0.0, 0.0, 1.0]]
    P[:, 1, 0] = 1.0
    r = np.array([[8.0, -10.0], [4.0, -10.0], [0.0, -10.0]])
    return P, r


def channel():
    """The channel of Problem 2: states trending (0) and obscure (1); actions
    rush (0) and polish (1).  r[s, a] is the expected weekly reward: rushing
    while trending pays 9 if the channel stays trending (probability 1/3) and
    6 if it drops (2/3), so r = 7; polishing while obscure pays 3 if the
    channel starts trending (1/3) and 0 otherwise, so r = 1.  Use it to check
    your hand computations of Problem 2."""
    P = np.zeros((2, 2, 2))
    P[0, 0] = [1 / 3, 2 / 3]
    P[0, 1] = [1.0, 0.0]
    P[1, 0] = [0.0, 1.0]
    P[1, 1] = [1 / 3, 2 / 3]
    r = np.array([[7.0, 3.0], [2.0, 1.0]])
    return P, r


# ----------------------------------------------------- the model you build
# The Gridworld of Lecture 2 (Russell and Norvig, Section 17.1): cells (x, y)
# with x = 1..4 and y = 1..3, wall at (2, 2), exits +1 at (4, 3) and -1 at
# (4, 2), start at (1, 1).  The geometry is given below; you write gridworld().
GW_CELLS = [(x, y) for y in (3, 2, 1) for x in (1, 2, 3, 4) if (x, y) != (2, 2)]
GW_INDEX = {c: i for i, c in enumerate(GW_CELLS)}  # cell -> state index
GW_EXITS = {(4, 3): 1.0, (4, 2): -1.0}  # exit cell -> the reward for entering it
GW_ACTIONS = ["N", "S", "E", "W"]  # action index 0, 1, 2, 3
GW_START = GW_INDEX[(1, 1)]
GW_DECISION = [GW_INDEX[c] for c in GW_CELLS if c not in GW_EXITS]  # the nine non-exit cells
GW_RMAX = 1.04  # the largest |R_{t+1}|: entering the -1 exit at step reward -0.04
_MOVE = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}
_SIDE = {"N": ("E", "W"), "S": ("E", "W"), "E": ("N", "S"), "W": ("N", "S")}  # the two slip directions
_ARROW = {"N": "^", "S": "v", "E": ">", "W": "<"}


def _gw_step(cell, direction):
    """The cell reached by moving one step from cell in the given direction;
    a move into the wall or off the grid leaves the position unchanged."""
    nxt = (cell[0] + _MOVE[direction][0], cell[1] + _MOVE[direction][1])
    return nxt if nxt in GW_INDEX else cell


def gridworld(step_reward=-0.04):
    """Return (P, r) for the eleven-state Gridworld with the given step reward,
    states indexed by GW_INDEX and actions by GW_ACTIONS.

    From a non-exit cell, action a moves in direction a with probability 0.8
    and in each of the two perpendicular directions _SIDE[a] with probability
    0.1 (use _gw_step for where a move lands; a move into the wall or the
    border stays put, so two of the three outcomes may land on the same cell:
    add their probabilities).  The reward R_{t+1} of a move is step_reward plus
    GW_EXITS[c2] if the cell c2 it lands on is an exit; r[s, a] is its
    expectation over the three outcomes.  The exit cells are terminal:
    absorbing under every action with reward 0.
    """
    n = len(GW_CELLS)
    P = np.zeros((n, 4, n))
    r = np.zeros((n, 4))

    for cell in GW_CELLS:
        s = GW_INDEX[cell]

        if cell in GW_EXITS:
            for a in range(4):
                P[s, a, s] = 1
                r[s, a] = 0
            continue

        for a in GW_ACTIONS:
            s1 = s
            s2 = s
            s3 = s
            next_cell1 = _gw_step(cell, a)
            if (next_cell1 in GW_CELLS):
                s1 = GW_INDEX[next_cell1]
            side1, side2 = _SIDE[a]
            next_cell2 = _gw_step(cell, side1)
            if (next_cell2 in GW_CELLS):
                s2 = GW_INDEX[next_cell2]
            next_cell3 = _gw_step(cell, side2)
            if (next_cell3 in GW_CELLS):
                s3 = GW_INDEX[next_cell3]
            a_index = GW_ACTIONS.index(a)
            P[s, a_index, s1] += 0.8
            P[s, a_index, s2] += 0.1
            P[s, a_index, s3] += 0.1
            reward1 = step_reward
            reward2 = step_reward
            reward3 = step_reward
            if (next_cell1 in GW_EXITS):
                reward1 += GW_EXITS[next_cell1]
            if (next_cell2 in GW_EXITS):
                reward2 += GW_EXITS[next_cell2]
            if (next_cell3 in GW_EXITS):
                reward3 += GW_EXITS[next_cell3]
            r[s, a_index] = 0.8 * reward1 + 0.1 * reward2 + 0.1 * reward3

    return P, r


def gw_values(V, fmt="6.3f"):
    """The 3x4 map of V as text, top row (y = 3) first."""
    rows = []
    for y in (3, 2, 1):
        rows.append(" ".join(f"{V[GW_INDEX[(x, y)]]:{fmt}}" if (x, y) in GW_INDEX else "  wall"
                             for x in (1, 2, 3, 4)))
    return "\n".join("    " + row for row in rows)


def gw_policy(pi):
    """The 3x4 map of a policy as arrows; T marks an exit, # the wall."""
    rows = []
    for y in (3, 2, 1):
        rows.append(" ".join(
            "#" if (x, y) not in GW_INDEX else "T" if (x, y) in GW_EXITS
            else _ARROW[GW_ACTIONS[pi[GW_INDEX[(x, y)]]]] for x in (1, 2, 3, 4)))
    return "\n".join("    " + row for row in rows)


def gw_arrows(pi):
    """The policy on the nine decision cells as one string, reading order."""
    return " ".join(_ARROW[GW_ACTIONS[pi[i]]] for i in GW_DECISION)


# ---------------------------------------------------------------- experiments
# Nothing below this line is yours to write.
def optimal_values(P, r, gamma):
    """(V*, pi*): value iteration to a tight tolerance and the greedy policy
    of the last iterate."""
    V, _ = value_iteration(P, r, gamma, theta=1e-13)
    return V, greedy_policy(q_from_v(P, r, gamma, V))


def loss_of(P, r, gamma, pi, V_star):
    """max_s [V*(s) - V^pi(s)]: what following pi instead of pi* costs."""
    return float(np.max(V_star - policy_evaluation(P, r, gamma, pi)))


def summarize(P, r, gamma, theta):
    """Everything the table of Problem 4(b) reports for one discount."""
    V_star, pi_star = optimal_values(P, r, gamma)
    V_K, iterates = value_iteration(P, r, gamma, theta=theta)
    errors = [float(np.max(np.abs(Vk - V_star))) for Vk in iterates]
    try:
        losses = [loss_of(P, r, gamma, greedy_policy(q_from_v(P, r, gamma, Vk)), V_star) for Vk in iterates]
    except Exception:  # A broken policy evaluation must not suppress the VI results or figure.
        losses = []
    return dict(
        V_star=V_star, pi_star=pi_star, iterates=iterates, errors=errors, losses=losses,
        K=len(iterates) - 1,
        k_opt=next((k for k, l in enumerate(losses) if l <= 1e-9), "N/A"),
        late_ratio=errors[-1] / errors[-2],
    )


def _clean(V, decimals=3):
    """Round for printing and turn -0.000 into 0.000."""
    return np.round(np.asarray(V, dtype=float), decimals) + 0.0


def side_by_side(maps, labels, width=22):
    rows = [m.split("\n") for m in maps]
    head = "    " + "".join(f"{lab:<{width}}" for lab in labels)
    body = ["    " + "".join(f"{row[i].strip():<{width}}" for row in rows) for i in range(len(rows[0]))]
    return "\n".join(line.rstrip() for line in [head] + body)


def experiment_4a():
    """Gridworld: evaluate 'always N' exactly and put it through the test."""
    P, r = gridworld()
    always_n = np.full(P.shape[0], GW_ACTIONS.index("N"), dtype=int)
    V_n, gap_n = optimality_test(P, r, 0.9, always_n)
    out = ["4(a) Gridworld, gamma = 0.9, pi_N = 'always N': policy evaluation and the optimality test",
           f"    V^(pi_N) by the linear solve (row y = 3 first); V^(pi_N)((1,1)) = {V_n[GW_START]:.4f}",
           gw_values(_clean(V_n)),
           "    the test: gap(s) = max_a Q^(pi_N)(s,a) - V^(pi_N)(s), 0.000 where pi_N passes",
           gw_values(_clean(gap_n))]
    fails = [c for c in GW_CELLS if c not in GW_EXITS and gap_n[GW_INDEX[c]] > 1e-9]
    if fails:
        worst = max(fails, key=lambda c: gap_n[GW_INDEX[c]])
        out.append(f"    pi_N fails in {len(fails)} of the 9 decision cells; the largest gap, "
                   f"{gap_n[GW_INDEX[worst]]:.3f}, is at {worst}")
    else:
        out.append("    pi_N passes the test in every decision cell")
    return out


def experiment_4b(theta, curves):
    """Gridworld value iteration at three discounts; fills curves for the figure."""
    P, r = gridworld()
    out = [f"4(b) Gridworld value iteration, step -0.04, theta = {theta:g}",
           "    K: iterations until the change |V_k - V_{k-1}| is <= theta.   k(pi*): first k whose",
           "    greedy policy is optimal.   ratio: |V_K - V*| / |V_{K-1} - V*|, the factor by which",
           "    the error shrinks in the last iteration.",
           "    gamma    K    k(pi*)  V*((1,1))   ratio"]
    S09 = None
    for gamma in (0.5, 0.9, 0.99):
        S = summarize(P, r, gamma, theta)
        out.append(f"    {gamma:<6}   {S['K']:<4} {S['k_opt']:<7} {S['V_star'][GW_START]:<11.4f} {S['late_ratio']:.3f}")
        if S["k_opt"] == "N/A":
            out.append("    k(pi*) unavailable: policy evaluation failed or no optimal greedy policy was found.")
        curves[gamma] = (S["errors"], float(np.max(np.abs(S["V_star"]))))
        if gamma == 0.9:
            S09 = S
    out.append("    gamma = 0.9: V* (row y = 3 first) and the optimal policy")
    out.append(side_by_side([gw_values(S09["V_star"]), gw_policy(S09["pi_star"])], ["V*:", "pi*:"], width=34))
    return out


def experiment_4c():
    """Gridworld at gamma = 0.9 with three step rewards."""
    out = ["4(c) Gridworld, gamma = 0.9, three step rewards: V*((1,1)) and the optimal policy"]
    maps, labels = [], []
    for sr in (-0.04, -2.0, 0.1):
        P2, r2 = gridworld(step_reward=sr)
        V2, pi2 = optimal_values(P2, r2, 0.9)
        maps.append(gw_policy(pi2))
        labels.append(f"step {sr:+.2f}: {V2[GW_START]:.4f}")
    out.append(side_by_side(maps, labels, width=26))
    return out


def save_figure(curves, path):
    """Figure 1: the error of value iteration against the iteration, and gamma^k |V*| dashed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    for gamma, (errs, norm) in curves.items():
        k = np.arange(len(errs))
        line, = ax.semilogy(k, errs, linewidth=1.2, label=rf"$\gamma={gamma}$", zorder=3)
        ax.semilogy(k, norm * gamma ** k, linestyle="--", linewidth=2.2, alpha=0.45,
                    color=line.get_color(), zorder=2)
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel(r"$\|V_k - V^\star\|_\infty$")
    ax.set_title(r"Gridworld, step reward $-0.04$ (dashed: $\gamma^k\|V^\star\|_\infty$)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _block(name, fn):
    """Run one experiment block; print its lines, or one line saying why it was skipped."""
    try:
        lines = fn()
    except NotImplementedError:
        print(f"[SKIPPED] {name}: a function it needs is not implemented yet")
        return
    except Exception as e:  # noqa: BLE001
        print(f"[SKIPPED] {name}: raised {type(e).__name__}: {e}")
        return
    print("\n".join(lines))


def run_experiments(theta=1e-8, figure="hw1_fig1.png"):
    """Problem 4: print the three blocks, each with the functions as they are; save the figure."""
    curves = {}
    _block("4(a)", experiment_4a)
    _block("4(b)", lambda: experiment_4b(theta, curves))
    _block("4(c)", experiment_4c)
    if not curves:
        print(f"Figure not saved: 4(b) did not run.")
        return
    try:
        save_figure(curves, figure)
        print(f"Figure saved: {figure}")
    except Exception as e:  # noqa: BLE001
        print(f"Could not make the figure: {type(e).__name__}: {e}")


# --------------------------------------------------------- Problem 3 outputs
def _report(name, fn):
    """Print the result returned by fn()."""
    try:
        detail = fn()
    except NotImplementedError:
        print(f"[TODO] {name}: not implemented yet")
        return
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] {name}: raised {type(e).__name__}: {e}")
        return
    tag = f"[RESULT] {name}: "
    if len(tag) + len(detail) <= 92 or "; " not in detail:
        print(tag + detail)
    else:
        pieces = detail.split("; ")
        print(tag + pieces[0] + ";")
        for piece in pieces[1:-1]:
            print(" " * len(tag) + piece + ";")
        print(" " * len(tag) + pieces[-1])


NEVER_REPAIR = np.array([0, 0, 0])
REPAIR_WHEN_BROKEN = np.array([0, 0, 1])


def _result_policy_evaluation():
    P, r = machine()
    V = policy_evaluation(P, r, 0.8, NEVER_REPAIR)
    V2 = policy_evaluation(P, r, 0.8, REPAIR_WHEN_BROKEN)
    return (f"machine, gamma = 0.8; never repair: V = {_clean(V, 4).tolist()}; "
            f"repair when broken: V = {_clean(V2, 4).tolist()}")


def _result_q_from_v():
    P, r = machine()
    Q = q_from_v(P, r, 0.8, np.array([12.0, -3.0, 2.0]))
    return (f"machine, gamma = 0.8, V = [12.0, -3.0, 2.0]; "
            f"Q(., run) = {_clean(Q[:, 0], 4).tolist()}; "
            f"Q(., repair) = {_clean(Q[:, 1], 4).tolist()}")


def _result_greedy_policy():
    Q = np.array([[-2.0, 1.0], [3.0, 0.0], [0.0, 4.0]])
    pi = greedy_policy(Q)
    Qt = np.array([[2.0, 2.0 + 5e-10], [2.0, 2.0 + 2e-9], [3.0, 3.0]])
    pi_t = greedy_policy(Qt)
    return f"Q: pi = {np.asarray(pi).tolist()}; Qt (tol = 1e-9): pi = {np.asarray(pi_t).tolist()}"


def _result_value_iteration():
    P, r = machine()
    V, iterates = value_iteration(P, r, 0.8, theta=1e-10, V0=[1.0, 2.0, 3.0])
    return (f"machine, gamma = 0.8, V_0 = [1.0, 2.0, 3.0]; "
            f"V_1 = {_clean(iterates[1], 3).tolist()}, V_2 = {_clean(iterates[2], 3).tolist()}; "
            f"V_3 = {_clean(iterates[3], 3).tolist()}; V_K = {_clean(V, 4).tolist()}; "
            f"K = {len(iterates) - 1} iterations to theta = 1e-10")


def _result_gridworld():
    P, r = gridworld(step_reward=-0.2)
    cases = [((2, 1), "E"), ((3, 2), "N"), ((3, 3), "S")]
    detail = ["step -0.2, P rows in GW_CELLS order"]
    for cell, action in cases:
        i, a = GW_INDEX[cell], GW_ACTIONS.index(action)
        detail.append(f"P({cell},{action})={_clean(P[i, a]).tolist()}, r={r[i, a]:.2f}")
    for c in GW_EXITS:
        i = GW_INDEX[c]
        detail.append(f"exit {c}: P(s,a,s)={_clean(P[i, :, i]).tolist()}, r={_clean(r[i]).tolist()}")
    P2, r2 = gridworld(step_reward=0.1)
    rewards = [r2[GW_INDEX[c], GW_ACTIONS.index(a)] for c, a in cases]
    detail.append(f"step +0.1: rewards for these rows = {_clean(rewards).tolist()}")
    detail.append(f"max |P(+0.1) - P(-0.2)| = {np.max(np.abs(P2 - P)):.3g}")
    return "; ".join(detail)


def main():
    print(f"Python {sys.version.split()[0]}, NumPy {np.__version__}")
    _report("policy_evaluation", _result_policy_evaluation)
    _report("q_from_v", _result_q_from_v)
    _report("greedy_policy", _result_greedy_policy)
    _report("value_iteration", _result_value_iteration)
    _report("gridworld", _result_gridworld)
    run_experiments()


if __name__ == "__main__":
    main()
