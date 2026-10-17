"""CS 639, Fall 2026 -- Homework 2, Problems 3-4.

Complete the five functions marked TODO, then run

    python hw2.py

The script runs each of your functions on one small input and draws the
results as the two figures of Problem 3, hw2_fig1.png and hw2_fig2.png (a
panel whose function you have not written yet says so); then it runs the
experiments of Problem 4 with your functions as they are and saves the three
figures of Problem 4, hw2_fig3.png to hw2_fig5.png (a block that needs an
unwritten function is skipped, and so is its figure).  It also prints every
number behind the figures, for your own checking; nothing printed goes into
the report.  Include each figure under its part of your PDF.  Problem 4(c)
also asks you to choose two step sizes, YOUR_ALPHA_TD and YOUR_ALPHA_MC,
right below the imports.  This file itself is not submitted.

Conventions, the same as in Homework 1 and the notes.  A finite MDP is two
arrays:

    P[s, a, s2] = P(s2 | s, a)      shape (n_states, n_actions, n_states)
    r[s, a]     = r(s, a)           shape (n_states, n_actions)

Terminal states are absorbing with reward 0 and are ordinary rows of P and r.
A deterministic policy is an integer array pi of shape (n_states,) with
pi[s] the action taken in s; a value function is a float array V of shape
(n_states,); Q[s, a] is a state-action table.  An "iteration" is one
application of T, every entry computed from the whole of the previous table;
a "round" of policy iteration is one evaluation followed by one improvement;
"evaluations" counts the linear solves, the policy stopped at included.  An
episode (Problems 3(b) and 4(c)) is a list of transitions (s, r, s2) in time order,
with s2 = None when the episode ends in a terminal state.  Only NumPy may be
used (Matplotlib draws the figures; nothing there is yours to write).
"""
import sys
import textwrap

import numpy as np


# ------------------------------------------------ Problem 4(c): your step sizes
# Figure 5 compares TD(0) at alpha = 0.05, 0.10, 0.15 with Monte Carlo at
# alpha = 0.01, 0.02, 0.03, 0.04.  After a first look at it, add one step size
# of your own for each method, any value in (0, 0.5] other than that method's
# given ones; the figure and the printed table then carry your two settings
# too, starred.  Until you set them, the seven given settings are shown.
YOUR_ALPHA_TD = 0.2  # TODO: your choice, any value in (0, 0.5]
YOUR_ALPHA_MC = 0.05  # TODO: your choice, any value in (0, 0.5]


# -------------------------------------------- given: the functions of Homework 1
def induced_chain(P, r, pi):
    """Return (P_pi, r_pi), the Markov chain with rewards left by the
    deterministic policy pi: P_pi[s, s2] = P(s2 | s, pi[s]) of shape
    (n_states, n_states) and r_pi[s] = r(s, pi[s]) of shape (n_states,)."""
    P = np.asarray(P, dtype=float)
    r = np.asarray(r, dtype=float)
    pi = np.asarray(pi, dtype=int)
    idx = np.arange(P.shape[0])
    return P[idx, pi, :], r[idx, pi]


def policy_evaluation(P, r, gamma, pi):
    """V^pi, the exact value of the deterministic policy pi, by the linear
    solve (I - gamma P_pi) V = r_pi of Lecture 3.  Requires gamma < 1."""
    P_pi, r_pi = induced_chain(P, r, pi)
    return np.linalg.solve(np.eye(len(r_pi)) - gamma * P_pi, r_pi)


def q_from_v(P, r, gamma, V):
    """The lookahead table Q_V[s, a] = r(s, a) + gamma * sum_s2 P(s2 | s, a) V[s2]."""
    P = np.asarray(P, dtype=float)
    r = np.asarray(r, dtype=float)
    V = np.asarray(V, dtype=float)
    return r + gamma * P @ V


def greedy_policy(Q, tol=1e-9):
    """A deterministic policy greedy with respect to Q; ties (entries within
    tol of the row maximum) go to the smallest action index."""
    Q = np.asarray(Q, dtype=float)
    n_states = Q.shape[0]
    pi = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        pi[s] = np.flatnonzero(Q[s] >= Q[s].max() - tol)[0]
    return pi


def value_iteration(P, r, gamma, theta=1e-8, V0=None, max_iter=10**6):
    """Value iteration, Lecture 5's box, from V0 (default zeros) until the
    first k with max_s |V_k(s) - V_{k-1}(s)| <= theta.  Returns (V_k, the
    list [V_0, ..., V_k]); len(iterates) - 1 iterations were made."""
    n_states = np.asarray(P).shape[0]
    V = np.zeros(n_states) if V0 is None else np.array(V0, dtype=float)
    iterates = [V.copy()]
    for _ in range(max_iter):
        V_new = q_from_v(P, r, gamma, V).max(axis=1)
        iterates.append(V_new.copy())
        if np.max(np.abs(V_new - V)) <= theta:
            return V_new, iterates
        V = V_new
    raise RuntimeError(f"value iteration did not reach theta={theta} in {max_iter} iterations")


def optimality_test(P, r, gamma, pi):
    """Lecture 4's test: (V^pi, gap) with gap[s] = max_a Q^pi(s, a) - V^pi(s) >= 0;
    pi passes at s iff gap[s] = 0 (up to rounding)."""
    V = policy_evaluation(P, r, gamma, pi)
    gap = q_from_v(P, r, gamma, V).max(axis=1) - V
    return V, gap


# --------------------------------------------------- the five functions you write
# Lecture 6: the improvement step with the tie rule, and policy iteration.
# Lecture 8: certainty equivalence.
# Lecture 9: the template with two targets, TD(0) and Monte Carlo.


def improve(Q, pi_old, tol=1e-9):
    """Lecture 6's improvement step with the tie rule.  Return the
    deterministic policy pi_new with pi_new[s] = argmax_a Q[s, a], except
    that pi_new[s] = pi_old[s] whenever Q[s, pi_old[s]] is within tol of
    max_a Q[s, a] (the current action is kept when it is among the
    maximizers); other ties go to the smallest action index, as in
    greedy_policy.  Do not modify pi_old.
    """
    Q = np.asarray(Q, dtype=float)
    pi_old = np.asarray(pi_old, dtype=int)
    # get the greedy action, however, it will make pi_new choose the smallest action index
    pi_new = greedy_policy(Q, tol)
    # if the old_action is already greedy, no change
    for s in range(Q.shape[0]):
        old_action = pi_old[s]
        max = np.max(Q[s])
        # restore the action
        if (max - Q[s, old_action] <= tol):
            pi_new[s] = old_action
    
    return pi_new



def policy_iteration(P, r, gamma, pi0, tol=1e-9, max_rounds=10**4):
    """Policy iteration, Lecture 6's box: from the deterministic policy pi0,
    evaluate exactly (policy_evaluation), form Q^pi = q_from_v(P, r, gamma,
    V^pi), improve with the tie rule (improve), and stop when the policy is
    unchanged.

    Return (policies, values): policies = [pi_0, pi_1, ..., pi_K] are the
    policies evaluated, each different from the one before, with pi_K the
    policy returned (the improvement step left it unchanged), and values =
    [V^{pi_0}, ..., V^{pi_K}] their exact values.  len(policies) is the
    number of evaluations.  Raise RuntimeError after max_rounds rounds.
    """
    pi = np.array(pi0, dtype=int)
    policies, values = [pi.copy()], []

    for i in range(max_rounds):
        V = policy_evaluation(P, r, gamma, pi)
        values.append(V.copy())

        Q_pi = q_from_v(P, r, gamma, V)
        pi_new = improve(Q_pi, pi, tol)

        if np.array_equal(pi_new, pi):
            return policies, values
        
        pi = pi_new
        policies.append(pi.copy())

    raise RuntimeError(
        f"policy iteration did not converge in {max_rounds} rounds"
    )


def certainty_equivalence(counts, reward_sums, gamma, pi0):
    """Certainty equivalence, Lecture 8's box.  counts[s, a, s2] is the
    number of sampled transitions from (s, a) that landed in s2 and
    reward_sums[s, a] the sum of the sampled rewards at (s, a); every pair
    has at least one sample.  Count and average: P_hat[s, a, s2] =
    counts[s, a, s2] / n(s, a) and r_hat[s, a] = reward_sums[s, a] / n(s, a)
    with n(s, a) = sum_s2 counts[s, a, s2].  Then plan in the estimate: run
    policy_iteration in (P_hat, r_hat, gamma) from pi0.

    Return (P_hat, r_hat, pi_hat, V_hat): the estimate, its optimal policy,
    and that policy's value in the estimate (the last entries of
    policy_iteration's lists).
    """
    counts = np.asarray(counts, dtype=float)
    reward_sums = np.asarray(reward_sums, dtype=float)

    n = counts.sum(axis = 2)
    P_hat = counts.copy()

    for s in range(n.shape[0]):
        for a in range(n.shape[1]):
            if (n[s,a] != 0):
                P_hat[s,a,:] = P_hat[s,a,:] / n[s,a]

    r_hat = reward_sums / n

    policies, values = policy_iteration(P_hat, r_hat, gamma, pi0)

    pi_hat = policies[-1]
    V_hat = values[-1]

    return P_hat, r_hat, pi_hat, V_hat


def td0_prediction(V, episode, gamma, alpha):
    """TD(0), Lecture 9's box, over one episode: for each transition
    (s, r, s2) in time order, target = r + gamma * V[s2] (the target is r
    alone when s2 is None, the terminal state being worth 0), then
    V[s] <- V[s] + alpha * (target - V[s]).  Update V in place and return it.
    """
    for epi in episode:
        target = epi[1]
        if (epi[2] is not None):
            target += gamma * V[epi[2]]
        V[epi[0]] += alpha * (target - V[epi[0]])

    return V

def g_t(t, n, episode, gamma):
    if t == n:
        return 0

    reward = episode[t][1]
    return reward + gamma * g_t(t + 1, n, episode, gamma)


def mc_prediction(V, episode, gamma, alpha):
    """Monte Carlo with the same template, over one complete episode: first
    compute the return that followed each transition, last transition first
    (G_T = 0, G_t = R_{t+1} + gamma G_{t+1}); then for t = 0, 1, ..., T - 1
    in time order, V[s_t] <- V[s_t] + alpha * (G_t - V[s_t]), every visit
    updated.  Update V in place and return it.  The episode must end in a
    terminal state (its last s2 is None).
    """
    for i in range(len(episode)):
        target = g_t(i, len(episode), episode, gamma)
        V[episode[i][0]] += alpha * (target - V[episode[i][0]])

    return V


# ------------------------------------------------------------ models given
def machine():
    """The three-state machine of Lectures 3-9: states G, W, B; actions run (0)
    and repair (1); run earns 8, 4, 0 and wears the machine down one level
    with probability 1/3; repair costs 10 and restores it to G."""
    P = np.zeros((3, 2, 3))
    P[:, 0, :] = [[2 / 3, 1 / 3, 0.0], [0.0, 2 / 3, 1 / 3], [0.0, 0.0, 1.0]]
    P[:, 1, 0] = 1.0
    r = np.array([[8.0, -10.0], [4.0, -10.0], [0.0, -10.0]])
    return P, r


def pond():
    """The pond of Problem 1: states full (0) and low (1); actions fish (0)
    and wait (1).  Fishing a full pond pays 8 and leaves it full with
    probability 1/3; fishing a low pond pays 1 and keeps it low; waiting pays
    0, keeps a full pond full, and lets a low pond recover with probability
    1/3.  Use it to check your hand computations of Problem 1."""
    P = np.zeros((2, 2, 2))
    P[0, 0] = [1 / 3, 2 / 3]
    P[0, 1] = [1.0, 0.0]
    P[1, 0] = [0.0, 1.0]
    P[1, 1] = [1 / 3, 2 / 3]
    r = np.array([[8.0, 0.0], [1.0, 0.0]])
    return P, r


def two_state():
    """The two-state example of Problem 2(b): in state 0, stay (0) pays 0 and
    stays, move (1) pays 0 and goes to state 1; in state 1 both actions pay 2
    and stay.  Use it to check your hand computations."""
    P = np.zeros((2, 2, 2))
    P[0, 0, 0] = 1.0
    P[0, 1, 1] = 1.0
    P[1, :, 1] = 1.0
    r = np.array([[0.0, 0.0], [2.0, 2.0]])
    return P, r


# RiverSwim (Strehl and Littman, 2008): six states in a row, s1 at the left
# bank and s6 at the right bank; actions left (0) and right (1).  Swimming left
# always moves one state left (and stays at s1).  Swimming right fights the
# current: from s2 to s5 it stays put w.p. 0.6, succeeds w.p. p, and is pushed
# back w.p. 0.4 - p; from s1 it succeeds w.p. p and stays otherwise; from s6
# it stays w.p. 1 - (0.4 - p) and is pushed back otherwise.  Rewards: 0.005
# for swimming left at s1, 1 for swimming right at s6, 0 elsewhere; R_max = 1.
# The current sets p: weak 0.30, medium 0.15, strong 0.05.
RS_CURRENT = {"weak": 0.30, "medium": 0.15, "strong": 0.05}
RS_RMAX = 1.0


def riverswim(current="weak", n_states=6):
    """Return (P, r) for RiverSwim with the given current."""
    p = RS_CURRENT[current]
    back = 0.4 - p
    P = np.zeros((n_states, 2, n_states))
    r = np.zeros((n_states, 2))
    for s in range(n_states):
        P[s, 0, max(s - 1, 0)] = 1.0
        if s == 0:
            P[s, 1, 1] += p
            P[s, 1, 0] += 1.0 - p
        elif s == n_states - 1:
            P[s, 1, s] += 1.0 - back
            P[s, 1, s - 1] += back
        else:
            P[s, 1, s + 1] += p
            P[s, 1, s] += 0.6
            P[s, 1, s - 1] += back
    r[0, 0] = 0.005
    r[n_states - 1, 1] = 1.0
    return P, r


def rs_policy(pi):
    """A RiverSwim policy as a string, s1 to s6, L = swim left, R = swim right."""
    return "".join("LR"[a] for a in pi)


# The Gridworld of Homework 1 (Russell and Norvig, Section 17.1), given here
# in full: cells (x, y) with x = 1..4 and y = 1..3, wall at (2, 2), exits +1 at
# (4, 3) and -1 at (4, 2), start at (1, 1); slip 0.8 / 0.1 / 0.1, bumps stay put.
GW_CELLS = [(x, y) for y in (3, 2, 1) for x in (1, 2, 3, 4) if (x, y) != (2, 2)]
GW_INDEX = {c: i for i, c in enumerate(GW_CELLS)}
GW_EXITS = {(4, 3): 1.0, (4, 2): -1.0}
GW_ACTIONS = ["N", "S", "E", "W"]
GW_START = GW_INDEX[(1, 1)]
GW_DECISION = [GW_INDEX[c] for c in GW_CELLS if c not in GW_EXITS]
_MOVE = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}
_SIDE = {"N": ("E", "W"), "S": ("E", "W"), "E": ("N", "S"), "W": ("N", "S")}
_ARROW = {"N": "^", "S": "v", "E": ">", "W": "<"}


def _gw_step(cell, direction):
    nxt = (cell[0] + _MOVE[direction][0], cell[1] + _MOVE[direction][1])
    return nxt if nxt in GW_INDEX else cell


def gridworld(step_reward=-0.04):
    """Return (P, r) for the eleven-state Gridworld with the given step reward."""
    n = len(GW_CELLS)
    P = np.zeros((n, 4, n))
    r = np.zeros((n, 4))
    for c in GW_CELLS:
        i = GW_INDEX[c]
        for ai, a in enumerate(GW_ACTIONS):
            if c in GW_EXITS:
                P[i, ai, i] = 1.0
                continue
            for d, prob in ((a, 0.8), (_SIDE[a][0], 0.1), (_SIDE[a][1], 0.1)):
                c2 = _gw_step(c, d)
                P[i, ai, GW_INDEX[c2]] += prob
                r[i, ai] += prob * (step_reward + GW_EXITS.get(c2, 0.0))
    return P, r


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


# The random walk of Sutton and Barto, Example 6.2: five states A..E (0..4) in
# a row, start at C; each step moves left or right with probability 1/2; the
# episode ends when it leaves either end, with reward 1 off the right end and
# 0 otherwise; gamma = 1.  The true values are 1/6, ..., 5/6.
RW_STATES = ["A", "B", "C", "D", "E"]
RW_START = 2
RW_TRUE = np.arange(1, 6) / 6.0


def rw_episode(rng):
    """One episode of the random walk as a list of transitions (s, r, s2),
    s2 = None at the end."""
    s, episode = RW_START, []
    while True:
        s2 = s + (1 if rng.random() < 0.5 else -1)
        if s2 < 0:
            episode.append((s, 0.0, None))
            return episode
        if s2 >= len(RW_STATES):
            episode.append((s, 1.0, None))
            return episode
        episode.append((s, 0.0, s2))
        s = s2


def sample_counts(P, r, n, rng):
    """The sampling oracle of Lecture 8: n samples (R, S') at every pair,
    returned as counts[s, a, s2] and reward_sums[s, a].  The reward of a pair
    is its mean r(s, a) (exact on the machine, whose profits repeat)."""
    P = np.asarray(P, dtype=float)
    r = np.asarray(r, dtype=float)
    counts = np.zeros(P.shape)
    for s in range(P.shape[0]):
        for a in range(P.shape[1]):
            counts[s, a] = rng.multinomial(n, P[s, a])
    return counts, n * r


# ---------------------------------------------------------------- experiments
# Nothing below this line is yours to write.
def optimal_values(P, r, gamma):
    """(V*, pi*): value iteration to a tight tolerance and the greedy policy
    of the last iterate."""
    V, _ = value_iteration(P, r, gamma, theta=1e-13)
    return V, greedy_policy(q_from_v(P, r, gamma, V))


def loss_of(P, r, gamma, pi, V_star):
    """max_s [V*(s) - V^pi(s)]: what following pi instead of pi* costs."""
    return max(0.0, float(np.max(V_star - policy_evaluation(P, r, gamma, pi))))


def _clean(V, decimals=3):
    """Round for printing and turn -0.000 into 0.000."""
    return np.round(np.asarray(V, dtype=float), decimals) + 0.0


def _fmt(V, decimals=2):
    return "(" + ", ".join(f"{v:.{decimals}f}" for v in _clean(V, decimals)) + ")"


def pi_vs_vi(P, r, gamma, pi0, theta, name_of):
    """One model of Problem 4(a): policy iteration from pi0 against value iteration from zero.
    Returns the policies visited with their losses and, for every iterate V_k of value
    iteration, the error ||V* - V_k|| and the loss of the greedy policy of V_k."""
    V_star, pi_star = optimal_values(P, r, gamma)
    policies, values = policy_iteration(P, r, gamma, pi0)
    losses = [max(0.0, float(np.max(V_star - V))) for V in values]
    V_K, iterates = value_iteration(P, r, gamma, theta=theta)
    errors = [float(np.max(np.abs(V_star - Vk))) for Vk in iterates]
    greedy_losses = [loss_of(P, r, gamma, greedy_policy(q_from_v(P, r, gamma, Vk)), V_star) for Vk in iterates]
    k_opt = next(k for k, loss in enumerate(greedy_losses) if loss <= 1e-9)
    return dict(gamma=gamma, V_star=V_star, pi_star=pi_star, policies=policies, losses=losses,
                errors=errors, greedy_losses=greedy_losses, K=len(iterates) - 1, k_opt=k_opt,
                names=[name_of(p) for p in policies])


def experiment_4a(theta=1e-8, curves=None):
    out = [f"4(a) Policy iteration against value iteration (theta = {theta:g})",
           "    K: VI iterations to its stop.   k(pi*): the first k whose greedy policy of V_k is optimal."]
    P, r = gridworld()
    always_n = np.full(P.shape[0], GW_ACTIONS.index("N"), dtype=int)
    S = pi_vs_vi(P, r, 0.9, always_n, theta, gw_arrows)
    out.append("    Gridworld, gamma = 0.9, step -0.04, PI from 'always N':")
    for k, (name, loss) in enumerate(zip(S["names"], S["losses"])):
        out.append(f"      round {k}: pi_{k} = {name}   loss = {loss:.4f}")
    out.append(f"      {len(S['policies'])} evaluations; VI: K = {S['K']}, k(pi*) = {S['k_opt']}; "
               f"V*((1,1)) = {S['V_star'][GW_START]:.4f}")
    if curves is not None:
        curves["Gridworld"] = S
    P, r = riverswim("weak")
    S = pi_vs_vi(P, r, 0.95, np.zeros(6, dtype=int), theta, rs_policy)
    out.append("    RiverSwim, weak current, gamma = 0.95, PI from 'always left':")
    for k, (name, loss) in enumerate(zip(S["names"], S["losses"])):
        out.append(f"      round {k}: pi_{k} = {name}   loss = {loss:.4f}")
    out.append(f"      {len(S['policies'])} evaluations; VI: K = {S['K']}, k(pi*) = {S['k_opt']}; "
               f"V*(s1) = {S['V_star'][0]:.4f}, V*(s6) = {S['V_star'][5]:.4f}")
    if curves is not None:
        curves["RiverSwim"] = S
    return out


def experiment_4b(rng, n_values=(1, 2, 3, 5, 10, 20, 30, 60, 100), datasets=1000, curves=None):
    P, r = machine()
    gamma = 0.9
    V_star, pi_star = optimal_values(P, r, gamma)
    never = np.zeros(3, dtype=int)
    out = [f"4(b) Certainty equivalence on the machine, gamma = 0.9: {datasets} datasets per n, "
           f"n samples per pair",
           "    P(wrong): share of datasets with pi_hat != pi*.   E[loss], E|V_hat - V*|: means over them.",
           f"    {'n':<5} {'P(wrong)':<10} {'E[loss]':<9} E|V_hat - V*|"]
    for n in n_values:
        wrong, losses, errs = 0, 0.0, 0.0
        for _ in range(datasets):
            counts, sums = sample_counts(P, r, n, rng)
            _, _, pi_hat, V_hat = certainty_equivalence(counts, sums, gamma, never)
            wrong += int(not np.array_equal(pi_hat, pi_star))
            losses += loss_of(P, r, gamma, pi_hat, V_star)
            errs += float(np.max(np.abs(V_hat - V_star)))
        out.append(f"    {n:<5} {wrong / datasets:<10.3f} {losses / datasets:<9.3f} {errs / datasets:.2f}")
        if curves is not None:
            curves[n] = (wrong / datasets, losses / datasets, errs / datasets)
    return out


def experiment_4c(rng, runs=100, episodes=100, checkpoints=(1, 10, 25, 50, 100), data=None):
    given = [("TD(0)", 0.05), ("TD(0)", 0.10), ("TD(0)", 0.15),
             ("MC", 0.01), ("MC", 0.02), ("MC", 0.03), ("MC", 0.04)]
    chosen = [(method, float(alpha)) for method, alpha in (("TD(0)", YOUR_ALPHA_TD), ("MC", YOUR_ALPHA_MC))
              if alpha is not None]
    yours = [setting for setting in chosen if setting not in given]
    out = [f"4(c) Random walk, gamma = 1, V_0 = 0.5: RMS error over the five states, averaged over "
           f"{runs} runs",
           f"    {'method':<8} {'alpha':<7}" + "".join(f"{'ep ' + str(c):>9}" for c in checkpoints)]
    # the same episodes for every setting: generate them once per run
    all_episodes = [[rw_episode(rng) for _ in range(episodes)] for _ in range(runs)]
    rms_of, snapshots = {}, {}
    for method, alpha in given + yours:
        rms = np.zeros(episodes + 1)
        for run in range(runs):
            V = np.full(len(RW_STATES), 0.5)
            rms[0] += np.sqrt(np.mean((V - RW_TRUE) ** 2))
            watch = run == 0 and (method, alpha) == ("TD(0)", 0.10)  # the left panel of Figure 5
            if watch:
                snapshots[0] = V.copy()
            for e, episode in enumerate(all_episodes[run], start=1):
                if method == "TD(0)":
                    td0_prediction(V, episode, 1.0, alpha)
                else:
                    mc_prediction(V, episode, 1.0, alpha)
                rms[e] += np.sqrt(np.mean((V - RW_TRUE) ** 2))
                if watch and e in (1, 10, 100):
                    snapshots[e] = V.copy()
        rms /= runs
        star = "*" if (method, alpha) in yours else ""
        out.append(f"    {method:<8} {f'{alpha:g}{star}':<7}" + "".join(f"{rms[c]:>9.4f}" for c in checkpoints))
        rms_of[(method, alpha)] = rms
    if yours:
        out.append("    * your step sizes, YOUR_ALPHA_TD and YOUR_ALPHA_MC below the imports")
    elif not chosen:
        out.append("    your step sizes are not set yet: YOUR_ALPHA_TD and YOUR_ALPHA_MC below the imports")
    for method, alpha in chosen:
        if (method, alpha) in given:
            out.append(f"    note: {method} at alpha = {alpha:g} is one of the given {method} step sizes; choose another")
        elif not 0 < alpha <= 0.5:
            out.append(f"    note: {method} at alpha = {alpha:g} is outside the (0, 0.5] the handout asks for")
    if data is not None:
        data.update(rms=rms_of, yours=yours, snapshots=snapshots, checkpoints=list(checkpoints))
    return out


# ------------------------------------------------------------------ figures
def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _hide_zeros(values):
    """Zeros as gaps, for a logarithmic axis."""
    values = np.asarray(values, dtype=float)
    return np.where(values > 1e-9, values, np.nan)


def save_figure_3(curves, path):
    """Figure 3 (Problem 4(a)): Lecture 7's two-run picture on both models, with the loss of the
    greedy policy of V_k and the three stopping points added."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    titles = {"Gridworld": "Gridworld, $\\gamma = 0.9$; PI from 'always N'",
              "RiverSwim": "RiverSwim (weak current), $\\gamma = 0.95$; PI from 'always left'"}
    for ax, name in zip(axes, ("Gridworld", "RiverSwim")):
        S = curves[name]
        gamma, K, k_opt, n_ev = S["gamma"], S["K"], S["k_opt"], len(S["policies"])
        k = np.arange(K + 1)
        E0 = S["losses"][0]
        ax.plot(k + 1, S["errors"], color="C0", linewidth=1.3, label=r"VI from $V_0 = 0$: error $\|V^\star - V_k\|$")
        ax.plot(k + 1, E0 * gamma ** k, "--", color="0.4", linewidth=1.1,
                label=r"$\gamma^k\,\|V^\star - V^{\pi_0}\|$: what Lecture 7's theorem promises PI")
        ax.plot(k + 1, _hide_zeros(S["greedy_losses"]), "s-", color="C1", markersize=3, linewidth=1,
                label=r"VI: loss of the greedy policy of $V_k$")
        ax.plot(np.arange(n_ev) + 1, _hide_zeros(S["losses"]), "o-", color="C3", markersize=4.5, linewidth=1.3,
                label=r"PI: loss of $\pi_k$")
        for x, color in ((k_opt + 1, "C1"), (n_ev, "C3"), (K + 1, "C0")):
            ax.axvline(x, color=color, linestyle=":", linewidth=1)
        top = max(E0, S["errors"][0]) * 3
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(3e-9, top)
        ax.set_xlim(0.85, (K + 1) * 1.25)
        ticks = [t for t in (1, 2, 3, 4, 6, 8, 11, 21, 51, 101, 201, 354) if t <= K + 1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t - 1) for t in ticks], fontsize=7)
        ax.minorticks_off()
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("$k$: iteration of VI, round of PI", fontsize=8)
        ax.set_title(titles[name], fontsize=8)
        ax.grid(alpha=0.3)
        ax.text(k_opt + 1, top / 1.6, f" greedy policy of $V_k$ optimal from $k = {k_opt}$", color="C1",
                fontsize=6.5, ha="left", va="top", rotation=90)
        ax.text(n_ev, 5e-9, f"PI stops: {n_ev} evaluations ", color="C3", fontsize=6.5, ha="right", va="bottom",
                rotation=90)
        ax.text(K + 1, 5e-9, f"VI stops: $K = {K}$ ", color="C0", fontsize=6.5, ha="right", va="bottom", rotation=90)
    axes[0].set_ylabel("max-norm error, loss", fontsize=8)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(path, dpi=400)
    plt.close(fig)


def save_figure_4(curves, path):
    """Figure 4 (Problem 4(b)): certainty equivalence against n: the value error and the loss, and
    the probability of a wrong policy."""
    plt = _pyplot()
    n = np.array(sorted(curves))
    p_wrong = np.array([curves[k][0] for k in n])
    loss = np.array([curves[k][1] for k in n])
    err = np.array([curves[k][2] for k in n])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ax1.loglog(n, err, "o-", linewidth=1.2, markersize=4, label=r"$\mathbb{E}\,\|\hat V - V^\star\|$")
    ax1.loglog(n, 21 / np.sqrt(n), "--", color="0.4", linewidth=1.4, label=r"$21/\sqrt{n}$")
    ax1.loglog(n, _hide_zeros(loss), "s-", linewidth=1.2, markersize=4, color="C3",
               label=r"$\mathbb{E}\,[\|V^\star - V^{\hat\pi}\|]$, the loss of $\hat\pi$")
    ax1.set_xlabel("$n$ samples per pair", fontsize=8)
    ax1.set_ylabel("mean over 1000 datasets", fontsize=8)
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3, which="both")
    ax1.tick_params(labelsize=7)
    ax2.semilogx(n, p_wrong, "o-", linewidth=1.2, markersize=4, color="C1")
    ax2.set_xlabel("$n$ samples per pair", fontsize=8)
    ax2.set_ylabel(r"$P(\hat\pi \neq \pi^\star)$", fontsize=8)
    ax2.set_ylim(-0.02, 0.75)
    ax2.grid(alpha=0.3, which="both")
    ax2.tick_params(labelsize=7)
    fig.suptitle("Certainty equivalence on the machine, $\\gamma = 0.9$", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=400)
    plt.close(fig)


def save_figure_5(data, path):
    """Figure 5 (Problem 4(c)): one run of TD(0) state by state, and the RMS error of every setting
    against episodes: Sutton and Barto's Figure 6.2."""
    plt = _pyplot()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.1))
    x = np.arange(len(RW_STATES))
    for e, V in sorted(data["snapshots"].items()):
        ax1.plot(x, V, "o-", markersize=3.5, linewidth=1.1, label=f"after {e} episode{'s' if e != 1 else ''}")
    ax1.plot(x, RW_TRUE, "k--", linewidth=1.1, label="true values $k/6$")
    ax1.set_xticks(x)
    ax1.set_xticklabels(RW_STATES, fontsize=7)
    ax1.tick_params(axis="y", labelsize=7)
    ax1.set_ylabel("estimated value", fontsize=8)
    ax1.set_ylim(0, 1)
    ax1.set_title(r"one run of TD(0), $\alpha = 0.1$", fontsize=8.5)
    ax1.legend(fontsize=6.5)
    ax1.grid(alpha=0.3)
    for (method, alpha), rms in data["rms"].items():
        yours = (method, alpha) in data["yours"]
        style = "-" if method == "TD(0)" else "--"
        ax2.plot(np.arange(len(rms)), rms, style, linewidth=(2.2 if yours else 1.0), alpha=(1.0 if yours else 0.85),
                 label=rf"{method}, $\alpha = {alpha:g}$" + (" (yours)" if yours else ""))
    ax2.set_xlabel("episodes", fontsize=8)
    ax2.set_ylabel("RMS error (5 states, 100 runs)", fontsize=8)
    ymax = max(np.max(curve) for curve in data["rms"].values())
    ax2.set_ylim(0, max(0.3, 1.05 * ymax))
    ax2.set_title("TD(0) solid, MC dashed", fontsize=8.5)
    ax2.legend(fontsize=6, ncol=2)
    ax2.grid(alpha=0.3)
    ax2.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=400)
    plt.close(fig)


def _block(name, fn):
    """Run one experiment block; print its lines, or one line saying why it was skipped.
    Returns True if the block ran."""
    try:
        lines = fn()
    except NotImplementedError:
        print(f"[SKIPPED] {name}: a function it needs is not implemented yet")
        return False
    except Exception as e:  # noqa: BLE001
        print(f"[SKIPPED] {name}: raised {type(e).__name__}: {e}")
        return False
    print("\n".join(lines))
    return True


def run_experiments(seed=639, figures=("hw2_fig3.png", "hw2_fig4.png", "hw2_fig5.png")):
    """Problem 4: print the three blocks, each with the functions as they are; save Figures 3 to 5."""
    data = [{} for _ in range(3)]
    blocks = (("4(a)", lambda: experiment_4a(curves=data[0]), save_figure_3),
              ("4(b)", lambda: experiment_4b(np.random.default_rng(seed), curves=data[1]), save_figure_4),
              ("4(c)", lambda: experiment_4c(np.random.default_rng(seed + 1), data=data[2]), save_figure_5))
    ran = [_block(name, fn) for name, fn, _ in blocks]
    for (name, _, saver), d, ok, path in zip(blocks, data, ran, figures):
        if not ok:
            print(f"Figure not saved: {name} did not run.")
            continue
        try:
            saver(d, path)
            print(f"Figure saved: {path}")
        except Exception as e:  # noqa: BLE001
            print(f"Could not make the figure: {type(e).__name__}: {e}")


# ------------------------------------- Problem 3: the five functions on small inputs
# Each function runs on one small input.  Its result is printed as a [RESULT] line, for your own
# checking, and drawn into Figure 1 (the planning functions) or Figure 2 (the learning functions).
WAIT_WAIT = np.array([1, 1])          # the pond's policy (w, w)
NEVER_REPAIR = np.array([0, 0, 0])    # the machine's policy "never repair"
TIE_Q = np.array([[2.0, 2.0 + 5e-10], [1.0, 3.0], [4.0, 4.0 - 1e-12]])
TIE_PI_OLD = np.array([1, 0, 1])
TIE_LABELS = [["$2$", r"$2 + 5\cdot 10^{-10}$"], ["$1$", "$3$"], ["$4$", r"$4 - 10^{-12}$"]]
RW_EPISODES = [
    [(2, 0.0, 3), (3, 0.0, 4), (4, 1.0, None)],
    [(2, 0.0, 1), (1, 0.0, 0), (0, 0.0, None)],
    [(2, 0.0, 1), (1, 0.0, 2), (2, 0.0, 3), (3, 0.0, 4), (4, 1.0, None)],
]
MACHINE_STATES = ["G", "W", "B"]
MACHINE_ACTIONS = ["run", "repair"]


def pond_name(pi):
    """A pond policy as (a_F, a_L) in the handout's letters: f = fish, w = wait."""
    return "(" + ", ".join("fw"[a] for a in np.asarray(pi, dtype=int)) + ")"


def _print_result(tag, detail):
    """Print tag + detail, one clause per line when the line would be long."""
    if len(tag) + len(detail) <= 92 or "; " not in detail:
        print(tag + detail)
    else:
        pieces = detail.split("; ")
        print(tag + pieces[0] + ";")
        for piece in pieces[1:-1]:
            print(" " * len(tag) + piece + ";")
        print(" " * len(tag) + pieces[-1])


def _report(name, fn, describe):
    """Run fn() and print a [RESULT] line from describe(data), or a [TODO]/[ERROR] line.  Return the
    outcome the figures are drawn from: ("ok", data), ("todo", message) or ("error", message)."""
    try:
        data = fn()
        detail = describe(data)
    except NotImplementedError:
        print(f"[TODO] {name}: not implemented yet")
        return "todo", f"{name}: not implemented yet"
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] {name}: raised {type(e).__name__}: {e}")
        return "error", f"{name} raised {type(e).__name__}: {e}"
    _print_result(f"[RESULT] {name}: ", detail)
    return "ok", data


def _result_improve():
    return dict(pi_old=TIE_PI_OLD.tolist(), pi_new=np.asarray(improve(TIE_Q, TIE_PI_OLD)).tolist(),
                greedy=greedy_policy(TIE_Q).tolist())


def _describe_improve(d):
    return f"Q with ties, pi_old = {d['pi_old']}; improve: pi = {d['pi_new']}; greedy_policy: pi = {d['greedy']}"


def _result_policy_iteration():
    P, r = pond()
    policies, values = policy_iteration(P, r, 0.9, WAIT_WAIT)
    return dict(policies=[np.asarray(p).tolist() for p in policies], values=[_clean(V, 4).tolist() for V in values])


def _describe_policy_iteration(d):
    parts = ["pond, gamma = 0.9, from (w, w)", "policies = " + ", ".join(pond_name(p) for p in d["policies"])]
    parts += [f"V^(pi_{k}) = {V}" for k, V in enumerate(d["values"])]
    parts.append(f"{len(d['policies'])} evaluations")
    return "; ".join(parts)


def _ce_counts():
    """Two samples at every pair of the machine: the dataset of Figure 2's left panel."""
    counts = np.zeros((3, 2, 3))
    counts[0, 0] = [1, 1, 0]
    counts[1, 0] = [0, 1, 1]
    counts[2, 0] = [0, 0, 2]
    counts[:, 1, 0] = 2
    return counts


def _result_certainty_equivalence():
    P, r = machine()
    P_hat, r_hat, pi_hat, V_hat = certainty_equivalence(_ce_counts(), 2 * r, 0.9, NEVER_REPAIR)
    return dict(P_hat=_clean(P_hat, 4).tolist(), r_hat=_clean(r_hat, 4).tolist(),
                pi_hat=np.asarray(pi_hat).tolist(), V_hat=_clean(V_hat, 3).tolist())


def _describe_certainty_equivalence(d):
    P_hat, r_hat = np.array(d["P_hat"]), np.array(d["r_hat"])
    return (f"machine, gamma = 0.9, 2 samples per pair; "
            f"P_hat(.|G,run) = {_clean(P_hat[0, 0]).tolist()}, P_hat(.|W,run) = {_clean(P_hat[1, 0]).tolist()}; "
            f"r_hat(., repair) = {_clean(r_hat[:, 1]).tolist()}; "
            f"pi_hat = {d['pi_hat']}, V_hat = {d['V_hat']}")


def _result_prediction(update):
    V = np.full(5, 0.5)
    rows = [_clean(V, 4).tolist()]
    for ep in RW_EPISODES:
        update(V, ep, 1.0, 0.1)
        rows.append(_clean(V, 4).tolist())
    return dict(rows=rows)


def _result_td0_prediction():
    return _result_prediction(td0_prediction)


def _result_mc_prediction():
    return _result_prediction(mc_prediction)


def _describe_prediction(d):
    parts = ["random walk, alpha = 0.1, V_0 = 0.5, the three episodes in turn"]
    parts += [f"after episode {i}: V = {row}" for i, row in enumerate(d["rows"][1:], start=1)]
    return "; ".join(parts)


RESULTS = (("improve", _result_improve, _describe_improve),
           ("policy_iteration", _result_policy_iteration, _describe_policy_iteration),
           ("certainty_equivalence", _result_certainty_equivalence, _describe_certainty_equivalence),
           ("td0_prediction", _result_td0_prediction, _describe_prediction),
           ("mc_prediction", _result_mc_prediction, _describe_prediction))


def report_results():
    """Problem 3: run the five functions on their inputs, print a line for each, and return the
    outcomes that Figures 1 and 2 are drawn from."""
    return {name: _report(name, fn, describe) for name, fn, describe in RESULTS}


# -------------------------------------------------------- Problem 3's figures
def _footer(fig):
    fig.text(0.995, 0.008, f"hw2.py \u00b7 Python {sys.version.split()[0]} \u00b7 NumPy {np.__version__}",
             ha="right", va="bottom", fontsize=6, color="0.45")


def _panel(ax, name, outcome, draw):
    """Draw one panel from an outcome of _report: the result if the function ran, else its message."""
    status, payload = outcome
    if status == "ok":
        try:
            draw(ax, payload)
            return
        except Exception as e:  # noqa: BLE001
            ax.clear()
            status, payload = "error", f"{name}: its result could not be drawn ({type(e).__name__}: {e})"
    ax.set_title(name, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.8")
    text = ("[TODO] " if status == "todo" else "[ERROR] ") + payload
    ax.text(0.5, 0.5, "\n".join(textwrap.wrap(text, 34)), ha="center", va="center", fontsize=7.5,
            color="0.45" if status == "todo" else "C3", transform=ax.transAxes)


def _draw_improve(ax, d):
    from matplotlib.patches import Rectangle

    for s in range(3):
        for a in range(2):
            face = "#cfe8cf" if d["pi_new"][s] == a else "white"
            ax.add_patch(Rectangle((a - 0.5, s - 0.5), 1, 1, facecolor=face, edgecolor="0.55", linewidth=0.8))
            ax.text(a, s, TIE_LABELS[s][a], ha="center", va="center", fontsize=8)
        ax.add_patch(Rectangle((d["pi_old"][s] - 0.42, s - 0.42), 0.84, 0.84, fill=False, edgecolor="k",
                               linewidth=1.4, linestyle="--"))
        ax.plot(d["greedy"][s] + 0.38, s - 0.36, "o", color="C3", markersize=3.5)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(2.5, -0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["$a = 0$", "$a = 1$"], fontsize=7.5)
    ax.xaxis.tick_top()
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([f"$s = {s}$" for s in range(3)], fontsize=7.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("improve on $Q$ with two near-ties, tol $= 10^{-9}$", fontsize=8)
    ax.set_xlabel(f"dashed: $\\pi_{{\\mathrm{{old}}}} = {d['pi_old']}$     filled: improve $\\to {d['pi_new']}$\n"
                  f"$\\bullet$: greedy_policy $\\to {d['greedy']}$", fontsize=7.5)


def _draw_pond_run(ax, d):
    P, r = pond()
    V_star, _ = optimal_values(P, r, 0.9)
    V = np.array(d["values"], dtype=float)
    k = np.arange(len(d["policies"]))
    for s, (name, color, dy) in enumerate((("$V^{\\pi_k}(\\mathrm{F})$", "C0", 7), ("$V^{\\pi_k}(\\mathrm{L})$", "C1", -11))):
        ax.plot(k, V[:, s], "o-", color=color, markersize=4.5, linewidth=1.3, label=name)
        for kk, v in zip(k, V[:, s]):
            ax.annotate(f"{v:g}", (kk, v), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=7.5,
                        color=color)
        ax.axhline(V_star[s], color=color, linestyle="--", linewidth=0.9, alpha=0.6)
        ax.annotate(f"$V^\\star(\\mathrm{{{'FL'[s]}}}) = {V_star[s]:g}$", (k[-1] + 0.1, V_star[s]),
                    textcoords="offset points", xytext=(0, 2), fontsize=7, color=color, va="bottom")
    ax.set_xticks(k)
    ax.set_xticklabels([f"{kk}\n{pond_name(p)}" for kk, p in zip(k, d["policies"])], fontsize=7.5)
    ax.set_xlim(-0.3, k[-1] + 0.8)
    lo, hi = min(0.0, float(V.min())), max(float(V_star.max()), float(V.max()))
    ax.set_ylim(lo - 0.08 * (hi - lo) - 1, hi + 0.14 * (hi - lo) + 1)
    ax.set_xlabel("round $k$ and its policy $\\pi_k = (a_{\\mathrm{F}}, a_{\\mathrm{L}})$", fontsize=8)
    ax.set_ylabel("value", fontsize=8)
    ax.set_title(f"policy_iteration on the pond from (w, w), $\\gamma = 0.9$: {len(k)} evaluations", fontsize=8)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    ax.tick_params(axis="y", labelsize=7)


def save_figure_1(results, path):
    """Figure 1 (Problem 3(a)): the improvement step with the tie rule on a table with two near-ties,
    and policy iteration on the pond, round by round."""
    plt = _pyplot()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.1), gridspec_kw=dict(width_ratios=(1, 1.6)))
    _panel(ax1, "improve", results["improve"], _draw_improve)
    _panel(ax2, "policy_iteration", results["policy_iteration"], _draw_pond_run)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _footer(fig)
    fig.savefig(path, dpi=400)
    plt.close(fig)


def _draw_ce(ax, d):
    from matplotlib.patches import Rectangle

    blues = _pyplot().cm.Blues
    P, r = machine()
    V_star, pi_star = optimal_values(P, r, 0.9)
    P_hat, r_hat = np.array(d["P_hat"]), np.array(d["r_hat"])
    headers = ["$s' = \\mathrm{G}$", "$s' = \\mathrm{W}$", "$s' = \\mathrm{B}$", "run", "repair",
               "$\\hat\\pi(s)$", "$\\hat V(s)$"]
    widths = [0.9, 0.9, 0.9, 0.8, 1.0, 1.0, 1.2]              # one per column, each wide enough for its text
    edges = np.concatenate([[-0.5], -0.5 + np.cumsum(widths)])
    xs = (edges[:-1] + edges[1:]) / 2
    for i in range(3):
        cells = [(f"{P_hat[i, 0, j]:.2f}", f"{P[i, 0, j]:.2f}", P_hat[i, 0, j]) for j in range(3)]
        cells += [(f"{r_hat[i, 0]:g}", f"{r[i, 0]:g}", None), (f"{r_hat[i, 1]:g}", f"{r[i, 1]:g}", None),
                  (MACHINE_ACTIONS[d["pi_hat"][i]], MACHINE_ACTIONS[pi_star[i]], None),
                  (f"{d['V_hat'][i]:.3f}", f"{V_star[i]:.2f}", None)]
        for j, (estimate, truth, shade) in enumerate(cells):
            face = blues(0.05 + 0.4 * shade) if shade is not None else "white"
            ax.add_patch(Rectangle((edges[j], i - 0.5), widths[j], 1, facecolor=face, edgecolor="0.55",
                                   linewidth=0.8))
            ax.text(xs[j], i - 0.13, estimate, ha="center", va="center", fontsize=7.5)
            ax.text(xs[j], i + 0.27, truth, ha="center", va="center", fontsize=6, color="0.45")
        ax.text(-0.62, i, f"$s = \\mathrm{{{MACHINE_STATES[i]}}}$", ha="right", va="center", fontsize=7.5)
    for j, head in enumerate(headers):
        ax.text(xs[j], -0.8, head, ha="center", va="center", fontsize=7)
    for j0, j1, head in ((0, 3, "$\\hat P(s' \\mid s, \\mathrm{run})$"), (3, 5, "$\\hat r(s, a)$")):
        ax.text((edges[j0] + edges[j1]) / 2, -1.3, head, ha="center", va="center", fontsize=7.5)
        ax.plot([edges[j0] + 0.05, edges[j1] - 0.05], [-1.07, -1.07], color="0.3", linewidth=0.7)
    ax.set_xlim(-1.45, edges[-1])
    ax.set_ylim(2.5, -1.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("certainty_equivalence on the machine\n$\\gamma = 0.9$, 2 samples per pair", fontsize=8)
    ax.set_xlabel("black: the estimate and its plan;  gray: the true model, $\\pi^\\star$, $V^\\star$", fontsize=7)


def _draw_values_grid(ax, d, title):
    M = np.array(d["rows"], dtype=float).T                    # states down the side, time across the top
    ax.imshow(M, cmap="RdBu_r", vmin=0.3, vmax=0.7, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:g}", ha="center", va="center", fontsize=7.5)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(["$V_0$"] + [f"after\nep {j}" for j in range(1, M.shape[1])], fontsize=7.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(RW_STATES[:M.shape[0]], fontsize=7.5)
    ax.set_xticks(np.arange(-0.5, M.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, M.shape[0]), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)
    ax.set_title(title, fontsize=8)


def save_figure_2(results, path):
    """Figure 2 (Problem 3(b)): certainty equivalence from one dataset of two samples per pair, and
    TD(0) against Monte Carlo after each of three episodes of the random walk."""
    plt = _pyplot()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8.0, 3.1), gridspec_kw=dict(width_ratios=(1.9, 1, 1)))
    _panel(ax1, "certainty_equivalence", results["certainty_equivalence"], _draw_ce)
    _panel(ax2, "td0_prediction", results["td0_prediction"],
           lambda ax, d: _draw_values_grid(ax, d, "td0_prediction, $\\alpha = 0.1$"))
    _panel(ax3, "mc_prediction", results["mc_prediction"],
           lambda ax, d: _draw_values_grid(ax, d, "mc_prediction, $\\alpha = 0.1$"))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.text(0.5, 0.045, "TD(0) and Monte Carlo: $\\gamma = 1$, $V_0 = 0.5$; the three episodes, in turn:   "
             "C\u2192D\u2192E (+1)    C\u2192B\u2192A (0)    C\u2192B\u2192C\u2192D\u2192E (+1)",
             ha="center", va="bottom", fontsize=7)
    _footer(fig)
    fig.savefig(path, dpi=400)
    plt.close(fig)


def problem3_figures(results, figures=("hw2_fig1.png", "hw2_fig2.png")):
    """Problem 3: draw Figures 1 and 2 from the five outcomes, written or not."""
    for saver, path in zip((save_figure_1, save_figure_2), figures):
        try:
            saver(results, path)
            print(f"Figure saved: {path}")
        except Exception as e:  # noqa: BLE001
            print(f"Could not make the figure: {type(e).__name__}: {e}")


def main():
    print(f"Python {sys.version.split()[0]}, NumPy {np.__version__}")
    results = report_results()
    problem3_figures(results)
    run_experiments()


if __name__ == "__main__":
    main()
