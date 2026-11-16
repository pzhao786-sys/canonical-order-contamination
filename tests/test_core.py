"""Logic tests with mock scorers (no torch, no GPU).

Three synthetic "models" on a file of n examples with ids 0..n-1:
  * null        : log-prob does not depend on order
  * memorised   : bonus for every adjacent pair (i, i+1) in the *forward* order
  * structured  : bonus for every adjacent pair from the same topic block
The test must: stay calibrated on null, flag canonical for both others, and the
reversed order must separate memorised (not preferred) from structured (preferred).

Run:  python -m tests.test_core      (from the experiments/ folder)
"""
import numpy as np

from contam.core import resolve_num_shards, run_file, shard_bounds, ttest_greater


class MockScorer:
    def __init__(self, mode, block=5, bonus=1.0, noise=1.0, seed=0):
        self.mode, self.block, self.bonus, self.noise = mode, block, bonus, noise
        self.rng = np.random.default_rng(seed)

    def tokenize(self, examples):
        return [[int(e)] for e in examples]  # one "token" = the example id

    def score(self, seqs):
        out = np.zeros(len(seqs))
        for i, s in enumerate(seqs):
            v = -3.0 * len(s) + self.noise * self.rng.standard_normal()
            for a, b in zip(s[:-1], s[1:]):
                if self.mode == "memorised" and b == a + 1:
                    v += self.bonus
                if self.mode == "structured" and a // self.block == b // self.block:
                    v += self.bonus
            out[i] = v
        return out


def test_shards():
    assert shard_bounds(10, 3) == [(0, 4), (4, 7), (7, 10)]
    assert resolve_num_shards(1000, 50) == 50
    assert resolve_num_shards(100, 50, min_shard_size=10) == 10
    assert resolve_num_shards(3, 50) == 1
    z, p = ttest_greater([1.0, 2.0, 3.0, 4.0], ddof=0)
    assert abs(z - 2.5 / np.std([1, 2, 3, 4]) * 2) < 1e-12 and 0 < p < 0.05


def run(mode, seed=0, n=1000):
    ex = [str(i) for i in range(n)]
    return run_file(MockScorer(mode, seed=seed), ex, num_shards=50, perms=30, seed=seed)["summary"]


def test_signatures():
    mem, struct = run("memorised"), run("structured")
    assert mem["p_canon"] < 1e-6, mem["p_canon"]
    assert mem["p_rev"] > 0.01, mem["p_rev"]                     # reversed: no advantage
    assert mem["p_canon_minus_rev_two_sided"] < 1e-6
    assert struct["p_canon"] < 1e-6, struct["p_canon"]
    assert struct["p_rev"] < 1e-6, struct["p_rev"]               # reversed: same advantage
    assert abs(struct["eff_canon_per_ex"] - struct["eff_rev_per_ex"]) < 0.05
    print("memorised :", {k: round(v, 4) for k, v in mem.items() if k.startswith(("p_", "eff_")) and not k.endswith("se")})
    print("structured:", {k: round(v, 4) for k, v in struct.items() if k.startswith(("p_", "eff_")) and not k.endswith("se")})


def test_null_calibration():
    ps = [run("null", seed=s, n=500)["p_canon"] for s in range(200)]
    rate = float(np.mean(np.array(ps) < 0.05))
    assert 0.01 <= rate <= 0.10, rate
    null_p = np.array(run("memorised")["null_p"])
    assert len(null_p) == 30 and (null_p > 1e-4).all()           # fake canonicals look null
    print(f"null false-positive rate at 0.05 over 200 runs: {rate:.3f}")


def test_fixed_cache():
    ex = [str(i) for i in range(200)]
    a = run_file(MockScorer("memorised", noise=0.0), ex, num_shards=20, perms=5, seed=1)
    b = run_file(MockScorer("memorised", noise=0.0), ex, num_shards=20, perms=5, seed=1,
                 fixed={"ll_canon": a["ll_canon"], "ll_rev": a["ll_rev"]})
    assert a["ll_shuf"] == b["ll_shuf"] and a["ll_canon"] == b["ll_canon"]


if __name__ == "__main__":
    test_shards(); test_signatures(); test_null_calibration(); test_fixed_cache()
    print("all tests passed")
