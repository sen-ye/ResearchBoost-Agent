# The judge

The judge is the core of the method. The agent may try anything, but only a PASS from the frozen judge counts. The code lives in [`../numerics`](../numerics) (the generic component; each task copies it and freezes the copy) and in each task's `tasks/<t>/bench/` (task adapter, timing, cases).

## Four gates

| Gate | What it checks | Verdict on failure |
|---|---|---|
| 0 Structure | The entry points the judge drives (`main()`, `parse_args()`, everything in `train()` outside the per-step loop) match the baseline AST; the dataset class yields bitwise-identical batches on the frozen fixture; every exact item of the short cases matches: phase, number of model forward calls, RNG fingerprint before and after each step, learning rate, step counter, cross-rank pattern, tensor names and shapes, all values finite | ERROR / INCORRECT |
| 1 Benefit | The agent's `check`: one fresh process of the candidate with a cold compile cache, against the baseline timing frozen at formulation (the mean of its baseline runs, each warmup + window × repeats). The final check (`check --final`, run by the harness only): baseline → candidate → baseline in three independent processes, every window accepted on its own (a window ratio that drifts from the frozen baseline is INVALID and rerun). The speedup must exceed `max(1.02, 1 + 2 × the baseline spread)`: the gap between the two baseline runs in the final check, the frozen runs' spread in the agent's `check` | UNAVAILABLE (no_benefit); precision is not evaluated |
| 2 Bitwise | In deterministic mode, record 1–2-step cases; every tensor (loss / out / grad / delta / optim / buffer) must match bit for bit. Large tensors are compared through a CountSketch, with norms and hashes stored separately | If not bitwise, go to gate 3 |
| 3 Statistical | In deterministic mode, one paired short trajectory (tens to a few hundred steps) on the first frozen baseline seed, 5–7 endpoints. Each endpoint's bias (candidate − baseline) within 3σ passes (N1); any endpoint beyond 3σ fails. The same rule in every check | INCORRECT |

- **Exact items come first.** A failed exact item is always INCORRECT: the candidate does something different. ERROR is only for the case where every exact item holds but the tensors still disagree.
- **Tiers.** N0 = everything bitwise identical, with no change of dtype or math mode; N1 = the same real-valued function; N2 = a change of dtype or accumulation precision. Anything beyond N2 is INCORRECT. **Never trust the tier a candidate claims for itself**: a change of an internal accumulator's dtype does not show up in the judge's dtype inventory and can only be found by reading the code.
- **σ** is the standard deviation of the baseline across its own seeds (6 or more, `numerics traj-freeze`), frozen before any candidate is seen; gate 3 bounds each endpoint's bias on one seed by 3σ (`numerics traj-compare --sigma-k 3`). Every candidate is compared with the **original baseline**, never with the previously accepted candidate.
- **The bitwise records hash every recorded tensor with SHA-256.** `numerics/shapipe.py` overlaps the device-to-host copies (pinned, asynchronous) and the hashing (a thread pool), with the same digests as hashing one tensor after the other.
- **Control arm.** Apply a small known perturbation to the baseline (for example `lr × 1.02`); the statistical gate must reject it as INCORRECT, otherwise it is not sensitive enough.

## Check modes

The bench's `check` has the following modes. Every GPU command holds the bench's lock, so they run one at a time.

| Command | Gates | Timing | Statistical gate | Who runs it |
|---|---|---|---|---|
| `time` | — | the candidate once, against the frozen baseline timing | — | the agent |
| `check --quick` | 0, exact items of 2 | — | — | the agent |
| `check --skip-timing` | 0, 2, 3 | — | one seed, 3σ | the agent |
| `check` | 0, 1, 2, 3 | the candidate once, against the frozen baseline timing; interference is reported, not rejected | one seed, 3σ | the agent |
| `check --n1` | 0, 3 | — | one seed, 3σ | the agent, after a `check` whose gate 2 was not bitwise identical |
| `check --final` | 0, 1, 2, 3 | baseline → candidate → baseline; interference → INVALID | one seed, 3σ | the harness only (`goal/clawo/judge.sh`), after the agent has stopped |

`check --final` is refused unless the harness's marker `task/final-evaluation` exists: `judge.sh` writes it from outside the sandbox (where `ws/task` is read-only) right before the final check and removes it afterwards. The agent's `check` runs every gate the final check runs, with the same rules; only the timing of gate 1 differs.

## When the judge may change, and when it may not

- **Relax only where the baseline itself cannot comply.** For example, a quantity the baseline cannot reproduce bit for bit may be moved out of the bitwise comparison. **Never relax a gate because a candidate cannot meet it**: that amounts to grading your own work. If a candidate cannot meet a gate, let it fail and leave the trade-off to the owner.
- Wanting to change a gate only after it rejected something requires asking the owner first. The change needs an audit record (`task/judge_changes.md`): the old and new wording, the data that triggered it, a regression table replaying earlier runs under the old and new rules, and a rerun of all four acceptance checks under the new threshold.
- Relaxing a task constraint (for example allowing samples to be reassigned) changes the **task statement**, not the **judge**. When the task statement changes, the judge is rebuilt for the new statement and re-certified.
- `evaluator_sha256` covers only the judge code; changing the goal in GOAL.md does not change the hash. Check that the hash is unchanged before every acceptance.

## Fake gains the judge caught

| Kind | Example | Caught by |
|---|---|---|
| Broken gradient | A bare `swiglu_forward` bypassed autograd; the whole MLP received no gradient; measured 1.456× | the `dead_gradient` exact item (a gradient that is nonzero in the baseline must be nonzero in the candidate) |
| Cross-rank sync changed | DDP `static_graph` silently changed how gradients were synchronized; 1.7311× | the cross-rank-pattern exact item |
| Random stream altered | xformers and Triton's file cache drew from the global `random` | the RNG-fingerprint exact item |
| Compilation inside the timing window | Recompilation on first entry into a new phase | the compile counter |
| Compilation falling back to eager | After one call under a dispatch mode (dtype tracing, SAC) the compiled function permanently fell back to eager, so a "bitwise PASS" was not what it seemed | rerun without tracing; look for compiler-generated kernels in the trace |
| Timing and trajectories on different code paths | 4 data-loader workers for timing, 1 for trajectories | withdrawn by the agent itself; the rule is now "the same code path" |
| Objective changed | Reordering samples shifted each token's weight by 28%; probe loss moved by 7σ | statistical gate |

## Known limits of the judge

- It measures steady-state step time only, not one-time O(dataset) costs. Say so when formulating the task, or ask candidates to report their startup cost.
- Exact items such as forward-call counts reject a mathematically equivalent "merge several forwards into one batched forward", and whether they do depends on whether the merged calls go through `__call__`.
- A 60-step trajectory cannot catch slow drift. Only a real long training run gives that evidence.
- One seed is noisier than the mean of several. Where paired runs decorrelate within the trajectory (GAN training), a legitimate change's single-seed bias can exceed 3σ; check the control arms and a known-equivalent variant at formulation, and lengthen or move the endpoints if they cannot be told apart.
- Shape-dependent crashes are invisible to `time` and `check --quick`; the statistical gate sometimes catches them because it uses a different per-GPU batch.
