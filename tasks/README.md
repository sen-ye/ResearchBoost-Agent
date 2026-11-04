# tasks

Each directory holds what the optimizing agent saw (`GOAL.md`) and the frozen judge it was measured by, plus the scripts and notes of whoever prepared the task. Reference tensors, reference trajectories and datasets are not included: they are machine-specific and are re-created with the prep scripts (see [../docs/run-a-task.md](../docs/run-a-task.md)). Results and links: [../README.md](../README.md).

`formulation/` and `prep/` are for the person or agent who prepares the task. Keep them out of the optimizing agent's sandbox.

| Task | Contents | Notes |
|---|---|---|
| [rae](rae) | `GOAL.md`, `bench/` (`run.py`, timing, adapter, cases), `numerics/`, `task/formulation.json`, frozen δ in `bench/traj/frozen.json` | The `GOAL.md` is the single-session goal run of this task (5.26×). After the runs, `run.py` lost a path only an earlier runner used (`candidate`, `--parent` / `--goal memory`, `profile --snapshot`), so its `evaluator_sha256` differs from the one recorded in `task/formulation.json`; `check`, `sweep`, `time`, `profile --tree` and `baseline` are unchanged |
| [self-forcing](self-forcing) | `bench/`, `numerics/`, `prep/` (records, trajectories, input freeze, quick acceptance); `interactive/` = the workspace the owner and an agent used together (`GOAL.md`, `NOTES.md`, bench patches) that reached 2.08× | |
| [llava](llava) | `GOAL.md`, `SUPERVISOR.md`, `bench/` (bench.py, timing, adapter, loader probe, per-rank load report), `numerics/`, `task/` (formulation, baseline, `judge_changes.md` audit, partition sensitivity, re-acceptance checks), `formulation/` (ceiling note, prep scripts) | The judge was reworked once after the owner relaxed three task constraints; the audit is in `task/judge_changes.md` |
| [dmd2](dmd2) | `GOAL.md`, `GOAL_LINE.txt`, `SUPERVISOR.md`, `bench/` (bench.py, recipe, timing, adapter, wandb stub), `numerics/`, `task/` (formulation, baseline, profile, timing reference), `formulation/prep` (environment, assets, fixture, acceptance, trajectory freeze) | The most complete template for a new task |

## Earlier judge rules

These benches are the frozen judges the runs were measured by, and they predate the current check rules ([../docs/judge.md](../docs/judge.md#check-modes)): every full `check` times reference → candidate → reference, and gate 3 runs 3 seeds (then 6) against δ = 2σ, with a one-seed preview in the agent's own check. A new task follows the current rules: the agent's `check` times the candidate against the frozen reference timing, gate 3 bounds one seed's bias by 3σ, `check --n1` repeats gate 3 alone, and `check --final` is the harness's. Copy the structure of these benches, not those two gates.

## Archived records

Files under `task/` (timing records, formulation, re-acceptance checks) are the records written during the runs. Machine paths in them were rewritten to `$ASSETS` / `$WORK` / `$LOCAL` and GPU UUIDs to `GPU-0`…`GPU-3` after the runs; numbers and digests are unchanged. The LLaVA and DMD2 `bench.py` now expand those variables when reading `bench.json` and `task/formulation.json` (set `ASSETS` before running them); because `bench.py` and `bench.json` are part of the evaluator hash, their `evaluator_sha256` also differs from the recorded one. The judging logic is unchanged.

## Language

The task material was written in Chinese and is translated here: `GOAL.md`, `GOAL_LINE.txt`, `SUPERVISOR.md`, the Self-Forcing `NOTES.md`, the LLaVA `judge_changes.md` and ceiling note, the baselines and the bench READMEs. Each translated `GOAL.md` says so in its first lines. Two files stay in Chinese because their content is hashed as a frozen input: `llava/task/partition_sensitivity.json` (its sha256 is recorded in `llava/task/formulation.json`), and `llava/formulation/prep/freeze_sensitivity.py`, whose string values are what it writes into that file.
