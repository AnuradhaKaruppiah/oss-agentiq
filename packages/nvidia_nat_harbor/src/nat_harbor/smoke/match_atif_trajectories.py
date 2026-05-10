# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare native and ATOF-derived ATIF trajectories with AgentEvals."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from collections.abc import Callable
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict

from nat_harbor.smoke.compare_atif_tools import ToolSequenceComparison
from nat_harbor.smoke.compare_atif_tools import compare_atif_tool_sequences

DEFAULT_NATIVE_REL = Path("agent/trajectory.json")
DEFAULT_CANDIDATE_REL = Path("agent/nemo-flow-atof-atif/trajectory.json")

MATCH = "match"
MISMATCH = "mismatch"
MISSING = "missing"
ERROR = "error"

TrajectoryMatchMode = str
ToolArgsMatchMode = str
ATIFConverter = Callable[[Mapping[str, Any]], list[dict[str, Any]]]
TrajectoryEvaluator = Callable[..., Any]
TrajectoryEvaluatorFactory = Callable[..., TrajectoryEvaluator]


class _FrozenModel(BaseModel):
    """Frozen Pydantic model base for smoke script contracts."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class AgentevalsLoader(_FrozenModel):
    """AgentEvals helpers loaded from an installed package or local source tree."""

    atif_to_openai_messages: ATIFConverter
    create_trajectory_match_evaluator: TrajectoryEvaluatorFactory


class MatchConfig(_FrozenModel):
    """Evaluator settings for one AgentEvals trajectory-match pass."""

    trajectory_match_mode: TrajectoryMatchMode
    tool_args_match_mode: ToolArgsMatchMode


class TrialMatchRow(_FrozenModel):
    """Report row for one Harbor trial directory."""

    task_id: str
    trial_name: str
    swebench_reward: float | None
    trajectory_match_mode: str
    tool_args_match_mode: str
    deterministic_comparison: str
    evaluator_key: str | None
    evaluator_score: bool | float | None
    match_category: str
    error: str | None
    native_path: str
    candidate_path: str

    def as_csv_row(self) -> dict[str, str]:
        """Return a CSV-friendly representation."""

        def _format_optional_number(value: float | None) -> str:
            return "" if value is None else f"{value:.6g}"

        return {
            "task_id": self.task_id,
            "trial_name": self.trial_name,
            "swebench_reward": _format_optional_number(self.swebench_reward),
            "trajectory_match_mode": self.trajectory_match_mode,
            "tool_args_match_mode": self.tool_args_match_mode,
            "deterministic_comparison": self.deterministic_comparison,
            "evaluator_key": self.evaluator_key or "",
            "evaluator_score": "" if self.evaluator_score is None else str(self.evaluator_score),
            "match_category": self.match_category,
            "error": self.error or "",
            "native_path": self.native_path,
            "candidate_path": self.candidate_path,
        }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _trial_task_id(trial_dir: Path) -> str:
    return trial_dir.name.rsplit("__", 1)[0]


def _trial_dirs(path: Path) -> list[Path]:
    if (path / "agent").is_dir():
        return [path]
    return sorted(child for child in path.iterdir() if child.is_dir() and (child / "agent").is_dir())


def _trial_reward(trial_dir: Path) -> float | None:
    result_path = trial_dir / "result.json"
    if not result_path.exists():
        return None
    try:
        result = _load_json(result_path)
    except (OSError, json.JSONDecodeError):
        return None

    reward = (((result.get("verifier_result") or {}).get("rewards") or {}).get("reward"))
    return float(reward) if isinstance(reward, int | float) else None


def _result_value(result: Any, key: str) -> Any:
    if isinstance(result, Mapping):
        return result.get(key)
    return getattr(result, key, None)


def _load_agentevals(agentevals_python_path: Path | None) -> AgentevalsLoader:
    if agentevals_python_path is not None:
        sys.path.insert(0, str(agentevals_python_path))

    try:
        from agentevals.trajectory import atif_to_openai_messages
        from agentevals.trajectory import create_trajectory_match_evaluator
    except ImportError as exc:
        raise SystemExit(
            "Could not import AgentEvals ATIF support. Install an AgentEvals build with "
            "atif_to_openai_messages, or pass --agentevals-python-path external/agentevals/python."
        ) from exc

    return AgentevalsLoader(
        atif_to_openai_messages=atif_to_openai_messages,
        create_trajectory_match_evaluator=create_trajectory_match_evaluator,
    )


def _match_category(score: Any) -> str:
    if score is True or score == 1:
        return MATCH
    return MISMATCH


def match_trial(
    trial_dir: Path,
    *,
    native_rel: Path,
    candidate_rel: Path,
    match_config: MatchConfig,
    atif_to_openai_messages: ATIFConverter,
    evaluator: TrajectoryEvaluator,
) -> TrialMatchRow:
    """Compare one Harbor trial's native ATIF and ATOF-derived ATIF artifacts."""

    native_path = trial_dir / native_rel
    candidate_path = trial_dir / candidate_rel
    task_id = _trial_task_id(trial_dir)
    reward = _trial_reward(trial_dir)

    deterministic_comparison = "missing"
    evaluator_key: str | None = None
    evaluator_score: bool | float | None = None
    match_category = MISSING
    error: str | None = None

    if not native_path.exists():
        error = "missing native trajectory"
    elif not candidate_path.exists():
        error = "missing candidate trajectory"

    if error is None:
        try:
            comparison: ToolSequenceComparison = compare_atif_tool_sequences(native_path, candidate_path)
            deterministic_comparison = comparison.classification
        except Exception as exc:  # noqa: BLE001 - keep batch comparison resilient.
            deterministic_comparison = "comparison_error"
            error = f"{type(exc).__name__}: {exc}"

    if error is None:
        try:
            native_messages = atif_to_openai_messages(_load_json(native_path))
            candidate_messages = atif_to_openai_messages(_load_json(candidate_path))
            result = evaluator(outputs=candidate_messages, reference_outputs=native_messages)
            evaluator_key = _result_value(result, "key")
            evaluator_score = _result_value(result, "score")
            match_category = _match_category(evaluator_score)
        except Exception as exc:  # noqa: BLE001 - report and continue.
            error = f"{type(exc).__name__}: {exc}"
            match_category = ERROR

    return TrialMatchRow(
        task_id=task_id,
        trial_name=trial_dir.name,
        swebench_reward=reward,
        trajectory_match_mode=match_config.trajectory_match_mode,
        tool_args_match_mode=match_config.tool_args_match_mode,
        deterministic_comparison=deterministic_comparison,
        evaluator_key=evaluator_key,
        evaluator_score=evaluator_score,
        match_category=match_category,
        error=error,
        native_path=str(native_path),
        candidate_path=str(candidate_path),
    )


def _write_csv(rows: list[TrialMatchRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].as_csv_row().keys()) if rows else list(TrialMatchRow.model_fields)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())


def _write_markdown(rows: list[TrialMatchRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _format_optional_number(value: float | None) -> str:
        return "" if value is None else f"{value:g}"

    summary = Counter(row.match_category for row in rows)
    lines = [
        "# ATIF Trajectory Match",
        "",
        "## Summary",
        "",
    ]
    if summary:
        lines.extend(f"- {category}: {count}" for category, count in sorted(summary.items()))
    else:
        lines.append("- no rows")
    lines.extend([
        "",
        "## Results",
        "",
        "| Task | Reward | Mode | Tool Args | Deterministic | AgentEvals | Category | Error |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- |",
    ])
    for row in rows:
        lines.append("| "
                     f"{row.task_id} | "
                     f"{_format_optional_number(row.swebench_reward)} | "
                     f"{row.trajectory_match_mode} | "
                     f"{row.tool_args_match_mode} | "
                     f"{row.deterministic_comparison} | "
                     f"{row.evaluator_score if row.evaluator_score is not None else ''} | "
                     f"{row.match_category} | "
                     f"{row.error or ''} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare native and ATOF-derived ATIF trajectories with AgentEvals trajectory match.",
    )
    parser.add_argument("--job-dir", required=True, type=Path, help="Harbor job dir or a single Harbor trial dir.")
    parser.add_argument("--output-dir", type=Path, help="Directory for report outputs. Defaults to --job-dir.")
    parser.add_argument(
        "--native-rel",
        type=Path,
        default=DEFAULT_NATIVE_REL,
        help="Native ATIF path relative to trial dir.",
    )
    parser.add_argument(
        "--candidate-rel",
        type=Path,
        default=DEFAULT_CANDIDATE_REL,
        help="ATOF-derived ATIF path relative to trial dir.",
    )
    parser.add_argument(
        "--agentevals-python-path",
        type=Path,
        default=Path(os.environ["AGENTEVALS_PYTHON_PATH"]) if "AGENTEVALS_PYTHON_PATH" in os.environ else None,
        help="Local AgentEvals Python source path while the ATIF adapter lives on a fork.",
    )
    parser.add_argument(
        "--trajectory-match-mode",
        choices=["strict", "unordered", "subset", "superset"],
        default="strict",
        help="AgentEvals trajectory match mode. Candidate trajectory is compared against native reference.",
    )
    parser.add_argument(
        "--tool-args-match-mode",
        choices=["exact", "ignore", "subset", "superset"],
        default="exact",
        help="AgentEvals tool argument match mode.",
    )
    parser.add_argument("--csv", type=Path, default=Path("atif-trajectory-match.csv"), help="CSV report path.")
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("atif-trajectory-match.md"),
        help="Markdown report path.",
    )
    parser.add_argument("--limit", type=int, help="Limit number of trials compared.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    loader = _load_agentevals(args.agentevals_python_path)
    match_config = MatchConfig(
        trajectory_match_mode=args.trajectory_match_mode,
        tool_args_match_mode=args.tool_args_match_mode,
    )
    evaluator = loader.create_trajectory_match_evaluator(
        trajectory_match_mode=match_config.trajectory_match_mode,
        tool_args_match_mode=match_config.tool_args_match_mode,
    )
    trial_dirs = _trial_dirs(args.job_dir)
    if args.limit is not None:
        trial_dirs = trial_dirs[:args.limit]

    rows = [
        match_trial(
            trial_dir,
            native_rel=args.native_rel,
            candidate_rel=args.candidate_rel,
            match_config=match_config,
            atif_to_openai_messages=loader.atif_to_openai_messages,
            evaluator=evaluator,
        ) for trial_dir in trial_dirs
    ]

    output_dir = args.output_dir or args.job_dir
    csv_path = args.csv if args.csv.is_absolute() else output_dir / args.csv
    markdown_path = args.markdown if args.markdown.is_absolute() else output_dir / args.markdown
    _write_csv(rows, csv_path)
    _write_markdown(rows, markdown_path)
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote {len(rows)} rows to {markdown_path}")


if __name__ == "__main__":
    main()
