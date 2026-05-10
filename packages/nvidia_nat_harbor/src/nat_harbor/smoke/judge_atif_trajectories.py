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
"""Score ATIF trajectories with AgentEvals LLM-as-judge."""

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

SCORE_SAME = "score_same"
CANDIDATE_HIGHER = "candidate_higher"
NATIVE_HIGHER = "native_higher"
NOT_SCORED = "not_scored"
ERROR = "error"

ATIFConverter = Callable[[Mapping[str, Any]], list[dict[str, Any]]]
TrajectoryJudge = Callable[..., Any]
TrajectoryJudgeFactory = Callable[..., TrajectoryJudge]


class _FrozenModel(BaseModel):
    """Frozen Pydantic model base for smoke script contracts."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class AgentevalsJudgeLoader(_FrozenModel):
    """AgentEvals helpers loaded from an installed package or local source tree."""

    atif_to_openai_messages: ATIFConverter
    create_trajectory_llm_as_judge: TrajectoryJudgeFactory
    trajectory_accuracy_prompt: str


class JudgeConfig(_FrozenModel):
    """Evaluator settings for one AgentEvals LLM-as-judge pass."""

    model: str
    continuous: bool
    score_threshold: float


class JudgeResult(_FrozenModel):
    """Normalized judge result for one ATIF artifact."""

    key: str | None
    score: bool | float | None
    numeric_score: float | None
    comment: str | None
    error: str | None


class TrialJudgeRow(_FrozenModel):
    """Report row for one Harbor trial directory."""

    task_id: str
    trial_name: str
    swebench_reward: float | None
    deterministic_comparison: str
    native_score: bool | float | None
    candidate_score: bool | float | None
    score_delta: float | None
    score_category: str
    native_comment: str | None
    candidate_comment: str | None
    native_error: str | None
    candidate_error: str | None
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
            "deterministic_comparison": self.deterministic_comparison,
            "native_score": "" if self.native_score is None else str(self.native_score),
            "candidate_score": "" if self.candidate_score is None else str(self.candidate_score),
            "score_delta": _format_optional_number(self.score_delta),
            "score_category": self.score_category,
            "native_comment": self.native_comment or "",
            "candidate_comment": self.candidate_comment or "",
            "native_error": self.native_error or "",
            "candidate_error": self.candidate_error or "",
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


def _numeric_score(score: Any) -> float | None:
    if isinstance(score, bool):
        return 1.0 if score else 0.0
    if isinstance(score, int | float):
        return float(score)
    return None


def _score_category(delta: float | None, threshold: float) -> str:
    if delta is None:
        return NOT_SCORED
    if abs(delta) <= threshold:
        return SCORE_SAME
    return CANDIDATE_HIGHER if delta > 0 else NATIVE_HIGHER


def _load_agentevals(agentevals_python_path: Path | None) -> AgentevalsJudgeLoader:
    if agentevals_python_path is not None:
        sys.path.insert(0, str(agentevals_python_path))

    try:
        from agentevals.trajectory import atif_to_openai_messages
        from agentevals.trajectory.llm import TRAJECTORY_ACCURACY_PROMPT
        from agentevals.trajectory.llm import create_trajectory_llm_as_judge
    except ImportError as exc:
        raise SystemExit(
            "Could not import AgentEvals ATIF judge support. Install an AgentEvals build with "
            "atif_to_openai_messages, or pass --agentevals-python-path external/agentevals/python."
        ) from exc

    return AgentevalsJudgeLoader(
        atif_to_openai_messages=atif_to_openai_messages,
        create_trajectory_llm_as_judge=create_trajectory_llm_as_judge,
        trajectory_accuracy_prompt=TRAJECTORY_ACCURACY_PROMPT,
    )


def _judge_artifact(
    *,
    artifact_path: Path,
    atif_to_openai_messages: ATIFConverter,
    judge: TrajectoryJudge,
) -> JudgeResult:
    try:
        messages = atif_to_openai_messages(_load_json(artifact_path))
        result = judge(outputs=messages)
    except Exception as exc:  # noqa: BLE001 - keep batch scoring resilient.
        return JudgeResult(
            key=None,
            score=None,
            numeric_score=None,
            comment=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    score = _result_value(result, "score")
    return JudgeResult(
        key=_result_value(result, "key"),
        score=score,
        numeric_score=_numeric_score(score),
        comment=_result_value(result, "comment") or _result_value(result, "reasoning"),
        error=None,
    )


def judge_trial(
    trial_dir: Path,
    *,
    native_rel: Path,
    candidate_rel: Path,
    judge_config: JudgeConfig,
    atif_to_openai_messages: ATIFConverter,
    judge: TrajectoryJudge,
) -> TrialJudgeRow:
    """Score one Harbor trial's native ATIF and ATOF-derived ATIF independently."""

    native_path = trial_dir / native_rel
    candidate_path = trial_dir / candidate_rel
    task_id = _trial_task_id(trial_dir)
    reward = _trial_reward(trial_dir)

    deterministic_comparison = "missing"
    empty_result = JudgeResult(
        key=None,
        score=None,
        numeric_score=None,
        comment=None,
        error=None,
    )
    native_result = empty_result
    candidate_result = empty_result
    native_error: str | None = None
    candidate_error: str | None = None

    if not native_path.exists():
        native_error = "missing native trajectory"
    if not candidate_path.exists():
        candidate_error = "missing candidate trajectory"

    if native_path.exists() and candidate_path.exists():
        try:
            comparison: ToolSequenceComparison = compare_atif_tool_sequences(native_path, candidate_path)
            deterministic_comparison = comparison.classification
        except Exception as exc:  # noqa: BLE001 - report and continue.
            deterministic_comparison = "comparison_error"
            native_error = native_error or f"{type(exc).__name__}: {exc}"

    if native_error is None:
        native_result = _judge_artifact(
            artifact_path=native_path,
            atif_to_openai_messages=atif_to_openai_messages,
            judge=judge,
        )
        native_error = native_result.error

    if candidate_error is None:
        candidate_result = _judge_artifact(
            artifact_path=candidate_path,
            atif_to_openai_messages=atif_to_openai_messages,
            judge=judge,
        )
        candidate_error = candidate_result.error

    score_delta = (
        candidate_result.numeric_score - native_result.numeric_score
        if candidate_result.numeric_score is not None and native_result.numeric_score is not None
        else None
    )
    score_category = (
        ERROR
        if native_error or candidate_error
        else _score_category(score_delta, judge_config.score_threshold)
    )
    return TrialJudgeRow(
        task_id=task_id,
        trial_name=trial_dir.name,
        swebench_reward=reward,
        deterministic_comparison=deterministic_comparison,
        native_score=native_result.score,
        candidate_score=candidate_result.score,
        score_delta=score_delta,
        score_category=score_category,
        native_comment=native_result.comment,
        candidate_comment=candidate_result.comment,
        native_error=native_error,
        candidate_error=candidate_error,
        native_path=str(native_path),
        candidate_path=str(candidate_path),
    )


def _write_csv(rows: list[TrialJudgeRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].as_csv_row().keys()) if rows else list(TrialJudgeRow.model_fields)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())


def _write_markdown(rows: list[TrialJudgeRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _format_optional_number(value: float | None) -> str:
        return "" if value is None else f"{value:g}"

    summary = Counter(row.score_category for row in rows)
    lines = [
        "# ATIF Trajectory Judge Scores",
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
        "| Task | Reward | Deterministic | Native | Candidate | Delta | Category | Errors |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- | --- |",
    ])
    for row in rows:
        errors = "; ".join(error for error in [row.native_error, row.candidate_error] if error)
        lines.append("| "
                     f"{row.task_id} | "
                     f"{_format_optional_number(row.swebench_reward)} | "
                     f"{row.deterministic_comparison} | "
                     f"{row.native_score if row.native_score is not None else ''} | "
                     f"{row.candidate_score if row.candidate_score is not None else ''} | "
                     f"{_format_optional_number(row.score_delta)} | "
                     f"{row.score_category} | "
                     f"{errors} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _default_model() -> str | None:
    return os.environ.get("AGENTEVALS_TRAJECTORY_JUDGE_MODEL") or os.environ.get("NAT_HARBOR_TRAJECTORY_JUDGE_MODEL")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score ATIF trajectories independently with AgentEvals LLM-as-judge.",
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
        help="Candidate ATIF path relative to trial dir.",
    )
    parser.add_argument(
        "--agentevals-python-path",
        type=Path,
        default=Path(os.environ["AGENTEVALS_PYTHON_PATH"]) if "AGENTEVALS_PYTHON_PATH" in os.environ else None,
        help="Local AgentEvals Python source path while the ATIF adapter lives on a fork.",
    )
    parser.add_argument(
        "--model",
        default=_default_model(),
        help="LangChain model identifier for AgentEvals, e.g. openai:o3-mini.",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Ask the judge for a continuous 0..1 score instead of a boolean score.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.05,
        help="Absolute delta threshold for score_same classification.",
    )
    parser.add_argument("--csv", type=Path, default=Path("atif-trajectory-judge.csv"), help="CSV report path.")
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("atif-trajectory-judge.md"),
        help="Markdown report path.",
    )
    parser.add_argument("--limit", type=int, help="Limit number of trials scored.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.model:
        raise SystemExit("Provide --model or set AGENTEVALS_TRAJECTORY_JUDGE_MODEL.")

    loader = _load_agentevals(args.agentevals_python_path)
    judge_config = JudgeConfig(
        model=args.model,
        continuous=args.continuous,
        score_threshold=args.score_threshold,
    )
    judge = loader.create_trajectory_llm_as_judge(
        prompt=loader.trajectory_accuracy_prompt,
        model=judge_config.model,
        continuous=judge_config.continuous,
    )
    trial_dirs = _trial_dirs(args.job_dir)
    if args.limit is not None:
        trial_dirs = trial_dirs[:args.limit]

    rows = [
        judge_trial(
            trial_dir,
            native_rel=args.native_rel,
            candidate_rel=args.candidate_rel,
            judge_config=judge_config,
            atif_to_openai_messages=loader.atif_to_openai_messages,
            judge=judge,
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
