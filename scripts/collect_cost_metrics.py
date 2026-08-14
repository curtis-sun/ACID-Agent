#!/usr/bin/env python3
"""Collect per-task KramaBench cost metrics into CSV/JSONL.

The script reads task-level result.json files produced by run_kramabench.py.
It intentionally does not evaluate correctness; it only extracts basic run
metadata and cost/runtime counters.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from da_agent.utils.runtime_config import KRAMABENCH_OUTPUT_DIR


BASE_COLUMNS = [
    "run_id",
    "agent_type",
    "model",
    "domain",
    "task_id",
    "finished",
    "answer",
    "score",
    "total_wall_time_sec",
    "llm_call_count",
    "usage_event_count",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cached_input_tokens",
    "step_count",
    "turn_count",
    "code_execution_count",
]


OPTIONAL_COLUMNS = [
    "result_json",
    "trace_jsonl",
    "trace_summary_json",
    "result_file_added_count",
    "result_file_changed_count",
    "action_count",
    "bash_action_count",
    "think_action_count",
    "error_action_count",
    "turn_failed_count",
]


EVENT_COLUMNS = [
    "run_id",
    "agent_type",
    "model",
    "domain",
    "task_id",
    "event_index",
    "step_id",
    "step_index",
    "attempt_index",
    "phase",
    "event_model",
    "start_time",
    "end_time",
    "duration_sec",
    "count_as_turn",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cached_input_tokens",
    "raw_usage_json",
    "result_json",
]


SCORE_METRIC_PRIORITY = [
    "success",
    "numeric_scientific_exact",
    "llm_paraphrase",
    "rae_score",
    "f1",
    "f1_approximate",
]


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    except Exception:
        return []
    return rows


def clean_final_answer_text(text: str) -> str:
    """Keep only the value after a final-answer marker when present."""
    text = (text or "").strip()
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text

    markers = ("final_answer:", "final answer:")
    for line in reversed(lines):
        lower = line.lower()
        for marker in markers:
            if lower.startswith(marker):
                return line.split(":", 1)[1].strip().strip("`")

    return text


def parse_score_value(value: Any) -> Any:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    lower = text.lower()
    if lower == "true":
        return 1.0
    if lower == "false":
        return 0.0
    try:
        return float(text)
    except Exception:
        return text


def load_evaluation_scores(path: str) -> Dict[str, Any]:
    if not path:
        return {}

    measures_path = Path(path).expanduser()
    if not measures_path.exists():
        raise FileNotFoundError(f"evaluation measures file not found: {measures_path}")

    by_task_metric: Dict[str, Dict[str, Any]] = {}
    with measures_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = str(row.get("task_id") or "").strip()
            metric = str(row.get("metric") or "").strip()
            if not task_id or metric not in SCORE_METRIC_PRIORITY:
                continue
            by_task_metric.setdefault(task_id, {})[metric] = parse_score_value(row.get("value"))

    scores: Dict[str, Any] = {}
    for task_id, metric_values in by_task_metric.items():
        for metric in SCORE_METRIC_PRIORITY:
            if metric in metric_values:
                scores[task_id] = metric_values[metric]
                break
    return scores


def infer_run_metadata(result_path: Path, output_root: Path) -> Dict[str, str]:
    rel_parts = result_path.relative_to(output_root).parts
    run_id = rel_parts[0] if len(rel_parts) >= 1 else ""
    domain = rel_parts[1] if len(rel_parts) >= 2 else ""
    task_id = rel_parts[2] if len(rel_parts) >= 3 else result_path.parent.name

    model = ""
    agent_type = ""
    if run_id:
        known_agents = ("codex", "claude-code", "acid")
        for marker in known_agents:
            needle = f"-{marker}-"
            if needle in run_id:
                agent_type = marker
                model = run_id.split(needle, 1)[0]
                break
        if not agent_type and run_id.startswith("ACIDAgent-"):
            agent_type = "acid"
            model = run_id[len("ACIDAgent-"):]

    return {
        "run_id": run_id,
        "agent_type": agent_type,
        "model": model,
        "domain": domain,
        "task_id": task_id,
    }


def result_files_count(result_files: Any, key: str) -> int:
    if not isinstance(result_files, dict):
        return 0
    value = result_files.get(key)
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        return len(value)
    return 0


def count_actions(trace_summary: Dict[str, Any]) -> Dict[str, int]:
    actions = trace_summary.get("actions")
    if not isinstance(actions, list):
        return {
            "action_count": 0,
            "bash_action_count": 0,
            "think_action_count": 0,
            "error_action_count": 0,
            "turn_failed_count": 0,
        }

    counts = {
        "action_count": len(actions),
        "bash_action_count": 0,
        "think_action_count": 0,
        "error_action_count": 0,
        "turn_failed_count": 0,
    }
    for item in actions:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or item.get("type") or "")
        if action == "Bash":
            counts["bash_action_count"] += 1
        elif action == "Think":
            counts["think_action_count"] += 1
        elif action == "Error":
            counts["error_action_count"] += 1
        elif action == "TurnFailed":
            counts["turn_failed_count"] += 1
    return counts


def find_trace_summary(result_path: Path, result_data: Dict[str, Any]) -> Optional[Path]:
    trace_files = result_data.get("trace_files")
    if isinstance(trace_files, dict):
        summary_name = trace_files.get("summary")
        if summary_name:
            candidate = result_path.parent / str(summary_name)
            if candidate.exists():
                return candidate

    for filename in (
        "codex_trace_summary.json",
        "claude_trace_summary.json",
        "acid_trace_summary.json",
        "da_agent_trace_summary.json",
    ):
        candidate = result_path.parent / filename
        if candidate.exists():
            return candidate
    return None


def find_trace_jsonl(result_path: Path, result_data: Dict[str, Any]) -> Optional[Path]:
    trace_files = result_data.get("trace_files")
    if isinstance(trace_files, dict):
        jsonl_name = trace_files.get("jsonl")
        if jsonl_name:
            candidate = result_path.parent / str(jsonl_name)
            if candidate.exists():
                return candidate

    for filename in (
        "codex_trace.jsonl",
        "claude_trace.jsonl",
        "acid_trace.jsonl",
        "da_agent_trace.jsonl",
    ):
        candidate = result_path.parent / filename
        if candidate.exists():
            return candidate
    return None


def find_cost_metric_events(result_path: Path, result_data: Dict[str, Any]) -> Optional[Path]:
    cost_metric_files = result_data.get("cost_metric_files")
    if isinstance(cost_metric_files, dict):
        events_name = cost_metric_files.get("events")
        if events_name:
            candidate = result_path.parent / str(events_name)
            if candidate.exists():
                return candidate

    candidate = result_path.parent / "cost_metrics_events.jsonl"
    if candidate.exists():
        return candidate
    return None


def extract_row(
    result_path: Path,
    output_root: Path,
    score_by_task: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result_data = load_json(result_path)
    metadata = infer_run_metadata(result_path, output_root)

    trace_summary_path = find_trace_summary(result_path, result_data)
    trace_summary: Dict[str, Any] = {}
    if trace_summary_path:
        trace_summary = load_json(trace_summary_path)

    metrics = result_data.get("cost_metrics")
    if not isinstance(metrics, dict) or not metrics:
        metrics = trace_summary.get("cost_metrics")
    if not isinstance(metrics, dict):
        metrics = {}

    result = result_data.get("result", trace_summary.get("result", ""))
    if isinstance(result, (dict, list)):
        answer = json.dumps(result, ensure_ascii=False)
    else:
        answer = "" if result is None else str(result)
    answer = clean_final_answer_text(answer)

    result_files = result_data.get("result_files")
    if not isinstance(result_files, dict):
        result_files = trace_summary.get("result_files")

    row: Dict[str, Any] = {
        **metadata,
        "finished": result_data.get("finished", trace_summary.get("finished", "")),
        "answer": answer,
        "score": (score_by_task or {}).get(metadata.get("task_id", ""), ""),
        "total_wall_time_sec": metrics.get("total_wall_time_sec", ""),
        "llm_call_count": metrics.get("llm_call_count", ""),
        "usage_event_count": metrics.get("usage_event_count", ""),
        "total_tokens": metrics.get("total_tokens", ""),
        "input_tokens": metrics.get("input_tokens", ""),
        "output_tokens": metrics.get("output_tokens", ""),
        "reasoning_tokens": metrics.get("reasoning_tokens", ""),
        "cached_input_tokens": metrics.get("cached_input_tokens", ""),
        "step_count": metrics.get("step_count", result_data.get("steps", trace_summary.get("steps", ""))),
        "turn_count": metrics.get("turn_count", ""),
        "code_execution_count": metrics.get("code_execution_count", ""),
        "result_json": str(result_path),
        "trace_jsonl": "",
        "trace_summary_json": "",
        "result_file_added_count": result_files_count(result_files, "added_files"),
        "result_file_changed_count": result_files_count(result_files, "changed_files"),
        "action_count": "",
        "bash_action_count": "",
        "think_action_count": "",
        "error_action_count": "",
        "turn_failed_count": "",
    }

    trace_jsonl_path = find_trace_jsonl(result_path, result_data)
    if trace_jsonl_path:
        row["trace_jsonl"] = str(trace_jsonl_path)

    if trace_summary_path:
        row["trace_summary_json"] = str(trace_summary_path)
        for key, value in count_actions(trace_summary).items():
            row[key] = value
        if not row["turn_count"] and isinstance(trace_summary.get("trajectory"), list):
            row["turn_count"] = sum(
                1 for item in trace_summary["trajectory"]
                if isinstance(item, dict) and item.get("type") == "turn.completed"
            )

    return row


def extract_event_rows(result_path: Path, output_root: Path) -> List[Dict[str, Any]]:
    result_data = load_json(result_path)
    events_path = find_cost_metric_events(result_path, result_data)
    if not events_path:
        return []

    metadata = infer_run_metadata(result_path, output_root)
    rows: List[Dict[str, Any]] = []
    for event in load_jsonl(events_path):
        raw_usage = event.get("raw_usage")
        rows.append({
            **metadata,
            "event_index": event.get("event_index", ""),
            "step_id": event.get("step_id", ""),
            "step_index": event.get("step_index", ""),
            "attempt_index": event.get("attempt_index", ""),
            "phase": event.get("phase", ""),
            "event_model": event.get("model", ""),
            "start_time": event.get("start_time", ""),
            "end_time": event.get("end_time", ""),
            "duration_sec": event.get("duration_sec", ""),
            "count_as_turn": event.get("count_as_turn", ""),
            "total_tokens": event.get("total_tokens", ""),
            "input_tokens": event.get("input_tokens", ""),
            "output_tokens": event.get("output_tokens", ""),
            "reasoning_tokens": event.get("reasoning_tokens", ""),
            "cached_input_tokens": event.get("cached_input_tokens", ""),
            "raw_usage_json": json.dumps(raw_usage, ensure_ascii=False)
            if raw_usage not in (None, "") else "",
            "result_json": str(result_path),
        })
    return rows


def iter_result_paths(output_root: Path, run_id: str = "") -> Iterable[Path]:
    search_root = output_root / run_id if run_id else output_root
    if not search_root.exists():
        return []
    return sorted(
        path for path in search_root.rglob("result.json")
        if path.is_file()
    )


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    columns = BASE_COLUMNS + OPTIONAL_COLUMNS
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def write_jsonl(rows: List[Dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_events_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in EVENT_COLUMNS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect basic KramaBench run metadata and cost metrics."
    )
    parser.add_argument(
        "--output-root",
        default=KRAMABENCH_OUTPUT_DIR,
        help="KramaBench output root containing run_id/domain/task_id/result.json.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional single run id under output-root. If omitted, scan all runs.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--jsonl",
        default="",
        help="Optional JSONL output path with the same rows.",
    )
    parser.add_argument(
        "--events-out",
        default="",
        help="Optional per-LLM-call events CSV output path.",
    )
    parser.add_argument(
        "--evaluation-measures",
        default="",
        help=(
            "Optional Kramabench evaluation measures CSV. When provided, add a "
            "per-task score field using success/llm_paraphrase/rae_score/f1."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    result_paths = list(iter_result_paths(output_root, args.run_id))
    score_by_task = load_evaluation_scores(args.evaluation_measures)
    rows = [extract_row(path, output_root, score_by_task) for path in result_paths]

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_path)

    if args.jsonl:
        jsonl_path = Path(args.jsonl).expanduser()
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(rows, jsonl_path)

    if args.events_out:
        event_rows: List[Dict[str, Any]] = []
        for path in result_paths:
            event_rows.extend(extract_event_rows(path, output_root))
        events_out_path = Path(args.events_out).expanduser()
        events_out_path.parent.mkdir(parents=True, exist_ok=True)
        write_events_csv(event_rows, events_out_path)
        print(f"Wrote {len(event_rows)} event rows to {events_out_path}")

    print(f"Wrote {len(rows)} rows to {out_path}")
    if args.jsonl:
        print(f"Wrote JSONL to {args.jsonl}")


if __name__ == "__main__":
    main()
