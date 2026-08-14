#!/usr/bin/env python3
"""Build a Claude majority/adjudicated KramaBench response cache.

For each task, this script combines three Claude Code baseline runs:

1. If at least two cleaned answers match, select the majority answer with no
   extra model call.
2. If all three cleaned answers differ, call Claude once with the three
   candidate answers and trace excerpts, forcing it to choose exactly one of
   the three candidates.

The output is a KramaBench-compatible response cache plus task/cost summaries.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TOKEN_FIELDS = (
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cached_input_tokens",
)

METRIC_SUM_FIELDS = (
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
)

TASK_COLUMNS = [
    "domain",
    "task_id",
    "decision_mode",
    "selected_answer",
    "selected_choice",
    "run1_answer",
    "run2_answer",
    "run3_answer",
    "run1_norm",
    "run2_norm",
    "run3_norm",
    "adjudication_reason",
    "base_total_tokens",
    "adjudication_total_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cached_input_tokens",
    "summed_wall_time_sec",
    "critical_path_wall_time_sec",
    "llm_call_count",
    "turn_count",
    "code_execution_count",
]


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    to_dict = getattr(value, "model_dump", None)
    if callable(to_dict):
        try:
            return jsonable(to_dict())
        except Exception:
            pass
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return jsonable(to_dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return jsonable(vars(value))
        except Exception:
            pass
    return str(value)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_response_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"response cache must be a JSON list: {path}")
    by_task: Dict[str, Dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id") or "").strip()
        if task_id:
            by_task[task_id] = item
    return by_task


def clean_final_answer_text(text: Any) -> str:
    if isinstance(text, (dict, list)):
        return json.dumps(text, ensure_ascii=False)
    text = "" if text is None else str(text)
    text = text.strip()
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    markers = ("final_answer:", "final answer:")
    for line in reversed(lines):
        lower = line.lower()
        for marker in markers:
            if lower.startswith(marker):
                return line.split(":", 1)[1].strip().strip("`").strip()
    return text.strip("`").strip()


def extract_answer(cache_item: Dict[str, Any]) -> str:
    model_output = cache_item.get("model_output")
    if isinstance(model_output, dict):
        answer = model_output.get("answer")
    else:
        answer = cache_item.get("result", "")
    return clean_final_answer_text(answer)


def _canonical_decimal(text: str) -> Optional[str]:
    raw = text.strip()
    if not raw:
        return None
    if re.search(r"[A-Za-z%]", raw):
        return None
    try:
        dec = Decimal(raw.replace(",", ""))
    except (InvalidOperation, ValueError):
        return None
    normalized = dec.normalize()
    # Avoid scientific notation for ordinary values.
    return format(normalized, "f").rstrip("0").rstrip(".") or "0"


def normalize_answer_for_vote(answer: str) -> str:
    text = clean_final_answer_text(answer)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("`").strip()

    numeric = _canonical_decimal(text)
    if numeric is not None:
        return f"number:{numeric}"

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return "json:" + json.dumps(parsed, ensure_ascii=False, sort_keys=True)
    except Exception:
        pass

    lower = text.casefold()
    if lower in ("true", "false"):
        return lower
    return lower


def number(value: Any) -> float:
    try:
        return float(value or 0)
    except Exception:
        return 0.0


def int_number(value: Any) -> int:
    try:
        return int(float(value or 0))
    except Exception:
        return 0


def get_cost_metrics(cache_item: Dict[str, Any]) -> Dict[str, Any]:
    metrics = cache_item.get("cost_metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    result = dict(metrics)
    result.setdefault("total_tokens", cache_item.get("token_usage_sut", 0))
    result.setdefault("input_tokens", cache_item.get("token_usage_sut_input", 0))
    result.setdefault("output_tokens", cache_item.get("token_usage_sut_output", 0))
    result.setdefault("total_wall_time_sec", cache_item.get("runtime", 0))
    return result


def add_metric_totals(items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    total: Dict[str, Any] = {key: 0 for key in METRIC_SUM_FIELDS}
    for metrics in items:
        for key in METRIC_SUM_FIELDS:
            if key in ("llm_call_count", "usage_event_count", "step_count", "turn_count", "code_execution_count"):
                total[key] += int_number(metrics.get(key))
            else:
                total[key] += number(metrics.get(key))
    for key, value in list(total.items()):
        if isinstance(value, float):
            total[key] = round(value, 3)
    return total


def task_order_from_workload(path: Optional[Path]) -> List[str]:
    if not path:
        return []
    data = load_json(path)
    tasks = data if isinstance(data, list) else data.get("tasks", []) if isinstance(data, dict) else []
    order = []
    for item in tasks:
        if isinstance(item, dict) and item.get("id"):
            order.append(str(item["id"]))
    return order


def compact_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    omitted = len(text) - head - tail
    return text[:head].rstrip() + f"\n\n[... omitted {omitted} chars ...]\n\n" + text[-tail:].lstrip()


def find_task_dir(run_output: Path, domain: str, task_id: str) -> Optional[Path]:
    candidates = [
        run_output / domain / task_id,
        run_output / task_id,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(run_output.rglob(f"{task_id}/result.json"))
    if matches:
        return matches[0].parent
    return None


def load_trace_context(
    run_output: Path,
    domain: str,
    task_id: str,
    *,
    max_chars: int,
) -> Tuple[str, Dict[str, str]]:
    task_dir = find_task_dir(run_output, domain, task_id)
    if not task_dir:
        return f"Trace unavailable: task directory not found under {run_output}", {}

    paths = {
        "task_dir": str(task_dir),
        "summary": "",
        "jsonl": "",
        "result_json": "",
    }
    result_json = task_dir / "result.json"
    if result_json.exists():
        paths["result_json"] = str(result_json)

    summary_path = None
    for name in ("claude_trace_summary.json", "codex_trace_summary.json", "acid_trace_summary.json"):
        candidate = task_dir / name
        if candidate.exists():
            summary_path = candidate
            break
    if summary_path:
        paths["summary"] = str(summary_path)
        summary = load_json(summary_path)
        if not isinstance(summary, dict):
            summary = {}
        lines = [
            f"trace_summary_file: {summary_path}",
            f"finished: {summary.get('finished', '')}",
            f"result: {summary.get('result', '')}",
            f"steps: {summary.get('steps', '')}",
        ]
        actions = summary.get("actions")
        if not isinstance(actions, list):
            actions = summary.get("trajectory") if isinstance(summary.get("trajectory"), list) else []
        lines.append("actions:")
        used = 0
        for idx, action in enumerate(actions):
            if not isinstance(action, dict):
                continue
            label = action.get("action") or action.get("type") or action.get("action_type") or "event"
            content = action.get("content") or action.get("code") or action.get("action") or ""
            observation = (
                action.get("observation")
                or action.get("observations")
                or action.get("result")
                or action.get("raw_observation")
                or ""
            )
            entry = (
                f"\n[{idx}] {label}\n"
                f"content:\n{compact_text(str(content), 1400)}\n"
                f"observation/result:\n{compact_text(str(observation), 1400)}\n"
            )
            lines.append(entry)
            used += len(entry)
            if used >= max_chars:
                break
        return compact_text("\n".join(lines), max_chars), paths

    jsonl_path = None
    for name in ("claude_trace.jsonl", "codex_trace.jsonl", "acid_trace.jsonl"):
        candidate = task_dir / name
        if candidate.exists():
            jsonl_path = candidate
            break
    if jsonl_path:
        paths["jsonl"] = str(jsonl_path)
        text = jsonl_path.read_text(encoding="utf-8", errors="replace")
        return compact_text(text, max_chars), paths

    if result_json.exists():
        data = load_json(result_json)
        return compact_text(json.dumps(data, ensure_ascii=False, indent=2), max_chars), paths

    return f"Trace unavailable: no trace files found under {task_dir}", paths


def build_adjudication_prompt(
    *,
    domain: str,
    task_id: str,
    task: str,
    candidates: List[str],
    trace_contexts: List[str],
) -> str:
    labels = ["A", "B", "C"]
    candidate_block = "\n".join(
        f"{label}: {answer}" for label, answer in zip(labels, candidates)
    )
    traces = []
    for label, answer, context in zip(labels, candidates, trace_contexts):
        traces.append(
            f"## Candidate {label}\n"
            f"Answer: {answer}\n"
            f"Trace excerpt:\n{context}"
        )

    return f"""You are adjudicating three independent Claude Code runs for one data-analysis benchmark task.

Choose exactly one of the three candidate answers. Do not compute a fourth answer.
Choose the answer whose execution path is best supported by the task wording, schema evidence, data evidence, and executed code path.
If multiple paths are plausible, prefer the one with the clearest direct support from the data schema and task wording.

# Domain
{domain}

# Task ID
{task_id}

# Task
{task}

# Candidate Answers
{candidate_block}

# Candidate Trace Evidence
{chr(10).join(traces)}

Return exactly this format:
CHOICE: <A|B|C>
FINAL_ANSWER: <copy the selected candidate answer exactly>
REASON: <one concise paragraph explaining why the selected path is best supported>
"""


def usage_value(usage: Any, *names: str) -> int:
    current = usage
    for name in names:
        if current is None:
            return 0
        if isinstance(current, dict):
            current = current.get(name)
        else:
            current = getattr(current, name, None)
    try:
        return int(current or 0)
    except Exception:
        return 0


def canonical_usage_fields(usage: Dict[str, Any], model_usage: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    input_tokens = usage_value(usage, "prompt_tokens") or usage_value(usage, "input_tokens")
    output_tokens = usage_value(usage, "completion_tokens") or usage_value(usage, "output_tokens")
    cached_input_tokens = (
        usage_value(usage, "prompt_tokens_details", "cached_tokens")
        or usage_value(usage, "input_tokens_details", "cached_tokens")
        or usage_value(usage, "cached_input_tokens")
        or usage_value(usage, "cache_read_input_tokens")
        or usage_value(usage, "cached_tokens")
    )
    reasoning_tokens = (
        usage_value(usage, "completion_tokens_details", "reasoning_tokens")
        or usage_value(usage, "output_tokens_details", "reasoning_tokens")
        or usage_value(usage, "reasoning_tokens")
        or usage_value(usage, "reasoning_output_tokens")
    )

    if model_usage and not (input_tokens or output_tokens):
        input_tokens = usage_value(model_usage, "inputTokens")
        output_tokens = usage_value(model_usage, "outputTokens")
        cached_input_tokens = usage_value(model_usage, "cacheReadInputTokens")

    total_tokens = usage_value(usage, "total_tokens")
    if not total_tokens:
        total_tokens = input_tokens + output_tokens

    return {
        "total_tokens": total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cached_input_tokens": cached_input_tokens,
    }


def first_model_usage(result_event: Dict[str, Any], model: str) -> Dict[str, Any]:
    model_usage = result_event.get("modelUsage")
    if not isinstance(model_usage, dict):
        return {}
    if isinstance(model_usage.get(model), dict):
        return model_usage[model]
    for value in model_usage.values():
        if isinstance(value, dict):
            return value
    return {}


def claude_env() -> Dict[str, str]:
    env = dict(os.environ)
    if not env.get("ANTHROPIC_API_KEY") and env.get("DASHSCOPE_API_KEY"):
        env["ANTHROPIC_API_KEY"] = env["DASHSCOPE_API_KEY"]
    if not env.get("ANTHROPIC_API_KEY") and env.get("OPENAI_API_KEY"):
        env["ANTHROPIC_API_KEY"] = env["OPENAI_API_KEY"]
    env.setdefault("ANTHROPIC_BASE_URL", "https://dashscope.aliyuncs.com/apps/anthropic")
    return env


def claude_command(
    prompt: str,
    model: str,
    runner: str,
    image: str,
    user: str,
) -> List[str]:
    claude_args = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        model,
        "--dangerously-skip-permissions",
    ]
    if runner == "host":
        return claude_args

    cmd = [
        "docker",
        "run",
        "--rm",
        "--user",
        user,
        "-e",
        "ANTHROPIC_API_KEY",
        "-e",
        "ANTHROPIC_BASE_URL",
        image,
    ]
    return cmd + claude_args


def run_claude_adjudication(
    prompt: str,
    model: str,
    timeout_sec: int,
    runner: str,
    image: str,
    user: str,
) -> Dict[str, Any]:
    env = claude_env()
    cmd = claude_command(prompt, model, runner, image, user)
    start = time.time()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_sec,
        env=env,
    )
    end = time.time()

    events: List[Dict[str, Any]] = []
    result_event: Dict[str, Any] = {}
    raw_lines = []
    for idx, line in enumerate(proc.stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        raw_lines.append(line)
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            event = {"type": "raw", "content": line}
        if isinstance(event, dict):
            event.setdefault("sequence_index", idx)
            events.append(event)
            if event.get("type") == "result":
                result_event = event

    text = str(result_event.get("result") or "") if result_event else proc.stdout
    usage = result_event.get("usage") if isinstance(result_event.get("usage"), dict) else {}
    fields = canonical_usage_fields(usage, first_model_usage(result_event, model))
    duration_sec = round(max(0.0, end - start), 3)
    metrics = {
        "total_wall_time_sec": duration_sec,
        "llm_call_count": int(result_event.get("num_turns") or 1) if result_event else 1,
        "usage_event_count": 1 if usage or first_model_usage(result_event, model) else 0,
        **fields,
        "step_count": 1,
        "turn_count": int(result_event.get("num_turns") or 1) if result_event else 1,
        "code_execution_count": 0,
    }

    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "events": events,
        "result_event": result_event,
        "text": text,
        "metrics": metrics,
        "start_time": start,
        "end_time": end,
    }


def parse_adjudication_choice(text: str, candidates: List[str]) -> Tuple[str, str, str]:
    choice_match = re.search(r"(?im)^\s*CHOICE\s*:\s*([ABC])\s*$", text or "")
    final_match = re.search(r"(?im)^\s*FINAL_ANSWER\s*:\s*(.+?)\s*$", text or "")
    reason_match = re.search(r"(?ims)^\s*REASON\s*:\s*(.+)$", text or "")
    choice = choice_match.group(1).upper() if choice_match else ""
    final_answer = clean_final_answer_text(final_match.group(1)) if final_match else ""
    reason = reason_match.group(1).strip() if reason_match else ""

    labels = {"A": 0, "B": 1, "C": 2}
    if choice in labels:
        return choice, candidates[labels[choice]], reason

    final_norm = normalize_answer_for_vote(final_answer)
    for idx, candidate in enumerate(candidates):
        if normalize_answer_for_vote(candidate) == final_norm:
            return "ABC"[idx], candidate, reason

    raise ValueError(f"adjudicator did not select one of the candidates: {text[:500]}")


def load_task_texts(workload_path: Optional[Path]) -> Dict[str, str]:
    if not workload_path:
        return {}
    data = load_json(workload_path)
    tasks = data if isinstance(data, list) else data.get("tasks", []) if isinstance(data, dict) else []
    result = {}
    for item in tasks:
        if isinstance(item, dict) and item.get("id"):
            result[str(item["id"])] = str(item.get("query") or item.get("question") or "")
    return result


def determine_task_order(
    run_maps: List[Dict[str, Dict[str, Any]]],
    workload_path: Optional[Path],
) -> List[str]:
    workload_order = task_order_from_workload(workload_path)
    all_tasks = set()
    for run_map in run_maps:
        all_tasks.update(run_map.keys())
    ordered = [task_id for task_id in workload_order if task_id in all_tasks]
    ordered.extend(sorted(all_tasks - set(ordered)))
    return ordered


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TASK_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in TASK_COLUMNS})


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a majority/adjudicated response cache from three Claude Code runs."
    )
    parser.add_argument("--domain", required=True)
    parser.add_argument("--run1-cache", required=True)
    parser.add_argument("--run2-cache", required=True)
    parser.add_argument("--run3-cache", required=True)
    parser.add_argument("--run1-output", default="")
    parser.add_argument("--run2-output", default="")
    parser.add_argument("--run3-output", default="")
    parser.add_argument("--workload", default="")
    parser.add_argument("--model", default="qwen3.5-397b-a17b")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--response-cache-out", default="")
    parser.add_argument("--sut-name", default="")
    parser.add_argument("--kramabench-results-dir", default="Kramabench/results")
    parser.add_argument("--max-trace-chars", type=int, default=12000)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument(
        "--claude-runner",
        choices=["docker", "host"],
        default="docker",
        help="Run Claude from the claude-code Docker image by default, matching the Claude baseline.",
    )
    parser.add_argument("--claude-image", default="claude-code")
    parser.add_argument("--claude-user", default=os.getenv("DA_AGENT_DOCKER_USER", "agent"))
    parser.add_argument(
        "--on-adjudication-error",
        choices=["fail", "run1"],
        default="fail",
        help="What to do if all answers differ and Claude adjudication fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domain = args.domain
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_paths = [
        Path(args.run1_cache).expanduser().resolve(),
        Path(args.run2_cache).expanduser().resolve(),
        Path(args.run3_cache).expanduser().resolve(),
    ]
    output_dirs = [
        Path(args.run1_output).expanduser().resolve() if args.run1_output else None,
        Path(args.run2_output).expanduser().resolve() if args.run2_output else None,
        Path(args.run3_output).expanduser().resolve() if args.run3_output else None,
    ]
    workload_path = Path(args.workload).expanduser().resolve() if args.workload else None

    run_maps = [load_response_cache(path) for path in cache_paths]
    task_texts = load_task_texts(workload_path)
    task_ids = determine_task_order(run_maps, workload_path)

    response_rows: List[Dict[str, Any]] = []
    task_rows: List[Dict[str, Any]] = []
    adjudication_events: List[Dict[str, Any]] = []
    adjudication_prompts: List[Dict[str, Any]] = []

    for task_id in task_ids:
        missing = [idx + 1 for idx, run_map in enumerate(run_maps) if task_id not in run_map]
        if missing:
            raise ValueError(f"task {task_id} missing from run(s): {missing}")

        items = [run_map[task_id] for run_map in run_maps]
        answers = [extract_answer(item) for item in items]
        norms = [normalize_answer_for_vote(answer) for answer in answers]
        counts = Counter(norms)
        base_metrics = [get_cost_metrics(item) for item in items]
        base_total = add_metric_totals(base_metrics)
        base_wall_times = [number(metrics.get("total_wall_time_sec")) for metrics in base_metrics]

        selected_answer = ""
        selected_choice = ""
        decision_mode = ""
        reason = ""
        adjudication_metrics = {key: 0 for key in METRIC_SUM_FIELDS}

        majority_norm, majority_count = counts.most_common(1)[0]
        if majority_count >= 2:
            selected_idx = norms.index(majority_norm)
            selected_answer = answers[selected_idx]
            selected_choice = "ABC"[selected_idx]
            decision_mode = "majority"
            reason = f"At least two runs produced the same normalized answer: {selected_answer}"
        else:
            if any(path is None for path in output_dirs):
                message = (
                    f"task {task_id} has three different answers but output dirs were not all provided"
                )
                if args.on_adjudication_error == "fail":
                    raise ValueError(message)
                selected_answer = answers[0]
                selected_choice = "A"
                decision_mode = "adjudication_failed_fallback_run1"
                reason = message
            else:
                trace_contexts = []
                trace_paths = []
                for output_dir in output_dirs:
                    assert output_dir is not None
                    context, paths = load_trace_context(
                        output_dir,
                        domain,
                        task_id,
                        max_chars=args.max_trace_chars,
                    )
                    trace_contexts.append(context)
                    trace_paths.append(paths)

                task_text = task_texts.get(task_id) or str(items[0].get("task") or items[0].get("Task") or "")
                prompt = build_adjudication_prompt(
                    domain=domain,
                    task_id=task_id,
                    task=task_text,
                    candidates=answers,
                    trace_contexts=trace_contexts,
                )
                adjudication_prompts.append({
                    "task_id": task_id,
                    "answers": {"A": answers[0], "B": answers[1], "C": answers[2]},
                    "trace_paths": trace_paths,
                    "prompt": prompt,
                })

                try:
                    adjudication = run_claude_adjudication(
                        prompt,
                        args.model,
                        args.timeout_sec,
                        args.claude_runner,
                        args.claude_image,
                        args.claude_user,
                    )
                    if adjudication["returncode"] != 0:
                        raise RuntimeError(adjudication["stdout"][-2000:])
                    choice, selected_answer, reason = parse_adjudication_choice(
                        adjudication["text"],
                        answers,
                    )
                    selected_choice = choice
                    decision_mode = "trace_adjudication"
                    adjudication_metrics = adjudication["metrics"]
                    for idx, event in enumerate(adjudication["events"]):
                        event_copy = dict(event)
                        event_copy.update({
                            "task_id": task_id,
                            "adjudication_event_index": len(adjudication_events),
                            "candidate_answers": {
                                "A": answers[0],
                                "B": answers[1],
                                "C": answers[2],
                            },
                        })
                        adjudication_events.append(event_copy)
                except Exception as exc:
                    if args.on_adjudication_error == "fail":
                        raise
                    selected_answer = answers[0]
                    selected_choice = "A"
                    decision_mode = "adjudication_failed_fallback_run1"
                    reason = f"adjudication failed: {exc}"

        total_metrics = add_metric_totals([base_total, adjudication_metrics])
        summed_wall = number(base_total.get("total_wall_time_sec")) + number(
            adjudication_metrics.get("total_wall_time_sec")
        )
        critical_path_wall = (
            max(base_wall_times or [0.0])
            + number(adjudication_metrics.get("total_wall_time_sec"))
        )
        total_metrics["summed_wall_time_sec"] = round(summed_wall, 3)
        total_metrics["critical_path_wall_time_sec"] = round(critical_path_wall, 3)
        total_metrics["base_total_tokens"] = int_number(base_total.get("total_tokens"))
        total_metrics["adjudication_total_tokens"] = int_number(
            adjudication_metrics.get("total_tokens")
        )
        total_metrics["decision_mode"] = decision_mode
        total_metrics["selected_choice"] = selected_choice

        subresponses = []
        for idx, item in enumerate(items):
            subresponses.append({
                "run": idx + 1,
                "choice": "ABC"[idx],
                "answer": answers[idx],
                "normalized_answer": norms[idx],
                "cost_metrics": get_cost_metrics(item),
            })
        if decision_mode == "trace_adjudication":
            subresponses.append({
                "run": "adjudicator",
                "model": args.model,
                "answer": selected_answer,
                "cost_metrics": adjudication_metrics,
                "reason": reason,
            })

        response_rows.append({
            "task_id": task_id,
            "model_output": {"id": "main-task", "answer": selected_answer},
            "code": "",
            "token_usage_sut": int_number(total_metrics.get("total_tokens")),
            "token_usage_sut_input": int_number(total_metrics.get("input_tokens")),
            "token_usage_sut_output": int_number(total_metrics.get("output_tokens")),
            "cost_metrics": total_metrics,
            "subresponses": subresponses,
            "runtime": total_metrics["summed_wall_time_sec"],
        })

        task_rows.append({
            "domain": domain,
            "task_id": task_id,
            "decision_mode": decision_mode,
            "selected_answer": selected_answer,
            "selected_choice": selected_choice,
            "run1_answer": answers[0],
            "run2_answer": answers[1],
            "run3_answer": answers[2],
            "run1_norm": norms[0],
            "run2_norm": norms[1],
            "run3_norm": norms[2],
            "adjudication_reason": reason,
            "base_total_tokens": total_metrics["base_total_tokens"],
            "adjudication_total_tokens": total_metrics["adjudication_total_tokens"],
            "total_tokens": total_metrics.get("total_tokens", 0),
            "input_tokens": total_metrics.get("input_tokens", 0),
            "output_tokens": total_metrics.get("output_tokens", 0),
            "reasoning_tokens": total_metrics.get("reasoning_tokens", 0),
            "cached_input_tokens": total_metrics.get("cached_input_tokens", 0),
            "summed_wall_time_sec": total_metrics["summed_wall_time_sec"],
            "critical_path_wall_time_sec": total_metrics["critical_path_wall_time_sec"],
            "llm_call_count": total_metrics.get("llm_call_count", 0),
            "turn_count": total_metrics.get("turn_count", 0),
            "code_execution_count": total_metrics.get("code_execution_count", 0),
        })

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    response_cache_out = (
        Path(args.response_cache_out).expanduser().resolve()
        if args.response_cache_out
        else out_dir / f"{domain}_claude_majority_response_cache_{timestamp}.json"
    )
    response_cache_out.parent.mkdir(parents=True, exist_ok=True)
    with response_cache_out.open("w", encoding="utf-8") as f:
        json.dump(jsonable(response_rows), f, indent=2, ensure_ascii=False)

    tasks_csv = out_dir / f"{domain}_claude_majority_tasks_{timestamp}.csv"
    tasks_jsonl = out_dir / f"{domain}_claude_majority_tasks_{timestamp}.jsonl"
    events_jsonl = out_dir / f"{domain}_claude_majority_adjudication_events_{timestamp}.jsonl"
    prompts_jsonl = out_dir / f"{domain}_claude_majority_adjudication_prompts_{timestamp}.jsonl"
    write_csv(task_rows, tasks_csv)
    write_jsonl(task_rows, tasks_jsonl)
    write_jsonl(adjudication_events, events_jsonl)
    write_jsonl(adjudication_prompts, prompts_jsonl)

    if args.sut_name:
        results_dir = Path(args.kramabench_results_dir).expanduser()
        if not results_dir.is_absolute():
            results_dir = (Path.cwd() / results_dir).resolve()
        cache_dir = results_dir / args.sut_name / "response_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        sut_cache = cache_dir / f"{domain}_{timestamp}.json"
        with sut_cache.open("w", encoding="utf-8") as f:
            json.dump(jsonable(response_rows), f, indent=2, ensure_ascii=False)
        print(f"Wrote evaluate-ready cache: {sut_cache}")

    print(f"Wrote response cache: {response_cache_out}")
    print(f"Wrote task CSV: {tasks_csv}")
    print(f"Wrote task JSONL: {tasks_jsonl}")
    print(f"Wrote adjudication events: {events_jsonl}")
    print(f"Wrote adjudication prompts: {prompts_jsonl}")
    print(f"Tasks: {len(task_rows)}")
    print(f"Trace adjudications: {sum(1 for row in task_rows if row['decision_mode'] == 'trace_adjudication')}")


if __name__ == "__main__":
    main()
