#!/usr/bin/env python3
from __future__ import annotations

"""
Terminal log filter for noisy Habitat/HM3D warnings.

# 方式1：过滤已有日志文件
python log_filter.py < raw.log > clean.log

# 方式2：包裹命令实时过滤（推荐）
python log_filter.py --run "python query_room_receptacle_objects.py --scene 00808-y9hTuugGdiq --disable-llm"

# 方式3：再加你自己的噪声规则
python log_filter.py --run "python your_script.py" --drop-regex "No Glob path result found"

# 方式4：只用自定义规则（禁用内置）
python log_filter.py --no-default-rules --drop-regex "your regex" < raw.log > clean.log

"""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Pattern, Tuple


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: Pattern[str]


def _compile_default_rules() -> List[Rule]:
    # Habitat metadata warning: missing glob path for stage/scene template files.
    return [
        Rule(
            name="habitat_metadata_no_glob",
            pattern=re.compile(
                r"\[Warning\]:\[Metadata\].*No Glob path result found.*unable to load templates from that path\.",
                flags=re.IGNORECASE,
            ),
        ),
        Rule(
            name="habitat_metadata_missing_template",
            pattern=re.compile(
                r"\[Warning\]:\[Metadata\].*(Stage Template|Scene Instance).*unable to load templates",
                flags=re.IGNORECASE,
            ),
        ),
    ]


def _compile_extra_rules(extra_regex: List[str]) -> List[Rule]:
    out: List[Rule] = []
    for i, raw in enumerate(extra_regex):
        text = (raw or "").strip()
        if not text:
            continue
        out.append(Rule(name=f"user_rule_{i+1}", pattern=re.compile(text)))
    return out


def _match_rule(line: str, rules: Iterable[Rule]) -> Optional[str]:
    for rule in rules:
        if rule.pattern.search(line):
            return rule.name
    return None


def _emit_summary(stats: Dict[str, int], total_in: int, total_out: int) -> None:
    dropped = total_in - total_out
    print(
        f"\n[log_filter] lines_in={total_in} lines_out={total_out} dropped={dropped}",
        file=sys.stderr,
        flush=True,
    )
    if not stats:
        print("[log_filter] no lines were suppressed by rules.", file=sys.stderr, flush=True)
        return
    print("[log_filter] suppressed by rule:", file=sys.stderr, flush=True)
    for key in sorted(stats.keys()):
        print(f"  - {key}: {stats[key]}", file=sys.stderr, flush=True)


def _process_stream(lines: Iterable[str], rules: List[Rule], print_summary: bool = True) -> int:
    suppressed: Dict[str, int] = {}
    total_in = 0
    total_out = 0
    for line in lines:
        total_in += 1
        hit = _match_rule(line, rules)
        if hit is not None:
            suppressed[hit] = suppressed.get(hit, 0) + 1
            continue
        sys.stdout.write(line)
        total_out += 1

    sys.stdout.flush()
    if print_summary:
        _emit_summary(suppressed, total_in=total_in, total_out=total_out)
    return 0


def _run_and_filter(command: str, rules: List[Rule], print_summary: bool = True) -> int:
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    suppressed: Dict[str, int] = {}
    total_in = 0
    total_out = 0
    for line in proc.stdout:
        total_in += 1
        hit = _match_rule(line, rules)
        if hit is not None:
            suppressed[hit] = suppressed.get(hit, 0) + 1
            continue
        sys.stdout.write(line)
        total_out += 1
        sys.stdout.flush()

    proc.wait()
    if print_summary:
        _emit_summary(suppressed, total_in=total_in, total_out=total_out)
        print(f"[log_filter] wrapped command exit_code={proc.returncode}", file=sys.stderr, flush=True)
    return int(proc.returncode or 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter noisy terminal logs (Habitat/HM3D focused).")
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Optional command to run and filter in real-time. If omitted, filter stdin.",
    )
    parser.add_argument(
        "--drop-regex",
        action="append",
        default=[],
        help="Additional regex to suppress (repeatable).",
    )
    parser.add_argument(
        "--no-default-rules",
        action="store_true",
        help="Disable built-in Habitat/HM3D suppression rules.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not print suppression summary to stderr.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rules: List[Rule] = []
    if not args.no_default_rules:
        rules.extend(_compile_default_rules())
    rules.extend(_compile_extra_rules(args.drop_regex or []))

    if args.run:
        return _run_and_filter(
            command=str(args.run),
            rules=rules,
            print_summary=not args.no_summary,
        )

    return _process_stream(
        lines=sys.stdin,
        rules=rules,
        print_summary=not args.no_summary,
    )


if __name__ == "__main__":
    raise SystemExit(main())
