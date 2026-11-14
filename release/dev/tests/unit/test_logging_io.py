"""`ego_pod.logging_io` 阶段日志与摘要打印工具的单元测试。"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest

from ego_pod import logging_io
from ego_pod.logging_io import Event, Stage, print_summary, stage_event

# ISO 8601 含时区偏移、秒精度（无小数秒）的整行格式：
# [2025-11-25T08:00:00+08:00] COLLECTOR START
LINE_RE = re.compile(
    r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}\] "
    r"(COLLECTOR|COMPILER|TRAINER|VAULT) (START|END)$"
)


@pytest.fixture
def fixed_clock(monkeypatch):
    """将 now_timestamp 固定为带 +08:00 偏移、秒精度的时间戳。"""

    fixed = "2025-11-25T08:00:00+08:00"
    monkeypatch.setattr(logging_io, "now_timestamp", lambda: fixed)
    return fixed


def test_now_timestamp_is_iso8601_with_offset_second_precision():
    ts = logging_io.now_timestamp()
    parsed = datetime.fromisoformat(ts)
    # 含时区偏移
    assert parsed.tzinfo is not None
    # 秒精度：无微秒部分
    assert parsed.microsecond == 0
    # 偏移量格式 +HH:MM / -HH:MM
    assert re.search(r"[+-]\d{2}:\d{2}$", ts)


def test_stage_event_format(fixed_clock, capsys):
    line = stage_event(Stage.COLLECTOR, Event.START)
    captured = capsys.readouterr()
    assert line == f"[{fixed_clock}] COLLECTOR START"
    assert captured.out == line + "\n"
    assert LINE_RE.match(line)


def test_stage_event_accepts_lowercase_strings(fixed_clock):
    line = stage_event("vault", "end")
    assert line == f"[{fixed_clock}] VAULT END"


@pytest.mark.parametrize("stage", ["COLLECTOR", "COMPILER", "TRAINER", "VAULT"])
@pytest.mark.parametrize("event", ["START", "END"])
def test_stage_event_all_combinations_match_pattern(fixed_clock, stage, event):
    line = stage_event(stage, event)
    assert LINE_RE.match(line)


def test_stage_event_rejects_unknown_stage(fixed_clock):
    with pytest.raises(ValueError):
        stage_event("UNKNOWN", Event.START)


def test_stage_event_rejects_unknown_event(fixed_clock):
    with pytest.raises(ValueError):
        stage_event(Stage.TRAINER, "PAUSE")


def test_print_summary_single_line(capsys):
    out = print_summary("dropped_short=2 merged_dup=1 total_before=10 total_after=7")
    captured = capsys.readouterr()
    assert out == "dropped_short=2 merged_dup=1 total_before=10 total_after=7"
    assert captured.out == out + "\n"


def test_print_summary_collapses_newlines():
    out = print_summary("line1\nline2\r\nline3\rline4")
    assert "\n" not in out
    assert "\r" not in out
    assert out == "line1 line2 line3 line4"


def test_stage_event_real_clock_matches_pattern():
    # 不打 monkeypatch，验证真实时钟产出也满足整行格式
    line = stage_event(Stage.COMPILER, Event.END)
    assert LINE_RE.match(line)
