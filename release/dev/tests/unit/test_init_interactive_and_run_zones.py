"""`ego init --interactive / --all-editions` 与 `ego run --zone / --kernel-only` 单测。"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch
import yaml

from ego_pod.commands.init import run_init
from ego_pod.commands.run import parse_cli_load_selection, run_repl
from ego_pod.config.validators import validate
from ego_pod.exit_codes import EXIT_OK, RuntimeError_
from ego_pod.trainer.cells import TrainedCell
from ego_pod.vault.package import EditionBuildInput, write_multi_edition_package


def _make_safetensor(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"lora_A": torch.ones((2,), dtype=torch.float32)}, str(path))
    return path


def test_init_all_editions_and_interactive_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证 --all-editions 生成四档规格，以及 --interactive 引导提示、默认路径告知与覆盖确认。"""
    monkeypatch.chdir(tmp_path)

    # 1. --all-editions 非交互直接生成四档规格
    out_all = io.StringIO()
    rc = run_init(
        argparse.Namespace(force=False, all_editions=True, interactive=False),
        stdout=out_all,
    )
    assert rc == EXIT_OK
    parsed_all = yaml.safe_load((tmp_path / "ego-config.yaml").read_text(encoding="utf-8"))
    assert set(parsed_all["editions"].keys()) == {"lite", "normal", "plus", "super"}
    validate(parsed_all, tmp_path / "ego-config.yaml", stream=io.StringIO())

    # 2. --interactive 文件已存在且用户拒绝覆盖 (输入 n) -> 不修改已有文件且以 EgoError 终止
    from ego_pod.exit_codes import EgoError

    with pytest.raises(EgoError):
        run_init(
            argparse.Namespace(force=False, all_editions=False, interactive=True),
            stdin=io.StringIO("n\n"),
            stdout=io.StringIO(),
        )

    # 3. --interactive 用户确认覆盖 (y)，全回车接受默认目录，选择 lite,plus 与 gemma4-2B 基座
    # 交互输入顺序：覆盖确认(y) -> 5 个数据源路径回车默认 -> editions(lite,plus) -> normal/首规格基座(gemma4-2B)
    interactive_in = io.StringIO("y\n\n\n\n\n\nlite,plus\ngemma4-2B\n")
    interactive_out = io.StringIO()
    rc_inter = run_init(
        argparse.Namespace(force=False, all_editions=False, interactive=True),
        stdin=interactive_in,
        stdout=interactive_out,
    )
    assert rc_inter == EXIT_OK
    # 接受默认数据源目录时输出明确提示
    assert "./ego-data/sense" in interactive_out.getvalue()
    parsed_inter = yaml.safe_load((tmp_path / "ego-config.yaml").read_text(encoding="utf-8"))
    assert set(parsed_inter["editions"].keys()) == {"lite", "plus"}
    validate(parsed_inter, tmp_path / "ego-config.yaml", stream=io.StringIO())


def test_ego_run_multi_edition_zone_depths_and_kernel_only(tmp_path: Path) -> None:
    """验证 ego run 支持多规格、--zone <name>:<depth> 重复参数、--kernel-only 互斥与语法校验。"""
    # 语法与互斥单元校验
    sel_zones = parse_cli_load_selection(["coding:2", "knowledge:1"], kernel_only=False)
    assert sel_zones is not None
    assert sel_zones.depths == {"coding": 2, "knowledge": 1}

    sel_kernel = parse_cli_load_selection(None, kernel_only=True)
    assert sel_kernel is not None
    assert sel_kernel.depths == {}

    with pytest.raises(RuntimeError_):
        parse_cli_load_selection(["coding:1"], kernel_only=True)

    for bad_spec in ["coding", "coding:abc", ":2", "coding:0", "coding:-1"]:
        with pytest.raises(RuntimeError_):
            parse_cli_load_selection([bad_spec], kernel_only=False)

    # 构造含 plus (L1..L3) 与 lite (L1) 的多规格包
    root = tmp_path / "ego-output"
    write_multi_edition_package(
        root,
        edition_builds=[
            EditionBuildInput(
                edition="lite",
                base_model="qwen3.5-0.8B",
                lora_rank=8,
                lora_alpha=16,
                cells=[
                    TrainedCell(None, 0, _make_safetensor(tmp_path / "t" / "lk.safetensors")),
                    TrainedCell("coding", 1, _make_safetensor(tmp_path / "t" / "lc1.safetensors")),
                ],
            ),
            EditionBuildInput(
                edition="plus",
                base_model="qwen3.5-4B",
                lora_rank=32,
                lora_alpha=64,
                cells=[
                    TrainedCell(None, 0, _make_safetensor(tmp_path / "t" / "pk.safetensors")),
                    TrainedCell("coding", 1, _make_safetensor(tmp_path / "t" / "pc1.safetensors")),
                    TrainedCell("coding", 2, _make_safetensor(tmp_path / "t" / "pc2.safetensors")),
                    TrainedCell("coding", 3, _make_safetensor(tmp_path / "t" / "pc3.safetensors")),
                ],
            ),
        ],
        training_backend="stub",
        include_integrity=True,
    )

    # 加载已训练的 plus 规格并按需激活 coding:2
    out = io.StringIO()
    rc = run_repl(
        argparse.Namespace(edition="plus", zone=["coding:2"], kernel_only=False),
        cwd=tmp_path,
        stdin=io.StringIO("你好\n:quit\n"),
        stdout=out,
        stderr=io.StringIO(),
    )
    assert rc == EXIT_OK

    # 加载已训练的 lite 规格并指定 --kernel-only
    rc_ko = run_repl(
        argparse.Namespace(edition="lite", zone=None, kernel_only=True),
        cwd=tmp_path,
        stdin=io.StringIO(":quit\n"),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc_ko == EXIT_OK

    # 请求包内未构建的 super 规格 -> RuntimeError_ 列出已构建的 lite, plus
    with pytest.raises(RuntimeError_) as exc_ed:
        run_repl(
            argparse.Namespace(edition="super", zone=None, kernel_only=False),
            cwd=tmp_path,
            stdin=io.StringIO(""),
        )
    assert "super" in str(exc_ed.value)
    assert "lite" in str(exc_ed.value)

    # 请求超出网格深度的 coding:5 -> RuntimeError_
    with pytest.raises(RuntimeError_):
        run_repl(
            argparse.Namespace(edition="plus", zone=["coding:5"], kernel_only=False),
            cwd=tmp_path,
            stdin=io.StringIO(""),
        )


@pytest.mark.parametrize(
    "bad_edition",
    ["", "UpperCase", "bad_underscore", "-leading-dash", "a" * 40],
)
def test_ego_run_rejects_malformed_edition_names(tmp_path: Path, bad_edition: str) -> None:
    with pytest.raises(RuntimeError_):
        run_repl(
            argparse.Namespace(edition=bad_edition, zone=None, kernel_only=False),
            cwd=tmp_path,
            stdin=io.StringIO(""),
        )


def test_init_interactive_rejects_unsupported_base_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ego_pod.exit_codes import ConfigError

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError) as exc:
        run_init(
            argparse.Namespace(force=False, all_editions=False, interactive=True),
            stdin=io.StringIO("\n\n\n\n\nnormal\ninvalid-model-999B\n"),
            stdout=io.StringIO(),
        )
    assert "qwen3.5-0.8B" in str(exc.value)
