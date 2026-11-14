"""集成测试共享夹具与工具（tests/integration）。

本 conftest 为 ``ego train`` / ``ego run`` 的端到端集成测试提供：

* **autouse ``force_stub_backend``**：在每个集成测试中强制 Trainer 选择
  Stub_Backend。在装有 torch/peft 等真实训练框架的机器上，``detect_backend``
  会优先选中某个真实后端，而这些后端在 MVP 中的 ``load_base_model`` 抛
  :class:`NotImplementedError`（被包装为 ``TrainerError``），导致 ``ego train``
  无法端到端跑通。本夹具把四个真实后端的 ``is_available`` 一律改写为返回
  ``False``，使 :func:`ego_pod.trainer.detect.detect_backend`（无论从何处导入
  调用）都回退到始终可用的 Stub_Backend，从而保证 ``ego train`` 在任何环境下
  都能确定性地成功。
* **``write_config`` 工厂夹具**：把一份 Ego_Config 以 YAML 写入磁盘，默认产出
  一份通过 MVP 校验的合法配置，并允许逐项覆盖以构造各类非法配置（顶层缺失、
  非 normal Edition、非法 base、非 local 数据源、非 qlora 方法等）。
* **``make_category_dirs`` 工厂夹具**：在给定基目录下创建五个 Data_Category 的
  本地数据源目录并填充样本文件，返回 ``category -> 目录路径`` 映射供 ``write_config``
  引用。
* **``run_train`` 夹具**：``chdir`` 到 ``tmp_path`` 后通过 :func:`ego_pod.cli.main`
  运行 ``ego train`` 子命令，返回 ``(退出码, stdout, stderr)``——既覆盖 CLI 顶层
  的异常→退出码映射，又便于断言阶段日志与各阶段摘要。

这些工厂夹具被三个集成测试文件（``test_train_stub`` / ``test_train_failure`` /
``test_run_after_train``）共享。测试均以 ``tmp_path`` 为工作目录，不依赖任何外部
资源或网络。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ego_pod.collector.pipeline import CATEGORY_ORDER

# ---------------------------------------------------------------------------
# 默认合法配置片段
# ---------------------------------------------------------------------------

#: 一份通过 R3/R8 校验的默认 QLoRA 训练参数（与 config.defaults 默认值一致）。
_DEFAULT_TRAINING: dict[str, object] = {
    "method": "qlora",
    "quantization": "4bit",
    "lora_rank": 16,
    "lora_alpha": 32,
    "learning_rate": 0.0002,
    "epochs": 3,
    "batch_size": 4,
}

#: 默认 editions 段：仅声明 normal，基座 qwen3.5-0.8B（满足 R3-AC6/AC7）。
_DEFAULT_EDITIONS: dict[str, object] = {"normal": {"base": "qwen3.5-0.8B"}}


# ---------------------------------------------------------------------------
# autouse：强制 Stub_Backend
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def force_stub_backend(monkeypatch):
    """强制 Trainer 探测回退到 Stub_Backend（见模块文档）。

    把四个真实后端的 ``is_available`` 改写为返回 ``False``。
    :func:`ego_pod.trainer.detect.detect_backend` 据此跳过全部真实后端并返回
    始终可用的 Stub_Backend，无论该函数被哪个模块导入调用（``trainer.pipeline``
    或 ``runtime.inference``）均生效。
    """

    from ego_pod.trainer.mlx_backend import MlxBackend
    from ego_pod.trainer.peft_backend import PeftCpuBackend, PeftCudaBackend
    from ego_pod.trainer.unsloth_backend import UnslothBackend

    for backend_cls in (UnslothBackend, PeftCudaBackend, MlxBackend, PeftCpuBackend):
        monkeypatch.setattr(
            backend_cls,
            "is_available",
            classmethod(lambda cls: False),
        )
    yield


# ---------------------------------------------------------------------------
# 工厂夹具：写配置文件
# ---------------------------------------------------------------------------


@pytest.fixture
def write_config():
    """返回一个把 Ego_Config 写入磁盘的工厂函数。

    工厂签名::

        _write(
            path,
            sources,              # dict[category -> 目录路径]
            *,
            source_types=None,    # dict[category -> type]，覆盖默认 "local"
            training=None,        # 覆盖整个 training 段
            editions=None,        # 覆盖整个 editions 段
            output=None,          # 覆盖整个 output 段
            drop_top=None,        # 可迭代的顶层键名，从最终配置中移除（构造 R3-AC5）
        ) -> Path

    默认产出一份通过 MVP 校验的合法配置；通过覆盖参数可构造各类非法配置。
    """

    def _write(
        path: str | Path,
        sources: dict[str, str | Path],
        *,
        source_types: dict[str, str] | None = None,
        training: dict | None = None,
        editions: dict | None = None,
        output: dict | None = None,
        drop_top: list[str] | None = None,
    ) -> Path:
        source_types = source_types or {}
        sources_section = {
            category: {
                "type": source_types.get(category, "local"),
                "path": str(directory),
            }
            for category, directory in sources.items()
        }
        config: dict[str, object] = {
            "sources": sources_section,
            "training": training if training is not None else dict(_DEFAULT_TRAINING),
            "editions": editions if editions is not None else dict(_DEFAULT_EDITIONS),
            "output": output if output is not None else {"dir": "./ego-output"},
        }
        for key in drop_top or ():
            config.pop(key, None)

        target = Path(path)
        target.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return target

    return _write


# ---------------------------------------------------------------------------
# 工厂夹具：创建五类数据源目录
# ---------------------------------------------------------------------------


@pytest.fixture
def make_category_dirs():
    """返回一个创建五个 Data_Category 本地目录并填充样本文件的工厂函数。

    工厂签名::

        _make(base, contents, *, create_empty=()) -> dict[category -> Path]

    其中 ``contents`` 为 ``{category: [(文件名, 文本内容), ...]}``：

    * ``category`` 出现在 ``contents`` 中时，创建其目录并写入给定文件
      （空字符串内容即写出 0 字节文件，用于构造空语料场景）；
    * ``category`` 出现在 ``create_empty`` 中时，仅创建空目录（不写文件）；
    * 其余 ``category`` 不创建目录（路径不存在）。

    返回的映射覆盖全部五个 ``category``（无论目录是否实际创建），其路径可直接
    交给 :func:`write_config`。
    """

    def _make(
        base: Path,
        contents: dict[str, list[tuple[str, str]]] | None = None,
        *,
        create_empty: tuple[str, ...] = (),
    ) -> dict[str, Path]:
        contents = contents or {}
        data_root = Path(base) / "ego-data"
        paths: dict[str, Path] = {}
        for category in CATEGORY_ORDER:
            directory = data_root / category
            if category in contents:
                directory.mkdir(parents=True, exist_ok=True)
                for filename, text in contents[category]:
                    (directory / filename).write_text(text, encoding="utf-8")
            elif category in create_empty:
                directory.mkdir(parents=True, exist_ok=True)
            paths[category] = directory
        return paths

    return _make


# ---------------------------------------------------------------------------
# 夹具：运行 ego train 子命令
# ---------------------------------------------------------------------------


@pytest.fixture
def run_train(monkeypatch, tmp_path, capsys):
    """返回一个运行 ``ego train`` 并返回 ``(rc, stdout, stderr)`` 的函数。

    调用时先 ``chdir`` 到 ``tmp_path``（使 Collector/Compiler/Trainer/Vault 的
    工作目录、``.ego/`` 中间产物目录与 ``ego-output/`` 产出目录均落于 ``tmp_path``），
    再通过 :func:`ego_pod.cli.main` 执行 ``train`` 子命令——经由 CLI 顶层的
    异常→退出码映射得到真实退出码。``extra_args`` 追加到 ``["train", ...]`` 之后
    （如 ``["--config", "ego-config.yaml"]``）。
    """

    from ego_pod.cli import main

    def _run(extra_args: list[str] | None = None) -> tuple[int, str, str]:
        monkeypatch.chdir(tmp_path)
        rc = main(["train", *(extra_args or [])])
        captured = capsys.readouterr()
        return rc, captured.out, captured.err

    return _run
