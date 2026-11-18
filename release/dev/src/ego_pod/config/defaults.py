"""``ego init`` 写入的默认与多规格 Ego_Config 模板。

本模块负责渲染 ``ego init`` 生成的 ``ego-config.yaml`` 文本：
- :func:`render_default_yaml`：非交互式默认模板（保持 ``normal`` + ``qwen3.5-0.8B``，
  并附带四档规格及远程数据源注释示例，100% 向后兼容既有 ``ego init`` 行为）。
- :func:`render_all_editions_yaml`：``ego init --all-editions`` 生成的四档标准规格
  （``lite`` / ``normal`` / ``plus`` / ``super``）模板。
- :func:`render_custom_yaml`：``ego init --interactive`` 使用的参数化渲染器。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ego_pod.config.schema import EDITION_PRESETS

__all__ = [
    "DEFAULT_SOURCE_PATHS",
    "DEFAULT_TRAINING",
    "DEFAULT_EDITION_NAME",
    "DEFAULT_BASE_MODEL",
    "DEFAULT_OUTPUT_DIR",
    "render_default_yaml",
    "render_all_editions_yaml",
    "render_custom_yaml",
    "write_default",
]

# 五个 Data_Category 的本地数据源占位目录（固定顺序）。
DEFAULT_SOURCE_PATHS: dict[str, str] = {
    "sense": "./ego-data/sense",
    "action": "./ego-data/action",
    "thought": "./ego-data/thought",
    "skill": "./ego-data/skill",
    "credentials": "./ego-data/credentials",
}

# QLoRA 训练参数默认值。
DEFAULT_TRAINING: dict[str, object] = {
    "method": "qlora",
    "quantization": "4bit",
    "lora_rank": 16,
    "lora_alpha": 32,
    "learning_rate": 2e-4,
    "epochs": 3,
    "batch_size": 4,
}

# 默认声明 normal edition，基座默认 qwen3.5-0.8B 以保证开箱零配置轻量可跑通。
DEFAULT_EDITION_NAME = "normal"
DEFAULT_BASE_MODEL = "qwen3.5-0.8B"

# 产出目录配置。
DEFAULT_OUTPUT_DIR = "./ego-output"


def _format_float(value: float) -> str:
    """将浮点数渲染为定点小数字符串，避免被 YAML 解析为字符串。"""

    text = repr(value)
    if "e" in text or "E" in text:
        text = f"{value:.12f}".rstrip("0").rstrip(".")
    return text


def render_default_yaml() -> str:
    """渲染默认 ``ego-config.yaml`` 的完整文本（含注释）。"""

    src = DEFAULT_SOURCE_PATHS
    tr = DEFAULT_TRAINING

    lines = [
        "# Ego Pod 配置文件 (ego-config.yaml)",
        "# 由 `ego init` 自动生成。所有相对路径均相对于本文件所在的工作目录。",
        "# 支持 local / url / s3 / oss 数据源与 lite / normal / plus / super / 自定义 Edition。",
        "",
        "# 五个 Data_Category 的数据源配置。",
        "# 请将下列占位路径替换为你自己存放对应数据的目录或受控远程数据源。",
        "sources:",
        "  sense:",
        "    type: local",
        f"    path: {src['sense']}",
        "  action:",
        "    type: local",
        f"    path: {src['action']}",
        "  thought:",
        "    type: local",
        f"    path: {src['thought']}",
        "  skill:",
        "    type: local",
        f"    path: {src['skill']}",
        "  credentials:",
        "    type: local",
        f"    path: {src['credentials']}",
        "",
        "# QLoRA 训练参数。",
        "training:",
        f"  method: {tr['method']}",
        f"  quantization: {tr['quantization']}",
        f"  lora_rank: {tr['lora_rank']}",
        f"  lora_alpha: {tr['lora_alpha']}",
        f"  learning_rate: {_format_float(float(tr['learning_rate']))}   # 等价于 2e-4",
        f"  epochs: {tr['epochs']}",
        f"  batch_size: {tr['batch_size']}",
        "",
        "# Ego 模型版本规格（可使用 `ego init --all-editions` 一键启用 lite/normal/plus/super）。",
        "editions:",
        f"  {DEFAULT_EDITION_NAME}:",
        f"    base: {DEFAULT_BASE_MODEL}",
        '    description: "默认 normal edition，基于 Qwen 3.5 0.8B + QLoRA。"',
        "#  lite:",
        "#    base: qwen3.5-0.8B",
        '#    description: "轻量端侧规格（Kernel + L1，<= 64MB）"',
        "#  plus:",
        "#    base: qwen3.5-4B",
        '#    description: "深度推理规格（Kernel + L1..L3，<= 1024MB）"',
        "#  super:",
        "#    base: qwen3.5-27B",
        '#    description: "旗舰全网格规格（Kernel + L1..L3，<= 4096MB）"',
        "",
        "# 可选：功能分区分类法（Zone_Config）。省略整块则使用内置默认基线。",
        "# zones:",
        "#   zones: [coding, knowledge]",
        "#   routing:",
        "#     skill: coding",
        "#     action: coding",
        "#     thought: knowledge",
        "",
        "# 产出目录与版本历史保留策略。",
        "output:",
        f"  dir: {DEFAULT_OUTPUT_DIR}",
        "  max_history: 5",
        "",
    ]
    return "\n".join(lines)


def render_custom_yaml(
    *,
    source_paths: Mapping[str, str] | None = None,
    editions: Mapping[str, Mapping[str, Any]] | None = None,
    training: Mapping[str, Any] | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_history: int = 5,
) -> str:
    """按给定数据源路径、训练参数与 Edition 集合渲染 ``ego-config.yaml``。"""

    src = dict(DEFAULT_SOURCE_PATHS)
    if source_paths is not None:
        src.update(source_paths)

    tr = dict(DEFAULT_TRAINING)
    if training is not None:
        tr.update(training)

    ed_map: dict[str, dict[str, Any]]
    if editions is not None:
        ed_map = {k: dict(v) for k, v in editions.items()}
    else:
        ed_map = {
            DEFAULT_EDITION_NAME: {
                "base": DEFAULT_BASE_MODEL,
                "description": "默认 normal edition，基于 Qwen 3.5 0.8B + QLoRA。",
            }
        }

    lines = [
        "# Ego Pod 配置文件 (ego-config.yaml)",
        "sources:",
    ]
    for cat in ("sense", "action", "thought", "skill", "credentials"):
        lines.extend(
            [
                f"  {cat}:",
                "    type: local",
                f"    path: {src[cat]}",
            ]
        )

    lines.extend(
        [
            "",
            "training:",
            f"  method: {tr['method']}",
            f"  quantization: {tr['quantization']}",
            f"  lora_rank: {tr['lora_rank']}",
            f"  lora_alpha: {tr['lora_alpha']}",
            f"  learning_rate: {_format_float(float(tr['learning_rate']))}",
            f"  epochs: {tr['epochs']}",
            f"  batch_size: {tr['batch_size']}",
            "",
            "editions:",
        ]
    )

    for ed_name, ed_cfg in ed_map.items():
        lines.append(f"  {ed_name}:")
        lines.append(f"    base: {ed_cfg['base']}")
        desc = str(ed_cfg.get("description", "")).replace('"', '\\"')
        if desc:
            lines.append(f'    description: "{desc}"')
        for opt_key in ("lora_rank", "lora_alpha", "max_level", "max_size_mb"):
            if opt_key in ed_cfg and ed_cfg[opt_key] is not None:
                lines.append(f"    {opt_key}: {ed_cfg[opt_key]}")
        if "zones" in ed_cfg and ed_cfg["zones"] is not None:
            z_list = ", ".join(str(z) for z in ed_cfg["zones"])
            lines.append(f"    zones: [{z_list}]")

    lines.extend(
        [
            "",
            "output:",
            f"  dir: {output_dir}",
            f"  max_history: {max_history}",
            "",
        ]
    )
    return "\n".join(lines)


def render_all_editions_yaml() -> str:
    """渲染包含全部四档标准规格（lite / normal / plus / super）的 ``ego-config.yaml``。"""

    editions: dict[str, dict[str, Any]] = {
        "lite": {
            "base": EDITION_PRESETS["lite"].default_base,
            "description": "轻量端侧规格（Kernel + L1，<= 64MB）",
        },
        "normal": {
            "base": EDITION_PRESETS["normal"].default_base,
            "description": "均衡日常规格（Kernel + L1..L2，<= 256MB）",
        },
        "plus": {
            "base": EDITION_PRESETS["plus"].default_base,
            "description": "深度推理规格（Kernel + L1..L3，<= 1024MB）",
        },
        "super": {
            "base": EDITION_PRESETS["super"].default_base,
            "description": "旗舰全网格规格（Kernel + L1..L3，<= 4096MB）",
        },
    }
    return render_custom_yaml(editions=editions)


def write_default(path: str | Path, *, all_editions: bool = False) -> Path:
    """将默认（或四档全规格）配置文本以 UTF-8（无 BOM）写入 ``path``。"""

    target = Path(path)
    content = render_all_editions_yaml() if all_editions else render_default_yaml()
    target.write_text(content, encoding="utf-8")
    return target.resolve()
