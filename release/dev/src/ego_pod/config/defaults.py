"""``ego init`` 写入的默认 Ego_Config 内容。

本模块负责渲染 ``ego init`` 生成的默认 ``ego-config.yaml`` 文本，并提供一个
将其写入磁盘的薄封装 :func:`write_default`。渲染产物满足 Requirements 2.1–2.5：

* 顶层包含 ``sources`` / ``training`` / ``editions`` / ``output`` 四个配置项
  （R2-AC2）。
* ``sources`` 下为 Sense / Action / Thought / Skill / Credentials 五个
  Data_Category 各填充一个条目，每个条目同时包含 ``type: local`` 字段与一个
  非空的本地目录占位路径（R2-AC3）。
* ``editions`` 仅声明一个名为 ``normal`` 的 Edition，其 ``base`` 为
  ``qwen3.5-0.8B``（R2-AC4）。
* ``training.method`` 为 ``qlora``，并填充五个数值参数 ``lora_rank=16``、
  ``lora_alpha=32``、``learning_rate=2e-4``、``epochs=3``、``batch_size=4``
  （R2-AC5）。

渲染输出的字段命名与 :mod:`ego_pod.config.schema` 中的 dataclass 一一对应，
因此 ``loader`` 解析回来后可直接构造 :class:`~ego_pod.config.schema.EgoConfig`。

注意：文件是否已存在、``--force`` 覆盖语义、I/O 错误下不留部分写入文件等
逻辑由 ``commands/init.py``（Requirements 2.6–2.9）负责，本模块仅关注
"渲染默认内容" 与 "以 UTF-8 写入给定路径" 两件事。
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "DEFAULT_SOURCE_PATHS",
    "DEFAULT_TRAINING",
    "DEFAULT_EDITION_NAME",
    "DEFAULT_BASE_MODEL",
    "DEFAULT_OUTPUT_DIR",
    "render_default_yaml",
    "write_default",
]


# --- 默认值常量 ----------------------------------------------------------------
# 集中定义便于单元测试断言默认值，并保证渲染与常量始终一致。

# 五个 Data_Category 的本地数据源占位目录（固定顺序）。
DEFAULT_SOURCE_PATHS: dict[str, str] = {
    "sense": "./ego-data/sense",
    "action": "./ego-data/action",
    "thought": "./ego-data/thought",
    "skill": "./ego-data/skill",
    "credentials": "./ego-data/credentials",
}

# QLoRA 训练参数默认值（与设计文档技术栈章一致）。
DEFAULT_TRAINING: dict[str, object] = {
    "method": "qlora",
    "quantization": "4bit",
    "lora_rank": 16,
    "lora_alpha": 32,
    "learning_rate": 2e-4,
    "epochs": 3,
    "batch_size": 4,
}

# MVP 仅声明 normal edition，基座固定为 qwen3.5-0.8B。
DEFAULT_EDITION_NAME = "normal"
DEFAULT_BASE_MODEL = "qwen3.5-0.8B"

# 产出目录配置（MVP 占位；Vault 实际在 CWD 下使用 ego-output/）。
DEFAULT_OUTPUT_DIR = "./ego-output"


def _format_float(value: float) -> str:
    """将浮点数渲染为定点小数字符串，避免被 YAML 解析为字符串。

    PyYAML 的隐式 float 解析器不接受形如 ``2e-4`` 的无小数点科学计数法，
    因此对含指数的表示展开为定点形式（如 ``0.0002``），保证 ``loader`` 解析
    回来时得到的是 ``float`` 而非 ``str``。
    """

    text = repr(value)
    if "e" in text or "E" in text:
        text = f"{value:.12f}".rstrip("0").rstrip(".")
    return text


def render_default_yaml() -> str:
    """渲染默认 ``ego-config.yaml`` 的完整文本（含注释）。

    :returns: 以 ``\\n`` 分隔、UTF-8 友好的 YAML 文本，末尾带一个换行符。
        其结构与 :class:`~ego_pod.config.schema.EgoConfig` 字段一一对应，可被
        ``loader`` 解析并通过 ``validators`` 的 MVP 范围校验。
    """

    src = DEFAULT_SOURCE_PATHS
    tr = DEFAULT_TRAINING

    lines = [
        "# Ego Pod 配置文件 (ego-config.yaml)",
        "# 由 `ego init` 自动生成。所有相对路径均相对于本文件所在的工作目录。",
        "# MVP 仅支持：local 数据源、qlora 训练方法、qwen3.5-0.8B 基座、normal edition。",
        "",
        "# 五个 Data_Category 的本地数据源目录。",
        "# 请将下列占位路径替换为你自己存放对应数据的本地目录。",
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
        "# Ego 模型版本规格。MVP 仅声明 normal edition。",
        "editions:",
        f"  {DEFAULT_EDITION_NAME}:",
        f"    base: {DEFAULT_BASE_MODEL}",
        '    description: "默认 normal edition，基于 Qwen 3.5 0.8B + QLoRA。"',
        "",
        "# 可选：功能分区分类法（Zone_Config）。省略整块则使用内置默认基线。",
        "# Optional: Functional_Zone taxonomy (Zone_Config). Omit the whole",
        "# block to use the built-in default baseline.",
        "# 默认基线（即本块保持注释时的行为）：",
        "#   coding    <- skill, action      knowledge <- thought",
        "#   未列出的类别（sense / credentials）以及任何未匹配分区的样本归入 Kernel。",
        "#   Categories not listed (sense / credentials) and any unmatched",
        "#   samples are routed to the Kernel.",
        "# 取消下面的注释即可自定义分区与路由（Zone_Name 由小写字母、数字、连字符组成且唯一）：",
        "# zones:",
        "#   # 声明的功能分区集合。",
        "#   zones: [coding, knowledge]",
        "#   # Data_Category -> 分区路由基线；未列出的类别路由到 Kernel。",
        "#   routing:",
        "#     skill: coding",
        "#     action: coding",
        "#     thought: knowledge",
        "#     # sense / credentials 未列出 -> 归入 Kernel",
        "",
        "# 产出目录配置（MVP 占位）。",
        "output:",
        f"  dir: {DEFAULT_OUTPUT_DIR}",
        "",
    ]
    return "\n".join(lines)


def write_default(path: str | Path) -> Path:
    """将默认配置文本以 UTF-8（无 BOM）写入 ``path``。

    本函数仅负责渲染并写入；文件是否已存在、``--force`` 覆盖、写入失败时的
    清理等策略由调用方（``commands/init.py``）处理。

    :param path: 目标 ``ego-config.yaml`` 路径。
    :returns: 写入文件的 :class:`~pathlib.Path`（已解析为绝对路径）。
    """

    target = Path(path)
    target.write_text(render_default_yaml(), encoding="utf-8")
    return target.resolve()
