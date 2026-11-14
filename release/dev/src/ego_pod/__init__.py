"""Ego Pod MVP — 个性化 AI 模型训练管线。

本包实现 Ego 项目蓝图 Phase 1 的最小可用版本，提供 ``ego init`` /
``ego train`` / ``ego run`` 三条命令，从本地目录数据出发，经由
Collector → Compiler → Trainer → Vault 流水线，产出一个基于
Qwen 3.5 0.8B + QLoRA 的 ``normal`` edition Ego 模型。
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
