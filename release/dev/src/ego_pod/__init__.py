"""Ego Pod — 个性化 AI 模型修炼场（Dojo）与多维稀疏网格管线。

提供 ``ego init`` / ``ego train`` / ``ego run`` / ``ego vault`` 命令，
经由 Collector → Compiler（深层专精度网格蒸馏）→ Trainer（多规格 QLoRA）→ Vault
（双层完整性签名与版本历史）流水线，产出可复刻人类原型（Human Prime）认知与风格的
Ego_Package 模型资产。
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
