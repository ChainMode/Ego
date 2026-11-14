"""Vault 版本管理模块。

负责把 Trainer 的临时产出按 Ego_Package 结构事务性地落盘到工作目录下的
``ego-output/``。本模块的核心写入能力由 ``package`` 提供
（``write_package`` 事务性写入 ``manifest.json`` 与 ``normal/`` 子目录，
``build_manifest`` / ``build_config`` 构造产物字典）；编排由 ``pipeline``
提供（``run_vault``：创建/识别 ``ego-output/`` 并打印 directory-created 事件、
调用写入、校验三文件存在、打印 Ego_Package 绝对路径作为 package-created 事件，
I/O 错误经 ``VaultError`` 上抛由 CLI 顶层映射为退出码 1）。
"""

from ego_pod.vault.package import (
    PackageArtifacts,
    build_config,
    build_manifest,
    write_package,
)
from ego_pod.vault.pipeline import OUTPUT_DIRNAME, run_vault

__all__ = [
    "PackageArtifacts",
    "build_config",
    "build_manifest",
    "write_package",
    "OUTPUT_DIRNAME",
    "run_vault",
]
