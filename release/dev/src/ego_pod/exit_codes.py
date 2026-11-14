"""退出码常量与 ``EgoError`` 异常层级集中定义。

本模块是 Ego Pod 全局的退出码与异常映射单一事实来源（single source of
truth）。CLI 顶层捕获 :class:`EgoError`，打印其消息到标准错误并以其
``exit_code`` 退出；各模块（Config / Collector / Compiler / Trainer /
Vault / Runtime）抛出对应的子类异常以携带正确的退出码语义。

退出码遵循设计文档 §Error Handling：

==================  ====  ============================================
常量                值    含义
==================  ====  ============================================
``EXIT_OK``          0    全部阶段成功
``EXIT_RUNTIME_FAILURE``  1    I/O / 训练 / 数据为空 / Edition 不匹配 / Vault 失败
``EXIT_USAGE_OR_CONFIG``  2    命令行参数错 / 配置文件错 / 未知子命令
==================  ====  ============================================
"""

# ---------------------------------------------------------------------------
# 退出码常量
# ---------------------------------------------------------------------------

#: 全部阶段成功结束。
EXIT_OK = 0

#: 运行期失败：I/O 错误、训练失败、可用数据为空、Edition 不匹配、Vault 写入失败等。
#: 对应需求 R2-AC6/AC9、R4、R6、R7、R8、R10-AC7、R11-AC2/AC3、R12 等运行期错误分支。
EXIT_RUNTIME_FAILURE = 1

#: 用法或配置错误：命令行参数缺失/非法、未知子命令、配置文件加载或校验失败。
#: 对应需求 R1-AC4/AC7、R3 全部加载与校验条款、R11-AC6 等。
EXIT_USAGE_OR_CONFIG = 2


# ---------------------------------------------------------------------------
# 异常层级
# ---------------------------------------------------------------------------


class EgoError(Exception):
    """Ego Pod 所有受控错误的基类。

    CLI 顶层以 ``except EgoError as e: print(e, file=sys.stderr); return
    e.exit_code`` 的形式统一处理本异常族，从而避免向用户透出 traceback。

    默认 ``exit_code`` 为 :data:`EXIT_RUNTIME_FAILURE`（1）；需要表达
    "用法/配置错误"语义的子类应将其覆盖为 :data:`EXIT_USAGE_OR_CONFIG`（2）。
    """

    #: 该异常映射到的进程退出码。子类可覆盖以表达不同语义。
    exit_code: int = EXIT_RUNTIME_FAILURE


class ConfigError(EgoError):
    """配置加载或校验失败。

    覆盖 :data:`EXIT_USAGE_OR_CONFIG`（2），对应需求 R3 全部加载/校验条款
    （文件缺失、无读权限、I/O 错误、YAML 解析错误、顶层字段缺失、
    normal-only / qwen3.5-0.8B-only / local-only / qlora-only / 五参数完备性
    校验失败）以及 R11-AC6 的 ``ego train`` 配置加载失败。
    """

    exit_code = EXIT_USAGE_OR_CONFIG


class CollectorError(EgoError):
    """数据采集阶段失败。

    沿用基类的 :data:`EXIT_RUNTIME_FAILURE`（1），对应需求 R4-AC4/AC7/AC8
    （根/子路径权限错误、五个 Data_Category 全部无可用常规文件）。
    """


class CompilerError(EgoError):
    """语料构建阶段失败。

    沿用基类的 :data:`EXIT_RUNTIME_FAILURE`（1），对应需求 R7-AC6/AC7
    （过滤后语料为空、无法补足最小样本数 16）等运行期错误。
    """


class TrainerError(EgoError):
    """训练阶段失败。

    沿用基类的 :data:`EXIT_RUNTIME_FAILURE`（1），对应需求 R8-AC7/AC9
    （基座加载失败、LoRA 参数缺失或非法）。
    """


class VaultError(EgoError):
    """产出持久化阶段失败。

    沿用基类的 :data:`EXIT_RUNTIME_FAILURE`（1），对应需求 R10-AC7
    （写入产物时遭遇磁盘 I/O 错误，需清理半成品后以退出码 1 结束）。
    """


class RuntimeError_(EgoError):
    """``ego run`` 加载与对话阶段失败。

    类名带尾部下划线以避免与 Python 内置 :class:`RuntimeError` 冲突。
    沿用基类的 :data:`EXIT_RUNTIME_FAILURE`（1），对应需求 R12-AC3/AC4/AC5
    （未找到 Ego_Package、Edition 非 normal、config/adapter 文件缺失或损坏）。
    """


__all__ = [
    "EXIT_OK",
    "EXIT_RUNTIME_FAILURE",
    "EXIT_USAGE_OR_CONFIG",
    "EgoError",
    "ConfigError",
    "CollectorError",
    "CompilerError",
    "TrainerError",
    "VaultError",
    "RuntimeError_",
]
