"""``ego run`` 的交互式 REPL 循环（需求 12.6、12.7、12.8）。

本模块实现 :func:`run_repl`：在 Ego_Package 加载成功后进入交互式命令行对话循环，
对应设计文档 §7「Runtime 模块」的 REPL 行为：

* **提示符**：每轮交互前在标准输出打印固定提示符 :data:`PROMPT`（``ego> ``），
  对每次非空输入调用一次响应生成并打印其回复（R12-AC6）。
* **空白/空行**：当用户输入仅由空白字符组成或为空行时，**不**调用推理，直接重新
  打印提示符等待下一次输入（R12-AC7）。
* **退出**：用户输入 :data:`QUIT_COMMAND`（``:quit``）或发送 EOF 信号（Ctrl-D）时，
  释放模型资源并以退出码 :data:`~ego_pod.exit_codes.EXIT_OK`（0）结束（R12-AC8）。
* **异常隔离**：响应生成过程中抛出的任何异常都被捕获，向标准错误打印一条**简要**
  错误信息（绝不向用户透出 traceback），随后继续 REPL 循环（R12-AC6 的健壮性约束，
  见设计文档「异常 → 不让 traceback 透出给用户（捕获后打印简要错误并继续 REPL）」）。

**依赖注入与解耦**：实际的「响应生成」由 Runtime 的 ``inference`` 模块（任务 10.1）
提供，而该模块可能尚未就绪或在不同环境下有不同实现。为避免硬依赖，本模块**不**导入
``inference``，而是通过参数接收一个满足 :class:`ResponseGenerator` 结构协议的对象
（或等价的可调用对象），由 ``commands.run``（任务 10.3）在加载完成后注入。这样
REPL 既可被真实推理引擎驱动，也可在测试中以注入的桩对象独立验证。

**资源释放**：若注入的生成器对象暴露 ``close()`` 方法，则在 ``:quit`` / EOF 退出时
（以及循环因任何原因结束时）调用之以释放模型资源（R12-AC8 的「释放资源」语义）；
释放阶段的异常被吞掉，不影响退出码 0。
"""

from __future__ import annotations

import sys
from typing import Callable, Protocol, TextIO, Union, runtime_checkable

from ego_pod.exit_codes import EXIT_OK

__all__ = [
    "PROMPT",
    "QUIT_COMMAND",
    "INFERENCE_ERROR_PREFIX",
    "ResponseGenerator",
    "run_repl",
]

#: REPL 每轮交互前打印的固定提示符（R12-AC6）。
PROMPT = "ego> "

#: 退出命令；用户输入该字符串（去除首尾空白后）即结束 REPL（R12-AC8）。
QUIT_COMMAND = ":quit"

#: 推理异常时向标准错误打印的简要错误文案前缀（R12-AC6，不透出 traceback）。
INFERENCE_ERROR_PREFIX = "推理失败"


@runtime_checkable
class ResponseGenerator(Protocol):
    """响应生成器的结构协议。

    Runtime 的 ``inference`` 模块（任务 10.1）所产出的推理引擎应满足本协议：
    对一次非空用户输入返回一条字符串回复（R12-AC6）。引擎可选地暴露 ``close()``
    方法以在 REPL 退出时释放模型资源（R12-AC8）。

    :func:`run_repl` 同时也接受一个等价的 ``Callable[[str], str]``，因此测试可直接
    注入一个普通函数作为响应生成器，无需构造完整对象。
    """

    def generate(self, user_input: str) -> str:
        """对一次非空用户输入返回一条回复。"""
        ...


#: :func:`run_repl` 可接受的响应生成器类型：满足 :class:`ResponseGenerator` 协议的
#: 对象，或等价的 ``Callable[[str], str]``。
GeneratorLike = Union[ResponseGenerator, Callable[[str], str]]


def _resolve_generate(generator: GeneratorLike) -> Callable[[str], str]:
    """将注入的生成器归一化为 ``Callable[[str], str]``。

    优先使用对象的 ``generate`` 方法（推理引擎的约定接口）；若不存在则在对象本身
    可调用时直接将其作为生成函数（便于测试注入普通函数）。

    :param generator: 满足 :class:`ResponseGenerator` 协议的对象或可调用对象。
    :returns: 接收用户输入字符串、返回回复字符串的可调用对象。
    :raises TypeError: ``generator`` 既无可调用的 ``generate`` 方法，自身也不可调用。
    """

    generate_method = getattr(generator, "generate", None)
    if callable(generate_method):
        return generate_method
    if callable(generator):
        return generator
    raise TypeError(
        "响应生成器必须暴露可调用的 generate(user_input) 方法或自身可调用，"
        f"但收到的是 {type(generator).__name__!r}"
    )


def _release(generator: GeneratorLike) -> None:
    """释放生成器持有的模型资源（R12-AC8）。

    若生成器对象暴露可调用的 ``close()`` 方法则调用之；任何释放阶段的异常都被
    吞掉，以保证 REPL 始终以退出码 0 正常结束、且不向用户透出 traceback。

    :param generator: 注入的响应生成器；普通函数无 ``close`` 方法时为无操作。
    """

    close = getattr(generator, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - 释放失败不应影响正常退出
            pass


def run_repl(
    generator: GeneratorLike,
    *,
    prompt: str = PROMPT,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """运行 ``ego run`` 的交互式 REPL 循环（需求 12.6、12.7、12.8）。

    循环逻辑（见模块文档）：每轮先打印固定提示符 ``prompt``；读取一行输入——

    * 读到 EOF（``readline()`` 返回空串，即 Ctrl-D）→ 释放资源并以退出码 0 结束
      （R12-AC8）。
    * 输入去除首尾空白后为空（空行或仅空白）→ 不调用推理，直接进入下一轮重打提示符
      （R12-AC7）。
    * 输入去除首尾空白后等于 :data:`QUIT_COMMAND`（``:quit``）→ 释放资源并以退出码
      0 结束（R12-AC8）。
    * 其余非空输入 → 调用一次响应生成并打印回复（R12-AC6）；若生成抛出异常，则向
      标准错误打印一条简要错误信息并继续循环，绝不透出 traceback（R12-AC6 健壮性）。

    :param generator: 响应生成器，满足 :class:`ResponseGenerator` 协议的对象（如任务
        10.1 的推理引擎）或等价的 ``Callable[[str], str]``。由 ``commands.run``
        （任务 10.3）注入；本模块不导入 ``inference`` 以避免硬依赖。
    :param prompt: 每轮交互前打印的提示符，默认 :data:`PROMPT`（``ego> ``）。
    :param stdin: 输入流，默认 :data:`sys.stdin`；测试可注入 :class:`io.StringIO`。
    :param stdout: 输出流（提示符与回复），默认 :data:`sys.stdout`。
    :param stderr: 错误流（推理异常的简要信息），默认 :data:`sys.stderr`。
    :returns: 始终为 :data:`~ego_pod.exit_codes.EXIT_OK`（0）——REPL 仅在 ``:quit``
        或 EOF 时正常结束（R12-AC8）。
    :raises TypeError: ``generator`` 无法被解析为响应生成函数（见 :func:`_resolve_generate`）。
    """

    source = sys.stdin if stdin is None else stdin
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr

    generate = _resolve_generate(generator)

    try:
        while True:
            # 每轮重打固定提示符（不换行，使输入与提示符同行）。
            print(prompt, end="", file=out, flush=True)

            line = source.readline()
            if line == "":
                # EOF（Ctrl-D）：补一个换行结束提示符所在行，随后退出（R12-AC8）。
                print("", file=out, flush=True)
                break

            user_input = line.strip()

            # R12-AC7：空行或仅空白字符 → 不调用推理，直接重打提示符。
            if user_input == "":
                continue

            # R12-AC8：:quit → 退出。
            if user_input == QUIT_COMMAND:
                break

            # R12-AC6：非空输入 → 调用一次响应生成；异常被捕获后打印简要错误并继续。
            try:
                reply = generate(user_input)
            except Exception as exc:  # noqa: BLE001 - 不让 traceback 透出给用户
                print(f"{INFERENCE_ERROR_PREFIX}: {exc}", file=err, flush=True)
                continue

            print(str(reply), file=out, flush=True)
    finally:
        # R12-AC8：无论因 :quit、EOF 还是其它原因结束，均释放模型资源。
        _release(generator)

    return EXIT_OK
