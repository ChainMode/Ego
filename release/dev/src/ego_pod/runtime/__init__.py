"""Runtime 运行时模块（``ego run`` 的模型加载与交互）。

本模块实现 ``ego run`` 命令的两项核心能力，对应设计文档 §7「Runtime 模块」：

* ``inference`` —— 从 ``ego-output/`` 加载 Ego_Package（``manifest.json`` /
  ``<edition>/config.json`` / ``<edition>/adapters/ego-lora.safetensors``），
  根据 ``manifest.json`` 的 ``training_backend`` 标记选择 Stub 推理路径或真实
  后端推理路径，并向上层（REPL）提供一个可对用户输入生成回复的会话对象
  （需求 12.1、12.5、12.9）。
* ``repl`` —— ``ego run`` 的交互式循环（任务 10.2，本任务不实现）。

本任务（10.1）仅实现 ``inference``：``load_inference_session`` 负责加载与
完整性校验（缺失/损坏文件抛 :class:`~ego_pod.exit_codes.RuntimeError_`，
退出码 1），返回的 :class:`~ego_pod.runtime.inference.InferenceSession` 暴露
``generate(user_input) -> str`` 供 REPL 逐轮调用。
"""

from ego_pod.runtime.inference import (
    InferenceSession,
    StubInferenceSession,
    BackendInferenceSession,
    load_inference_session,
    STUB_TRAINING_BACKEND,
    STUB_MODE_STARTUP_NOTICE,
    STUB_PLACEHOLDER_REPLY,
)

__all__ = [
    "InferenceSession",
    "StubInferenceSession",
    "BackendInferenceSession",
    "load_inference_session",
    "STUB_TRAINING_BACKEND",
    "STUB_MODE_STARTUP_NOTICE",
    "STUB_PLACEHOLDER_REPLY",
]
