"""`ego_pod.trainer.detect` 后端探测与优先级选择的单元测试。

覆盖设计文档「多后端 Trainer 抽象」一节与需求 R8：

* R8-AC1：按固定优先级 Unsloth(CUDA) → PeftCuda(CUDA) → Mlx(Apple Silicon)
  → PeftCpu(CPU) → Stub 逐个探测，选择第一个 ``is_available()`` 为真的后端，
  且 Stub_Backend 始终可用作为最终回退；
* R8-AC2：选中后端暴露 ``name`` 与 ``hardware`` 属性供流水线打印。

测试通过 monkeypatch 模拟各后端的 ``is_available()`` 返回值（等价于模拟
import 成功/失败与 CUDA/MLX 硬件可用性），以及通过 patch ``_load_backend_class``
模拟后端模块不可导入（ImportError）的情形，验证选中顺序与 Stub 兜底。

测试不依赖任何真实 GPU/ML 依赖或网络。
"""

from __future__ import annotations

import pytest

from ego_pod.exit_codes import TrainerError
from ego_pod.trainer import detect
from ego_pod.trainer.mlx_backend import MlxBackend
from ego_pod.trainer.peft_backend import PeftCpuBackend, PeftCudaBackend
from ego_pod.trainer.stub import StubBackend
from ego_pod.trainer.unsloth_backend import UnslothBackend

# 按优先级从高到低排列的真实后端类（不含始终可用的 Stub 末位回退）。
_REAL_BACKENDS = (UnslothBackend, PeftCudaBackend, MlxBackend, PeftCpuBackend)
_ALL_BACKENDS = _REAL_BACKENDS + (StubBackend,)


# ---------------------------------------------------------------------------
# 辅助：按可用集合模拟各后端 is_available()
# ---------------------------------------------------------------------------


def _set_availability(monkeypatch, available: set[type]) -> None:
    """将每个候选后端类的 ``is_available()`` patch 为固定返回值。

    ``available`` 中的后端类返回 ``True``，其余返回 ``False``。这等价于模拟
    "import 成功 + 硬件可用"（True）与"import 失败或硬件不可用"（False）。

    Stub_Backend 的真实实现 ``is_available()`` 恒为 ``True``，此处保持其默认
    行为（除非显式从 ``available`` 中排除以验证异常分支）。
    """

    for backend_cls in _ALL_BACKENDS:
        is_avail = backend_cls in available
        monkeypatch.setattr(
            backend_cls,
            "is_available",
            classmethod(lambda cls, _v=is_avail: _v),
        )


# ---------------------------------------------------------------------------
# R8-AC1：优先级选择顺序
# ---------------------------------------------------------------------------


def test_all_available_selects_unsloth_first(monkeypatch):
    """全部后端可用时，选中最高优先级 Unsloth。"""

    _set_availability(monkeypatch, set(_ALL_BACKENDS))

    backend = detect.detect_backend()

    assert isinstance(backend, UnslothBackend)
    assert backend.name == "Unsloth"
    assert backend.hardware == "CUDA"


def test_unsloth_unavailable_falls_to_peft_cuda(monkeypatch):
    """Unsloth 不可用、其余可用时，选中 PeftCuda。"""

    _set_availability(
        monkeypatch, {PeftCudaBackend, MlxBackend, PeftCpuBackend, StubBackend}
    )

    backend = detect.detect_backend()

    assert isinstance(backend, PeftCudaBackend)
    assert backend.hardware == "CUDA"


def test_no_cuda_falls_to_mlx(monkeypatch):
    """两个 CUDA 后端均不可用、MLX 可用时，选中 Mlx。"""

    _set_availability(monkeypatch, {MlxBackend, PeftCpuBackend, StubBackend})

    backend = detect.detect_backend()

    assert isinstance(backend, MlxBackend)
    assert backend.hardware == "Apple Silicon"


def test_only_peft_cpu_and_stub_falls_to_peft_cpu(monkeypatch):
    """仅 PeftCpu 与 Stub 可用时，选中优先级更高的 PeftCpu。"""

    _set_availability(monkeypatch, {PeftCpuBackend, StubBackend})

    backend = detect.detect_backend()

    assert isinstance(backend, PeftCpuBackend)
    assert backend.hardware == "CPU"


def test_all_real_backends_unavailable_falls_to_stub(monkeypatch):
    """所有真实后端不可用时，回退到 Stub_Backend。"""

    _set_availability(monkeypatch, {StubBackend})

    backend = detect.detect_backend()

    assert isinstance(backend, StubBackend)
    assert backend.name == "stub"
    assert backend.hardware == "STUB"


def test_stub_is_default_fallback_with_real_is_available(monkeypatch):
    """不 patch Stub：仅模拟四个真实后端全不可用，验证真实 Stub 兜底。

    保留 ``StubBackend.is_available`` 的真实实现（恒为 ``True``），仅将四个
    真实后端置为不可用，确认探测链最终回退到 Stub。
    """

    for backend_cls in _REAL_BACKENDS:
        monkeypatch.setattr(
            backend_cls, "is_available", classmethod(lambda cls: False)
        )

    backend = detect.detect_backend()

    assert isinstance(backend, StubBackend)
    assert StubBackend.is_available() is True


# ---------------------------------------------------------------------------
# 选中顺序的穷举验证：第一个可用者即为选中者
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first_available_index", "expected_cls"),
    [
        (0, UnslothBackend),
        (1, PeftCudaBackend),
        (2, MlxBackend),
        (3, PeftCpuBackend),
        (4, StubBackend),
    ],
)
def test_first_available_in_priority_is_selected(
    monkeypatch, first_available_index, expected_cls
):
    """从某个优先级起全部可用、其之前全部不可用 → 选中该优先级后端。"""

    available = set(_ALL_BACKENDS[first_available_index:])
    _set_availability(monkeypatch, available)

    backend = detect.detect_backend()

    assert isinstance(backend, expected_cls)


# ---------------------------------------------------------------------------
# 模拟 import 失败（_load_backend_class 返回 None）时跳过该候选
# ---------------------------------------------------------------------------


def test_unimportable_backend_is_skipped(monkeypatch):
    """最高优先级后端模块不可导入（ImportError）时被跳过，继续向下探测。

    通过 patch ``_load_backend_class`` 对 Unsloth 返回 ``None`` 模拟其模块因
    缺失依赖而无法导入；其余后端正常加载。即便所有 is_available 均为真，
    选中者也应为 PeftCuda（Unsloth 被跳过）。
    """

    _set_availability(monkeypatch, set(_ALL_BACKENDS))

    real_loader = detect._load_backend_class

    def fake_loader(module_path: str, class_name: str):
        if class_name == "UnslothBackend":
            return None  # 模拟 import unsloth 失败
        return real_loader(module_path, class_name)

    monkeypatch.setattr(detect, "_load_backend_class", fake_loader)

    backend = detect.detect_backend()

    assert isinstance(backend, PeftCudaBackend)


def test_is_available_exception_treated_as_unavailable(monkeypatch):
    """后端 is_available() 探测抛异常视为不可用并跳过（不向上传播）。"""

    _set_availability(monkeypatch, set(_ALL_BACKENDS))

    def boom(cls):
        raise RuntimeError("硬件查询失败")

    monkeypatch.setattr(UnslothBackend, "is_available", classmethod(boom))

    backend = detect.detect_backend()

    # Unsloth 探测抛异常被视为不可用，回退到下一优先级 PeftCuda
    assert isinstance(backend, PeftCudaBackend)


def test_trainer_error_when_even_stub_unloadable(monkeypatch):
    """连 Stub 都无法加载（安装损坏）时抛 TrainerError。"""

    def fake_loader(module_path: str, class_name: str):
        return None  # 所有候选均"不可导入"

    monkeypatch.setattr(detect, "_load_backend_class", fake_loader)

    with pytest.raises(TrainerError):
        detect.detect_backend()


# ---------------------------------------------------------------------------
# _load_backend_class 行为
# ---------------------------------------------------------------------------


def test_load_backend_class_returns_class_for_existing():
    cls = detect._load_backend_class("ego_pod.trainer.stub", "StubBackend")
    assert cls is StubBackend


def test_load_backend_class_returns_none_for_missing_module():
    cls = detect._load_backend_class("ego_pod.trainer.does_not_exist", "Whatever")
    assert cls is None


def test_load_backend_class_returns_none_for_missing_attr():
    cls = detect._load_backend_class("ego_pod.trainer.stub", "NoSuchBackend")
    assert cls is None


# ---------------------------------------------------------------------------
# 硬件检测工具
# ---------------------------------------------------------------------------


def test_apple_silicon_detection_true_on_arm64_macos(monkeypatch):
    monkeypatch.setattr(detect.sys, "platform", "darwin")
    monkeypatch.setattr(detect.platform, "machine", lambda: "arm64")
    assert detect.apple_silicon_is_available() is True


def test_apple_silicon_detection_false_on_x86_macos(monkeypatch):
    monkeypatch.setattr(detect.sys, "platform", "darwin")
    monkeypatch.setattr(detect.platform, "machine", lambda: "x86_64")
    assert detect.apple_silicon_is_available() is False


def test_apple_silicon_detection_false_on_linux(monkeypatch):
    monkeypatch.setattr(detect.sys, "platform", "linux")
    monkeypatch.setattr(detect.platform, "machine", lambda: "arm64")
    assert detect.apple_silicon_is_available() is False


def test_cuda_is_available_returns_bool():
    # 不依赖具体硬件，仅验证返回布尔值且不抛异常。
    assert isinstance(detect.cuda_is_available(), bool)
