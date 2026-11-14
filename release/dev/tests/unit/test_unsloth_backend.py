"""`ego_pod.trainer.unsloth_backend.UnslothBackend` 的继承与契约单元测试。

覆盖设计文档「Unsloth 后端改为继承共享实现」一节与需求：

* R9-AC1：Unsloth 通过继承复用 :class:`_PeftBackendBase` 的真实训练/导出实现，
  仅覆盖 :meth:`is_available` 与 :meth:`load_base_model`；
* R9-AC5：``identifier == "unsloth"``，作为写入 ``manifest.json`` 的稳定标识来源；
* R10-AC1：仍满足 ``TrainingBackend`` Protocol 形状不变。

本文件刻意**不**在模块顶层导入 ``unsloth`` / ``peft`` / ``transformers`` /
``torch``：被测模块的可选依赖均以 ``try/except ImportError`` 哨兵 + 函数内惰性
导入包裹，故本套测试可在不安装任何 ML 依赖的环境（如 CI）中运行（R8-AC4/AC5）。
"""

from __future__ import annotations

from ego_pod.trainer.base import TrainingBackend
from ego_pod.trainer.peft_backend import _PeftBackendBase
from ego_pod.trainer.unsloth_backend import UnslothBackend


# ---------------------------------------------------------------------------
# R9-AC1：UnslothBackend 通过“继承”复用共享实现（而非组合/委托）
# ---------------------------------------------------------------------------


def test_unsloth_is_subclass_of_peft_base():
    # 继承关系：_PeftBackendBase 必须出现在 MRO 中（不是组合委托）。
    assert issubclass(UnslothBackend, _PeftBackendBase)
    assert _PeftBackendBase in UnslothBackend.__mro__


# ---------------------------------------------------------------------------
# R9-AC5：后端身份属性（manifest training_backend 标识来源）
# ---------------------------------------------------------------------------


def test_identity_attributes():
    backend = UnslothBackend()
    assert backend.identifier == "unsloth"
    assert backend.name == "Unsloth"
    assert backend.hardware == "CUDA"


# ---------------------------------------------------------------------------
# R10-AC1：实例结构上满足 runtime_checkable 的 TrainingBackend Protocol
# ---------------------------------------------------------------------------


def test_satisfies_training_backend_protocol():
    backend = UnslothBackend()
    assert isinstance(backend, TrainingBackend)


# ---------------------------------------------------------------------------
# R9-AC1：train / export_lora_safetensors 继承自基类（未被覆盖）
# ---------------------------------------------------------------------------


def test_train_inherited_from_base():
    # 同一函数对象 => 直接复用基类实现（继承，而非重新定义）。
    assert UnslothBackend.train is _PeftBackendBase.train
    # 未在子类自身命名空间中重新定义。
    assert "train" not in UnslothBackend.__dict__


def test_export_inherited_from_base():
    assert (
        UnslothBackend.export_lora_safetensors
        is _PeftBackendBase.export_lora_safetensors
    )
    assert "export_lora_safetensors" not in UnslothBackend.__dict__


# ---------------------------------------------------------------------------
# R9-AC1：is_available / load_base_model 由 UnslothBackend 覆盖（非基类实现）
# ---------------------------------------------------------------------------


def test_load_base_model_overridden():
    # 子类自身定义了 load_base_model，且与基类的不是同一函数对象。
    assert "load_base_model" in UnslothBackend.__dict__
    assert UnslothBackend.load_base_model is not _PeftBackendBase.load_base_model


def test_is_available_overridden():
    # UnslothBackend 自身声明 is_available；_PeftBackendBase 本身并不定义它
    # （仅 PeftCuda/PeftCpu 子类各自定义），故二者来源不同。
    assert "is_available" in UnslothBackend.__dict__
    assert "is_available" not in _PeftBackendBase.__dict__
