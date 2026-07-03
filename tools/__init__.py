"""AUBO 项目工具的稳定公开入口。"""

from __future__ import annotations

from importlib import import_module

_LAZY_EXPORTS = {
    "AuboActionToolFns": ("tools.rl", "AuboActionToolFns"),
    "AuboBufferToolFns": ("tools.rl", "AuboBufferToolFns"),
    "AuboCameraFns": ("tools.camera", "AuboCameraFns"),
    "AuboContactToolFns": ("tools.contact", "AuboContactToolFns"),
    "AuboToolFns": ("tools.scene", "AuboToolFns"),
}

__all__ = [
    "AuboActionToolFns",
    "AuboBufferToolFns",
    "AuboCameraFns",
    "AuboContactToolFns",
    "AuboToolFns",
]


def __getattr__(name: str):
    """仅在调用公开工具时加载 Torch/Isaac 依赖。"""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'tools' has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
