# SPDX-License-Identifier: GPL-3.0-or-later
"""Call any registered ComfyUI node (V1 or V3) from Python, the way the executor does.

V3 nodes get their flat inputs ("method.shrink", "ref_images.ref_image_0") nested by ComfyUI's own
get_finalized_class_inputs / build_nested_inputs, so the node sees exactly what it sees in a graph.
"""
from __future__ import annotations

import inspect

import nodes
from comfy_api.internal import _ComfyNodeInternal, make_locked_method_func
from comfy_api.latest import _io


def _cls(class_type: str):
    cls = nodes.NODE_CLASS_MAPPINGS.get(class_type)
    if cls is None:
        raise RuntimeError(f"Artfat Face Refine needs the node '{class_type}', which is not installed.")
    return cls


def has_node(class_type: str) -> bool:
    return class_type in nodes.NODE_CLASS_MAPPINGS


async def call_node(class_type: str, **inputs):
    """Run one node and return its outputs as a tuple."""
    cls = _cls(class_type)
    if issubclass(cls, _ComfyNodeInternal):
        _, _hidden, v3_data = _io.get_finalized_class_inputs(cls.INPUT_TYPES(), inputs)
        v3_data = dict(v3_data or {})
        v3_data.setdefault("hidden_inputs", {})  # the executor fills these (prompt, unique_id...); none are needed here
        cls.VALIDATE_CLASS()
        clone = cls.PREPARE_CLASS_CLONE(v3_data)
        func = make_locked_method_func(cls, cls.FUNCTION, clone)
        args = _io.build_nested_inputs(dict(inputs), v3_data)
    else:
        func = getattr(cls(), cls.FUNCTION)
        args = inputs
    result = func(**args)
    if inspect.isawaitable(result):
        result = await result
    if isinstance(result, dict) and "result" in result:
        result = result["result"]
    if not isinstance(result, tuple) and hasattr(result, "result"):
        result = result.result
    return tuple(result) if result is not None else ()
