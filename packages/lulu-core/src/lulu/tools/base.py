"""Tool: what the harness registers with the model and dispatches to.

A Tool is deliberately dumb about permissions -- it doesn't know whether
it's allowed to run, it just runs when asked. permissions.py decides
whether dispatch happens at all; ToolRegistry (registry.py) just maps
name -> handler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

ToolHandler = Callable[[dict[str, Any]], str]


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
