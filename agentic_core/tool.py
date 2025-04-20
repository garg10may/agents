"""
Tool base class, registry, dynamic loading
"""
from typing import Callable, Dict, Any
class Tool:
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func
class ToolRegistry:
    _tools: Dict[str, Tool] = {}
    @classmethod
    def register_tool(cls, tool: Tool):
        cls._tools[tool.name] = tool
    @classmethod
    def get_tools(cls, names=None):
        if names is None:
            return list(cls._tools.values())
        return [cls._tools[n] for n in names if n in cls._tools]
    @classmethod
    def call_tool(cls, name: str, *args, **kwargs) -> Any:
        return cls._tools[name].func(*args, **kwargs)
