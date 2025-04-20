"""
Agent class: role, persona, tool access, memory, messaging, act() loop
"""
from typing import List, Dict, Any, Optional, Callable
from .tool import ToolRegistry
from .memory import Memory
from .messaging import Messaging

class Agent:
    def __init__(self, name: str, system_prompt: str, toolset: Optional[List[str]] = None, model: str = "gpt-3.5-turbo-1106"):
        self.name = name
        self.system_prompt = system_prompt
        self.toolset = toolset or []
        self.model = model
        self.memory = Memory()
        self.messaging = Messaging(self.name)
    def available_tools(self):
        return ToolRegistry.get_tools(self.toolset)
    def act(self, input_msg: str, max_iters: int = 5, verbose: bool = True) -> (str, List[str]):
        # Placeholder for agent act loop
        return "[Agent output]", []
