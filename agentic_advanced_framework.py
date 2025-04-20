import os
import json
import threading
from typing import List, Dict, Any, Callable, Optional
from queue import Queue
import openai
from dotenv import load_dotenv
from agentic_framework import (
    web_search, calculator, summarize, get_time, get_date, extract_entities, translate,
    wikipedia_search, code_executor, sentiment_analysis, url_reader, function_schemas, tool_funcs
)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Shared Memory/Workspace ---
class Workspace:
    def __init__(self):
        self.memory: Dict[str, Any] = {}
        self.lock = threading.Lock()
    def get(self, key, default=None):
        with self.lock:
            return self.memory.get(key, default)
    def set(self, key, value):
        with self.lock:
            self.memory[key] = value
    def append(self, key, value):
        with self.lock:
            self.memory.setdefault(key, []).append(value)
    def all(self):
        with self.lock:
            return dict(self.memory)

# --- Agent Definition ---
class Agent:
    def __init__(self, name: str, system_prompt: str, toolset: Optional[List[str]] = None, model: str = "gpt-3.5-turbo-1106"):
        self.name = name
        self.system_prompt = system_prompt
        self.toolset = toolset or list(tool_funcs.keys())
        self.model = model
        self.memory: List[Dict[str, Any]] = []
        self.workspace: Optional[Workspace] = None
        self.next_agent: Optional[str] = None  # For dynamic routing
    def available_functions(self):
        return [f for f in function_schemas if f["name"] in self.toolset]
    def set_workspace(self, ws: Workspace):
        self.workspace = ws
    def send_message(self, to_agent: str, content: str):
        if self.workspace:
            self.workspace.append(f"msg_{to_agent}", {"from": self.name, "content": content})
    def receive_messages(self) -> List[Dict[str, Any]]:
        if self.workspace:
            msgs = self.workspace.get(f"msg_{self.name}", [])
            self.workspace.memory[f"msg_{self.name}"] = []  # Clear after reading
            return msgs
        return []
    def act(self, input_msg: str, max_iters: int = 5, verbose: bool = True) -> (str, List[str]):
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": input_msg}]
        reasoning_steps = []
        for i in range(max_iters):
            # Add received messages to context
            for msg in self.receive_messages():
                messages.append({"role": "user", "content": f"[Message from {msg['from']}]: {msg['content']}"})
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                functions=self.available_functions(),
                function_call="auto"
            )
            msg = response.choices[0].message
            step_info = f"**{self.name} Step {i+1}:**\n"
            if msg.function_call:
                fn_name = msg.function_call.name
                fn_args = json.loads(msg.function_call.arguments)
                step_info += f"Agent called tool: `{fn_name}` with args: {fn_args}\n"
                result = tool_funcs[fn_name](**fn_args)
                step_info += f"Tool output: {result}\n"
                messages.append({"role": "function", "name": fn_name, "content": str(result)})
                # Optionally, store result in workspace
                if self.workspace:
                    self.workspace.set(f"{self.name}_last_tool_result", result)
            else:
                step_info += f"Agent produced final answer."
                reasoning_steps.append(step_info)
                return msg.content, reasoning_steps
            reasoning_steps.append(step_info)
        return "[Agent did not complete the goal in time.]", reasoning_steps

# --- Dynamic Routing & Critic Loops ---
def advanced_pipeline(goal: str, agent_configs: List[Dict[str, Any]],
                     routing_fn: Optional[Callable[[str, str, Workspace], str]] = None,
                     max_iters: int = 5, verbose: bool = True, parallel: bool = False,
                     human_callback: Optional[Callable[[str, str], bool]] = None):
    workspace = Workspace()
    agents = {cfg["name"]: Agent(**cfg) for cfg in agent_configs}
    for agent in agents.values():
        agent.set_workspace(workspace)
    steps = []
    current_agent = agent_configs[0]["name"]
    input_msg = goal
    while True:
        agent = agents[current_agent]
        answer, agent_steps = agent.act(input_msg, max_iters=max_iters, verbose=verbose)
        steps.extend(agent_steps)
        # Human-in-the-loop: user can approve/modify/stop
        if human_callback and not human_callback(current_agent, answer):
            steps.append(f"**{current_agent}:** Human stopped or modified output.")
            break
        # Critic/review loop: if agent is a reviewer, can send back for revision
        if "review" in agent.name.lower() and "needs revision" in answer.lower():
            steps.append(f"**{current_agent}:** Sent back for revision.")
            current_agent = agent_configs[1]["name"]  # Send back to writer, for example
            input_msg = workspace.get(f"{current_agent}_last_tool_result", answer)
            continue
        # Dynamic routing: routing_fn decides next agent
        if routing_fn:
            next_agent = routing_fn(current_agent, answer, workspace)
            if next_agent is None:
                break
            current_agent = next_agent
            input_msg = answer
        else:
            idx = [cfg["name"] for cfg in agent_configs].index(current_agent)
            if idx + 1 >= len(agent_configs):
                break
            current_agent = agent_configs[idx + 1]["name"]
            input_msg = answer
    return answer, steps, workspace.all()

# --- Example Usage ---
if __name__ == "__main__":
    agent_configs = [
        {
            "name": "Researcher",
            "system_prompt": "You are a research agent. Use web search, Wikipedia, and summarization to gather and condense information for the topic provided.",
            "toolset": ["web_search", "wikipedia_search", "summarize"]
        },
        {
            "name": "Writer",
            "system_prompt": "You are a blog writer. Use summarization, translation, and entity extraction to write a clear, engaging article from the research notes provided.",
            "toolset": ["summarize", "translate", "extract_entities"]
        },
        {
            "name": "Reviewer",
            "system_prompt": "You are a critical reviewer. Use sentiment analysis and summarization to review and improve the article provided. If changes are needed, reply with 'Needs revision'.",
            "toolset": ["summarize", "sentiment_analysis"]
        }
    ]
    def routing_fn(current_agent, answer, ws):
        # Example: If reviewer says 'Needs revision', send back to Writer
        if "reviewer" in current_agent.lower() and "needs revision" in answer.lower():
            return "Writer"
        idx = [cfg["name"] for cfg in agent_configs].index(current_agent)
        if idx + 1 >= len(agent_configs):
            return None
        return agent_configs[idx + 1]["name"]
    final_answer, steps, memory = advanced_pipeline(
        "Write a blog post about the latest AI breakthrough, then review it for sentiment and summarize the review.",
        agent_configs,
        routing_fn=routing_fn,
        max_iters=5,
        verbose=True
    )
    print("\n--- Advanced Multi-Agent Reasoning Steps ---\n")
    for step in steps:
        print(step)
    print("\n--- Final Output ---\n")
    print(final_answer)
    print("\n--- Shared Workspace ---\n")
    print(memory)
