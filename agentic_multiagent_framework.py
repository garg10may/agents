import os
import json
import requests
from dotenv import load_dotenv
import openai
from datetime import datetime
from typing import List, Dict, Any, Optional

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# --- TOOL DEFINITIONS (same as before, omitted for brevity) ---
# ... (reuse all tool functions and schemas from agentic_framework.py)
from agentic_framework import (
    web_search, calculator, summarize, get_time, get_date, extract_entities, translate,
    wikipedia_search, code_executor, sentiment_analysis, url_reader, function_schemas, tool_funcs
)

# --- AGENT CLASS ---
class Agent:
    def __init__(self, name: str, system_prompt: str, toolset: Optional[List[str]] = None):
        self.name = name
        self.system_prompt = system_prompt
        self.toolset = toolset or list(tool_funcs.keys())

    def available_functions(self):
        return [f for f in function_schemas if f["name"] in self.toolset]

    def act(self, messages: List[Dict[str, Any]], max_iters: int = 5, verbose: bool = True):
        reasoning_steps = []
        for i in range(max_iters):
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-1106",
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
            else:
                step_info += f"Agent produced final answer."
                reasoning_steps.append(step_info)
                return msg.content, reasoning_steps, messages
            reasoning_steps.append(step_info)
        return "[Agent did not complete the goal in time.]", reasoning_steps, messages

# --- MULTI-AGENT PIPELINE ---
def multiagent_pipeline(goal: str, agent_configs: List[Dict[str, Any]], max_iters: int = 5, verbose: bool = True):
    agents = [Agent(**cfg) for cfg in agent_configs]
    messages = [{"role": "system", "content": agents[0].system_prompt}, {"role": "user", "content": goal}]
    all_steps = []
    last_output = None
    for idx, agent in enumerate(agents):
        if idx > 0:
            # Pass previous output as user message to next agent
            messages = [{"role": "system", "content": agent.system_prompt}, {"role": "user", "content": last_output}]
        answer, steps, messages = agent.act(messages, max_iters=max_iters, verbose=verbose)
        all_steps.extend(steps)
        last_output = answer
    return last_output, all_steps

if __name__ == "__main__":
    # Example: Researcher, Writer, Reviewer pipeline
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
            "system_prompt": "You are a critical reviewer. Use sentiment analysis and summarization to review and improve the article provided.",
            "toolset": ["summarize", "sentiment_analysis"]
        }
    ]
    user_goal = input("Enter your multi-agent goal: ")
    final_answer, steps = multiagent_pipeline(user_goal, agent_configs)
    print("\n--- Multi-Agent Reasoning Steps ---\n")
    for step in steps:
        print(step)
    print("\n--- Final Output ---\n")
    print(final_answer)
