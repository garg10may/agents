import os
import requests
from dotenv import load_dotenv
import openai
from typing import Callable, Dict, List, Any

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# --- Define Tools ---
def web_search_tool(query: str) -> str:
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {"q": query}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        snippets = [res.get("snippet") or res.get("title") for res in data.get("organic", [])]
        return "\n".join(snippet for snippet in snippets if snippet)
    else:
        return f"[Error fetching search results: {response.status_code}]"

def calculator_tool(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"[Calculator error: {e}]"

def summarize_tool(text: str) -> str:
    prompt = f"Summarize the following text in 2 sentences:\n{text}"
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

# --- Tool Registry ---
tools = {
    "web_search": {
        "description": "Search the web for information on any topic.",
        "func": web_search_tool
    },
    "calculator": {
        "description": "Evaluate a math expression.",
        "func": calculator_tool
    },
    "summarize": {
        "description": "Summarize a block of text.",
        "func": summarize_tool
    }
}

# --- Agent Reasoning Loop ---
def agentic_reasoning_loop(goal: str, max_iters: int = 5) -> str:
    history = []
    for step in range(max_iters):
        # Compose the agent's prompt
        tool_descriptions = "\n".join(f"- {name}: {meta['description']}" for name, meta in tools.items())
        context = "\n".join(f"Step {i+1}: {h}" for i, h in enumerate(history))
        prompt = (
            f"You are an agent with access to these tools:\n{tool_descriptions}\n\n"
            f"Your goal: {goal}\n"
            f"Previous steps:\n{context if context else 'None'}\n\n"
            "Decide your next action in the format:\n"
            "TOOL: <tool_name>\nINPUT: <input_for_tool>\n"
            "If you believe the goal is complete, reply with:\nDONE: <final_answer>\n"
        )
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        agent_reply = response.choices[0].message.content.strip()
        print(f"[Agent step {step+1}]\n{agent_reply}\n")
        if agent_reply.startswith("DONE:"):
            return agent_reply[len("DONE:"):].strip()
        elif agent_reply.startswith("TOOL:"):
            lines = agent_reply.split("\n")
            tool_line = next((l for l in lines if l.startswith("TOOL:")), None)
            input_line = next((l for l in lines if l.startswith("INPUT:")), None)
            if tool_line and input_line:
                tool_name = tool_line[len("TOOL:"):].strip()
                tool_input = input_line[len("INPUT:"):].strip()
                if tool_name in tools:
                    tool_result = tools[tool_name]["func"](tool_input)
                    history.append(f"Agent used {tool_name} with input '{tool_input}' and got: {tool_result}")
                else:
                    history.append(f"Agent tried unknown tool '{tool_name}'")
            else:
                history.append("Agent reply format error.")
        else:
            history.append("Agent reply format error.")
    return "[Agent did not complete the goal in time.]"

if __name__ == "__main__":
    user_goal = input("Enter your complex goal: ")
    final_answer = agentic_reasoning_loop(user_goal)
    print("\n--- Final Agent Answer ---\n")
    print(final_answer)
