import os
import json
import requests
from dotenv import load_dotenv
import openai
from datetime import datetime

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# --- TOOL DEFINITIONS ---
def web_search(query: str) -> str:
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        snippets = [res.get("snippet") or res.get("title") for res in data.get("organic", [])]
        return "\n".join(snippet for snippet in snippets if snippet)
    else:
        return f"[Error fetching search results: {response.status_code}]"

def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"[Calculator error: {e}]"

def summarize(text: str) -> str:
    prompt = f"Summarize the following text in 2 sentences:\n{text}"
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

def get_time(_: str = None) -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_date(_: str = None) -> str:
    return datetime.now().strftime("%Y-%m-%d")

def extract_entities(text: str) -> str:
    prompt = f"Extract all named entities (people, places, organizations, dates, etc.) from the following text as a comma-separated list:\n{text}"
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

def translate(text: str, target_language: str) -> str:
    prompt = f"Translate the following text to {target_language}:\n{text}"
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

def wikipedia_search(query: str) -> str:
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("extract", "No summary found.")
    else:
        return f"[Wikipedia error: {response.status_code}]"

def code_executor(code: str, language: str = "python") -> str:
    if language != "python":
        return "[Only Python code execution is supported in this demo.]"
    try:
        local_vars = {}
        exec(code, {}, local_vars)
        return str(local_vars) if local_vars else "[Code executed successfully, no output.]"
    except Exception as e:
        return f"[Code execution error: {e}]"

def sentiment_analysis(text: str) -> str:
    prompt = f"What is the sentiment of the following text? Respond with Positive, Negative, or Neutral.\n{text}"
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20
    )
    return response.choices[0].message.content.strip()

def url_reader(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.text[:2000] + ("..." if len(resp.text) > 2000 else "")
        else:
            return f"[URL error: {resp.status_code}]"
    except Exception as e:
        return f"[URL error: {e}]"

# --- FUNCTION SCHEMAS FOR OPENAI FUNCTION CALLING ---
function_schemas = [
    {
        "name": "web_search",
        "description": "Search the web for information.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    },
    {
        "name": "calculator",
        "description": "Evaluate a math expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]
        }
    },
    {
        "name": "summarize",
        "description": "Summarize a block of text.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"]
        }
    },
    {
        "name": "get_time",
        "description": "Get the current time.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "get_date",
        "description": "Get the current date.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "extract_entities",
        "description": "Extract named entities from text.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"]
        }
    },
    {
        "name": "translate",
        "description": "Translate text to a target language.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "target_language": {"type": "string"}
            },
            "required": ["text", "target_language"]
        }
    },
    {
        "name": "wikipedia_search",
        "description": "Search Wikipedia for a summary.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    },
    {
        "name": "code_executor",
        "description": "Execute a Python code snippet.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "language": {"type": "string", "default": "python"}
            },
            "required": ["code"]
        }
    },
    {
        "name": "sentiment_analysis",
        "description": "Analyze sentiment of text.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"]
        }
    },
    {
        "name": "url_reader",
        "description": "Read and return the content of a URL (first 2000 chars).",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"]
        }
    }
]

# --- TOOL REGISTRY ---
tool_funcs = {
    "web_search": web_search,
    "calculator": calculator,
    "summarize": summarize,
    "get_time": get_time,
    "get_date": get_date,
    "extract_entities": extract_entities,
    "translate": translate,
    "wikipedia_search": wikipedia_search,
    "code_executor": code_executor,
    "sentiment_analysis": sentiment_analysis,
    "url_reader": url_reader
}

# --- MAIN AGENT LOOP ---
def agentic_function_calling_agent(goal: str, max_iters: int = 8, verbose: bool = True):
    messages = [
        {"role": "system", "content": "You are a helpful, tool-using agent. Use the available functions to solve the user's request. If you have enough information, provide the final answer."},
        {"role": "user", "content": goal}
    ]
    reasoning_steps = []
    for i in range(max_iters):
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
            functions=function_schemas,
            function_call="auto"
        )
        st.info(response)
        msg = response.choices[0].message

        step_info = f"**Step {i+1}:**\n"
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
            return msg.content, reasoning_steps
        reasoning_steps.append(step_info)
    return "[Agent did not complete the goal in time.]", reasoning_steps

if __name__ == "__main__":
    user_goal = input("Enter your complex goal: ")
    final_answer = agentic_function_calling_agent(user_goal)
    print("\n--- Final Agent Answer ---\n")
    print(final_answer)
