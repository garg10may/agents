"""
Code execution tool
"""
def code_executor(code: str) -> str:
    try:
        exec_globals = {}
        exec(code, exec_globals)
        return str(exec_globals)
    except Exception as e:
        return f"[Code execution error: {e}]"
