"""
Dynamic/conditional routing logic
"""
def default_routing_fn(current_agent, answer, workspace):
    # Example: linear pipeline
    pipeline = workspace.get('pipeline', [])
    idx = pipeline.index(current_agent)
    if idx + 1 < len(pipeline):
        return pipeline[idx + 1]
    return None
