import unittest
from agentic_core.agent import Agent

class TestAgent(unittest.TestCase):
    def test_agent_init(self):
        agent = Agent(name="TestAgent", system_prompt="Test prompt", toolset=["summarize"])
        self.assertEqual(agent.name, "TestAgent")
        self.assertEqual(agent.system_prompt, "Test prompt")
        self.assertIn("summarize", agent.toolset)
    def test_agent_available_tools(self):
        agent = Agent(name="TestAgent", system_prompt="Test", toolset=["summarize"])
        tools = agent.available_tools()
        self.assertIsInstance(tools, list)

if __name__ == "__main__":
    unittest.main()
