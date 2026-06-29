import unittest
from lumen.inference.tools import calculator_tool, web_search_tool

class TestAgentTools(unittest.TestCase):
    def test_calculator_tool(self):
        # Test basic addition
        self.assertEqual(calculator_tool("2 + 2"), "4")
        # Test multiplication
        self.assertEqual(calculator_tool("5 * 10"), "50")
        # Test complex expression
        self.assertEqual(calculator_tool("(10 + 5) * 2"), "30")
        # Test invalid expression
        self.assertTrue("Error" in calculator_tool("2 + / 2"))

    def test_web_search_tool(self):
        # Search for a known entity
        result = web_search_tool("Artificial intelligence")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 10)
        
        # Test empty or weird query
        result2 = web_search_tool("aklsjdfkajsdf")
        self.assertTrue("No results" in result2)

if __name__ == "__main__":
    unittest.main()
