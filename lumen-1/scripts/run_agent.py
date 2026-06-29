import sys
import os

# Add the project root to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lumen.inference.agent import Agent

class MockEngine:
    """
    A mock engine for demonstration. 
    In production, this would be replaced by the actual Lumen-1 Transformer Engine.
    """
    def generate(self, prompt: str) -> str:
        # Simple mock logic for demonstration
        if "Sri Lanka" in prompt and "search" not in prompt:
            return '<tool_call>{"name": "search", "arguments": {"query": "Sri Lanka"}}</tool_call>'
        elif "calculate" in prompt.lower() or "2 + 2" in prompt:
            return '<tool_call>{"name": "calculator", "arguments": {"expression": "2 + 2"}}</tool_call>'
        
        # If it sees an observation in the prompt, it answers
        if "[OBSERVATION]" in prompt:
            return "Based on the tool result, I have found the answer to your request."
            
        return "Hello! I am Lumen-1. I can search the web and perform calculations. How can I help you?"

def main():
    print("Initializing Lumen-1 Advanced Agent...")
    
    # Load your actual model here
    # engine = load_lumen_model("checkpoints/final/model.pt")
    engine = MockEngine() 
    
    agent = Agent(model_engine=engine)
    
    print("\nAgent is ready! (Type 'exit' to quit)")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            response = agent.run(user_input)
            print(f"\nLumen-1: {response}")
            
        except KeyboardInterrupt:
            break
            
    print("\nGoodbye!")

if __name__ == "__main__":
    main()
