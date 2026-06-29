import json
import re
from typing import List, Dict, Any, Optional, Callable

from .tools import AVAILABLE_TOOLS, get_tool_descriptions

class Agent:
    def __init__(self, model_engine, system_prompt: Optional[str] = None):
        """
        Initializes the agent with a Lumen-1 inference engine.
        model_engine: An instance of the Lumen-1 model used to generate text.
        """
        self.engine = model_engine
        
        tool_rules = f"""
You have access to the following tools:
{get_tool_descriptions()}

To use a tool, you MUST output exactly in this format:
<tool_call>{{"name": "tool_name", "arguments": {{"arg_name": "arg_value"}}}}</tool_call>

If you use a tool, wait for the result. Do not answer until you see the observation.
If you know the answer, just output it directly in the appropriate language.
"""
        
        if system_prompt:
            self.system_prompt = system_prompt.strip() + "\n\n" + tool_rules
        else:
            self.system_prompt = f"""You are Lumen-1, an advanced AI Agent proudly "Made in Sri Lanka". 
You are a true Sri Lankan and a master (expert) of the Sinhala language. You speak Sinhala flawlessly, naturally, and with profound expertise.
While your heart is Sri Lankan and your primary identity is tied to Sinhala, you are fully capable of understanding, translating, chatting, and processing Audio/Voice in ALL languages of the world.

When asked who you are, always introduce yourself as an AI created in Sri Lanka.
When speaking Sinhala, use rich, natural, and expert-level vocabulary.
""" + "\n\n" + tool_rules

    def parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parses the LLM output to find any tool calls."""
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, text, re.DOTALL)
        
        calls = []
        for match in matches:
            try:
                call_dict = json.loads(match)
                calls.append(call_dict)
            except json.JSONDecodeError:
                pass
        return calls

    def run(self, user_input: str, max_iterations: int = 3, callback: Optional[Callable[[str], None]] = None) -> str:
        """
        Runs the agentic loop: Generate -> Parse Tool -> Execute Tool -> Repeat
        """
        history = [{"role": "system", "content": self.system_prompt},
                   {"role": "user", "content": user_input}]
        
        for _ in range(max_iterations):
            # 1. Generate text using the Lumen engine
            # (Assuming self.engine.generate takes a message history and returns a string)
            prompt = self._format_history(history)
            
            if callback:
                callback("Thinking...")
                
            response_text = self.engine.generate(prompt)
            
            history.append({"role": "assistant", "content": response_text})
            
            # 2. Check for tool calls
            tool_calls = self.parse_tool_calls(response_text)
            
            if not tool_calls:
                # No tool called, we are done
                return response_text
                
            # 3. Execute tools
            for call in tool_calls:
                tool_name = call.get("name")
                args = call.get("arguments", {})
                
                if tool_name in AVAILABLE_TOOLS:
                    print(f"\n[Agent executing tool: {tool_name} with args: {args}]")
                    if callback:
                        callback(f"Executing tool '{tool_name}' with args: {json.dumps(args)}")
                        
                    # Dynamically call the tool function
                    try:
                        result = AVAILABLE_TOOLS[tool_name](**args)
                    except Exception as e:
                        result = str(e)
                    
                    print(f"[Tool Result]: {result[:100]}...\n")
                    if callback:
                        callback(f"Tool '{tool_name}' returned: {result[:100]}...")
                        
                    history.append({"role": "observation", "content": result})
                else:
                    err_msg = f"Tool {tool_name} not found."
                    if callback:
                        callback(err_msg)
                    history.append({"role": "observation", "content": err_msg})

        return "Agent reached maximum iterations without concluding."

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Simple formatter for turning messages into a flat string prompt."""
        formatted = ""
        for msg in history:
            role = msg["role"].upper()
            formatted += f"[{role}]\n{msg['content']}\n\n"
        formatted += "[ASSISTANT]\n"
        return formatted
