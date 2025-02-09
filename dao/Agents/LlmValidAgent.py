from dao.Agents.BaseAgent import BaseAgent
from dao.LLMCaller import llmCaller
import json

class LlmValidAgent:
    def __init__(self):
        self.SystemPrompt="You are a professional format validator and fixer."
        self.UserPrompt=BaseAgent.load_prompt("./prompts/LlmValidPrompt.txt")

    def infer(self, model_name, inputs, outputFormat, engine="hf"):

        instruction=self.replace_prompt(
            inputs=inputs, 
            outputFormat=outputFormat
        )

        response = llmCaller.infer(instruction, model_name, engine=engine)

        return response

    def replace_prompt(self, inputs, outputFormat, **kwargs):
        """
        加载prompt并替换关键位置,{{Inputs}}
        """
        return BaseAgent.build_first_prompt(system_prompt=self.SystemPrompt,user_prompt=self.UserPrompt.replace("{{OutputFormat}}", outputFormat).replace("{{Inputs}}", json.dumps(inputs, indent=2, ensure_ascii=False)))
    
llmValidAgent = LlmValidAgent()