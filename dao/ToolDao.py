from models.BaseCallLLM import BaseCallLLM
from dao.LLMCaller import llmCaller

import json
import re
import os
import pandas as pd

class ToolDao:
    def __init__(self) -> None:
        self.tools = []
        self.prompt_root = BaseCallLLM.prompt_root
        self.toolSystemPrompt = BaseCallLLM.toolSystemPrompt
        self.toolUserPrompt = BaseCallLLM.load_prompt_from_txt(BaseCallLLM.toolUserPromptPath)

        self.function_maps = {
            "calculate_mathematical_expression": BaseCallLLM.calculate_mathematical_expression
        }
        self.tools = self.function_maps.values()

    def tool_pipeline(self, inputs1, model_name, engine="hf", **kwargs):
        """
        输入模型参数，返回工具调用结果
        """
        dataDict = kwargs.get("dataDict", {})
        llmCaller.init_model(model_name)
        instruction = BaseCallLLM.build_first_prompt(
            system_prompt=self.toolSystemPrompt,
            user_prompt=self.toolUserPrompt.replace(f"{{maths}}", str(inputs1))
        )
        dataDict["ToolCall"] = {}
        dataDict["ToolCall"]["inputs"] = instruction
        response = self._caller(instruction, model_name, dataDict=kwargs.get("dataDict", {}))
        # print(instruction)
        # print(response)
        return response


    def _caller(self, instruction, model_name, engine="hf", **kwargs):
        dataDict = kwargs.get("dataDict", {})
        dataDict["ToolCall"] = dataDict.get("ToolCall", {})
        output = llmCaller.infer(instruction, model_name, tools=self.tools,engine=engine)
        # print(f"tool output is {output}")
        dataDict["ToolCall"]["toolOutput"] = output
        output_tool = self.parse_tool_calls(output)
        # print(output)

        if output_tool != []:
            for res_item in output_tool:

                function_name = self.function_maps.get(res_item.get("name", "-"), BaseCallLLM.calculate_mathematical_expression)
                # expression = res_item["arguments"]["expression"]
                try:
                    # print(res_item["arguments"]["expression"])
                    result = function_name(res_item["arguments"]["expression"])
                    instruction.append({"role": "assistant", "tool_calls": [{"type": "function", "function": res_item}]})
                    instruction.append({
                        "role": "tool",
                        "content": result
                    })
                except Exception as E:
                    
                    break
            # print(instruction)
            output = llmCaller.infer(instruction, model_name, engine=engine)

        dataDict["ToolCall"]["callerOutput"] = output
        return output

    def parse_tool_calls(self, model_output):
        """
        解析大模型输出中的多个 <tool_call> 并返回解析后的结果列表.

        Args:
            model_output (str): 大模型生成的包含多个 tool_call 的文本.

        Returns:
            list: 每个元素是解析后的 tool_call 字典.
        """
        # 正则表达式匹配 <tool_call> ... </tool_call>
        tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
        tool_calls = re.findall(tool_call_pattern, model_output, re.DOTALL)

        parsed_tool_calls = []
        
        # 解析每个 <tool_call> 中的 JSON 内容
        for tool_call in tool_calls:
            try:
                # 解析 JSON 数据
                tool_call_json = json.loads(tool_call.strip())
                parsed_tool_calls.append(tool_call_json)
            except json.JSONDecodeError as e:
                print(f"JSON decoding failed: {e}")
        
        return parsed_tool_calls

    def tablestring_to_headstring(self, tableDict):
        df = pd.DataFrame(tableDict['data'], columns=tableDict["columns"])

        df_head = tableDict





toolDao = ToolDao()