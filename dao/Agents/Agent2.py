from dao.Agents.BaseAgent import BaseAgent
from dao.LLMCaller import llmCaller
from dao.Agents.LlmValidAgent import llmValidAgent
import json
import logging
import os
import pandas as pd
import traceback
class Agent2:
    def __init__(self):
        self.SystemPrompt="You are a table analyst."
        self.UserPrompt=BaseAgent.load_prompt("./prompts/Agent2PromptSimple.txt")
        self.UserPrompt2a=BaseAgent.load_prompt("./prompts/Agent2PromptSimple2a.txt")
        
        self.OutputFormat = "the key \"subTable\" mapped to a code block containing your Python code snippet.\n```json\n{\n\"subTable\": \"```python\ntable_df.loc[:,:]\n```\"\n}\n```"

        self.ReflectionSystem="You are tasked with reviewing and reflecting on a previously provided output to ensure its correctness."
        self.ReflectionPrompt=BaseAgent.load_prompt("./prompts/Agent2PromptSimpleWithReflection.txt")

    def infer_with_no_reflection(self, dataDict, model_name, engine="hf", **kwargs):
        logger = logging.getLogger(os.environ.get('logger_name'))
        dataDict["agent2"] = {}

        if isinstance(dataDict["table"], str):
            table_json = json.loads(dataDict["table"].strip())
        else:
            table_json =  dataDict["table"]
        instruction=self.replace_prompt_with_reflection(
            columns=table_json["columns"],
            data=table_json["data"],
            index=[i for i in range(len(table_json["data"]))]
        )
        dataDict["agent2"]["input1"] = instruction
        response = llmCaller.infer(instruction, model_name, engine=engine, dataMethod=kwargs["dataMethod"]).replace("\\n","\n")
        dataDict["agent2"]["output1"] = response.replace("\\n","\n")
        try:
            table_json_fix = BaseAgent.extract_json(response)
            table_json_fix["columns"]
            table_json_fix["data"]
            table_df = pd.DataFrame(table_json_fix['data'], columns=table_json_fix['columns'])
            logger.info(f"Agent2 解析的 JSON 表格结构无误")
        except Exception as E:
            table_json_fix = table_json
            logger.error("Agent2 解析错误 {error}， 没有 columns 和 data：\n{table}\n".format(error=E, table=response))
            
        return table_json_fix

    def infer_with_2a(self, dataDict, model_name, engine="hf", **kwargs):
        logger = logging.getLogger(os.environ.get('logger_name'))
        dataDict["agent2"] = {}
        if isinstance(dataDict["table"], str):
            table_json = json.loads(dataDict["table"].strip())
        else:
            table_json = dataDict["table"]
        instruction=self.replace_prompt_with_reflection_2a(
            columns=table_json["columns"],
            data=table_json["data"],
            index=[i for i in range(len(table_json["data"]))]
        )
        dataDict["agent2"]["input1"] = instruction
        response = llmCaller.infer(instruction, model_name, engine=engine, dataMethod=kwargs["dataMethod"], agent_mode=kwargs.get("agent_mode", None)).replace("\\n","\n")
        dataDict["agent2"]["output1"] = response.replace("\\n","\n")
        try:
            table_json_fix = BaseAgent.extract_json(response)
            table_json_fix["columns"]
            table_json_fix["data"]
            table_df = pd.DataFrame(table_json_fix['data'], columns=table_json_fix['columns'])
        except Exception as E:
            table_json_fix = table_json
            logger.error(E)
            logger.error("Agent2 的JSON 解析的报错内容为：\n"+response)

    def infer_with_reflection(self, dataDict, model_name, engine="hf", **kwargs):
        logger = logging.getLogger(os.environ.get('logger_name'))
        dataDict["agent2"] = {}
        if isinstance(dataDict["table"], str):
            table_json = json.loads(dataDict["table"].strip())
        else:
            table_json = dataDict["table"]
        instruction=self.replace_prompt_with_reflection(
            columns=table_json["columns"],
            data=table_json["data"],
            index=[i for i in range(len(table_json["data"]))]
        )
        dataDict["agent2"]["input1"] = instruction
        response = llmCaller.infer(instruction, model_name, engine=engine, dataMethod=kwargs["dataMethod"], agent_mode=kwargs.get("agent_mode", None)).replace("\\n","\n")
        dataDict["agent2"]["output1"] = response.replace("\\n","\n")
        try:
            table_json_fix = BaseAgent.extract_json(response)
            table_json_fix["columns"]
            table_json_fix["data"]
            table_df = pd.DataFrame(table_json_fix['data'], columns=table_json_fix['columns'])
        except Exception as E:
            table_json_fix = table_json
            logger.error(E)
            logger.error("Agent2 的JSON 解析的报错内容为：\n"+response)

            ## ===
            # tb = traceback.extract_tb(E.__traceback__)
            # last_call = tb[-1]  # 获取最后一次调用堆栈信息

            # # 获取代码所在行的内容
            # filename = last_call.filename
            # lineno = last_call.lineno
            # with open(filename, "r") as file:
            #     lines = file.readlines()
            #     code_line = lines[lineno - 1].strip()
            # ## ===
            # error_message = f"Error message: My Python code exec `{code_line}` and raise the exception `{E}`."
            # errors = f"Your output format of JSON is incorrect, check every symbol, such as '\"', ',' and so on.\n{error_message}\nPlease correct the JSON format and provide an updated response."
            # logger.error(errors)
            # instruction.append({
            #     "assistant": response,
            #     "user": errors
            # })
            # logger.error("重新推理{id}\n\n{response}".format(id=dataDict["id"],response=response))
            # response = llmCaller.infer(instruction, model_name, engine=engine).replace("\\n","\n")
            # logger.error(f"重新推理结果")
            # try:
            #     table_json_fix = BaseAgent.extract_json(response)
            #     table_json_fix["columns"]
            #     table_json_fix["data"]
            #     table_df = pd.DataFrame(table_json_fix['data'], columns=table_json_fix['columns'])
            #     logger.error(f"重新推理后解析正确：{table_json_fix}")
            # except:
            #     logger.error(f"重新推理后解析错误：{response}")
            #     table_json_fix = table_json

        ## archive reflection
        reflectionInstruction = self.replace_prompt_reflection(
            columns=table_json_fix["columns"],
            data=table_json_fix["data"]
        )
        dataDict["agent2"]["input2"] = reflectionInstruction

        response = llmCaller.infer(reflectionInstruction, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
        dataDict["agent2"]["output2"] = response
        try:
            ref = self.parse_reflection(response, dataDict)
            # If reflection is valid, set it to the question or parsed sub-queries
            if ref is True:
                ref = table_json_fix

            dataDict["agent2"]["parseError"] = 0
        except Exception as e:
            # Handle any exceptions, fallback to default reflection
            logger.error("Agent2 “反思” 错误 {error}：\n{table}\n".format(error=e, table=response))
            ref = table_json
            dataDict["agent2"]["parseError"] = 1
            
        return table_json_fix
        

    def parse_reflection(self, response, data_dict={}):
        ex_response = BaseAgent.extract_reflection(response)
        
        data_dict["agent2"]["Agent2Reflection"] = ex_response["isCorrect"]
        return True if ex_response["isCorrect"] else {
            "columns": ex_response["correctedOutput"]["columns"],
            "data": ex_response["correctedOutput"]["data"]
        }

    def infer(self, dataDict, model_name, engine="hf", agent1_query=None):
        datas = json.loads(dataDict["table"])
        instruction=self.replace_prompt(
            columns=datas["columns"],
            data=datas["data"][1:4],
            query=agent1_query["Query"],
            QueryHint=agent1_query["QueryHint"]
        )
        response = llmCaller.infer(instruction, model_name, engine=engine)
        # print(BaseAgent.extract_json(response))
        PythonCode = response.strip().strip("```python").strip("```").strip()

        return PythonCode

    def replace_prompt_with_reflection_2a(self,**inputs):
        return BaseAgent.build_first_prompt(system_prompt=self.SystemPrompt,user_prompt=self.UserPrompt2a.replace("{{Inputs}}", json.dumps(inputs, indent=2, ensure_ascii=False)))
    def replace_prompt_with_reflection(self,**inputs):
        return BaseAgent.build_first_prompt(system_prompt=self.SystemPrompt,user_prompt=self.UserPrompt.replace("{{Inputs}}", json.dumps(inputs, indent=2, ensure_ascii=False)))

    def replace_prompt_reflection(self,**inputs):
        return BaseAgent.build_first_prompt(system_prompt=self.ReflectionSystem,user_prompt=self.ReflectionPrompt.replace("{{Inputs}}", json.dumps(inputs, indent=2, ensure_ascii=False)))


    def replace_prompt(self, columns, data, query, **kwargs):
        """
        加载prompt并替换关键位置,{{Inputs}}
        """
        inputs = {
            "ColumnNames": columns,
            "SampleContents": data,
            "Query": query
        }
        inputs.update(kwargs)
        return BaseAgent.build_first_prompt(system_prompt=self.SystemPrompt,user_prompt=self.UserPrompt.replace("{{Inputs}}", json.dumps(inputs, indent=2, ensure_ascii=False)))
    
agent2 = Agent2()