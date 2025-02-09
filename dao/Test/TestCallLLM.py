from models.BaseCallLLM import BaseCallLLM
from dao.LLMCaller import llmCaller
from dao.ToolDao import toolDao
from dao.TableBench.QueryReviseCaller import queryReviseCaller
from dao.TableCleanDao import tableCleanDao
from dao.PotDao import potDao
from dao.Agents.Agent1 import agent1
from dao.Agents.Agent2 import agent2
from dao.Agents.Agent3 import agent3
from dao.Agents.Agent4 import agent4
from dao.BasePrompt import BasePrompt

import logging
import os
import json
import pandas as pd

from tools import *
import ast
import random
import re, json
from loguru import logger

numbers = 1

class TestCallLLM(BaseCallLLM):
    def __init__(self):
        super().__init__()
        self.DatasetName = "Test"
        self.RawDataRoot = "./dao/Test"
        self.RawDataPath = {
            "PoT": os.path.join(self.RawDataRoot, "ModifiedRawQTabData.json")
        }

        self._logger = logging.getLogger(os.environ.get('logger_name'))



    def inference(self, model_name, to_path, head=None, engine="hf",**kwargs):
        response_result = []
        number = 0

        for dataType, dataMap in self.RawDataPath.items():
            toData_id = {}
            ### 固定 dataType
            dataType = kwargs["tablebenchMode"]
            if os.path.exists(to_path):
                with open(to_path, 'r', encoding='utf-8') as file:
                    toData = json.load(file)
                toData_id = {i["id"]:i for i in toData}
            with open(dataMap, 'r', encoding='utf-8') as file:
                for LineData in json.load(file):
                    if LineData["id"] not in toData_id.keys():
                        pass
                    elif LineData["id"] in toData_id.keys():
                        response_result.append(toData_id[LineData["id"]])
                        continue
                    else:
                        continue
                    # print(LineData["qtype"])
                    response = self._build_pipeline(LineData, model_name, engine=engine, dataMethod=dataType, agent_mode=kwargs["agent_mode"])

                    response_result.append(LineData)
                    with open(to_path, 'w', encoding='utf-8') as f:
                        json.dump(response_result, f, ensure_ascii=False, indent=4)
    def dataMaps(self, dataDict):
        queries = dataDict["GeneratedQuery"]["Queries"]

        # Generate a random integer within the range of query indices
        random_index = random.randint(0, len(queries) - 1)

        # Select the query based on the random index
        questionString = queries[0]
        dataDict["question"] = questionString
    def _build_pipeline(self, dataDict, model_name, tools=None, engine="hf", head=None, **kwargs):
        self._logger = logging.getLogger(os.environ.get('logger_name'))

        global numbers

        self.dataMaps(dataDict)
        agent_mode = kwargs["agent_mode"]
        if agent_mode == "1+2+3":
            agent1_json = agent1.infer_with_reflection(dataDict, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
            agent2_json = agent2.infer_with_reflection(dataDict, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
            agent3_answer = agent3.infer(dataDict, model_name, engine=engine, queries=agent1_json, table=agent2_json, dataMethod=kwargs["dataMethod"])
        elif agent_mode == "1_no_ref+2_no_ref+3":
            agent1_json = agent1.infer_with_no_reflection(dataDict, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
            agent2_json = agent2.infer_with_no_reflection(dataDict, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
            agent3_answer = agent3.infer(dataDict, model_name, engine=engine, queries=agent1_json, table=agent2_json, dataMethod=kwargs["dataMethod"])
        elif agent_mode == "1_no_ref+3":
            agent1_json = agent1.infer_with_no_reflection(dataDict, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
            agent3_answer = agent3.infer(dataDict, model_name, engine=engine, queries=agent1_json, table=json.loads(dataDict["table"]), dataMethod=kwargs["dataMethod"])
        elif agent_mode == "2_no_ref+3":
            agent2_json = agent2.infer_with_no_reflection(dataDict, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
            agent3_answer = agent3.infer(dataDict, model_name, engine=engine, queries=[dataDict["question"]], table=agent2_json, dataMethod=kwargs["dataMethod"])
        elif agent_mode == "1+3":
            agent1_json = agent1.infer_with_reflection(dataDict, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
            agent3_answer = agent3.infer(dataDict, model_name, engine=engine, queries=agent1_json, table=json.loads(dataDict["table"]), dataMethod=kwargs["dataMethod"])
        elif agent_mode == "2+3":
            agent2_json = agent2.infer_with_reflection(dataDict, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
            agent3_answer = agent3.infer(dataDict, model_name, engine=engine, queries=[dataDict["question"]], table=agent2_json, dataMethod=kwargs["dataMethod"])
        elif agent_mode == "3":
            agent3_answer = agent3.infer(dataDict, model_name, engine=engine, queries=[dataDict["question"]], table=json.loads(dataDict["table"]), dataMethod=kwargs["dataMethod"])
        elif agent_mode == "raw":
            agent3_answer = self._raw_infer(dataDict, model_name=model_name, engine=engine, dataMethod=kwargs["dataMethod"])
        else:
            raise ValueError(f"{agent_mode} is not a valid mode, please check it")
        dataDict["ExtractResult"] = agent3_answer
        self._extract_answer(dataDict)

    def tableGen(self, data, columns):
        return pd.DataFrame(data=data,columns=columns).to_json(orient="split")
    def extract_final_annswer(self, text):
        extractAnswer = re.findall("Final Answer:(.*)", text, re.IGNORECASE)
        if extractAnswer == []:
            return ""
        else:
            answer = extractAnswer[-1].strip()
            return answer

    def _raw_infer(self, dataDict, model_name, engine, **kwargs):
        # print(dataDict.keys())
        tableString = self.tableGen(data=dataDict["table"]["data"],columns=dataDict["table"]["columns"])
        # Get the list of queries
        queries = dataDict["GeneratedQuery"]["Queries"]

        # Generate a random integer within the range of query indices
        random_index = random.randint(0, len(queries) - 1)

        # Select the query based on the random index
        questionString = queries[random_index]
        tableDict = dataDict["table"]
        tabledf = pd.DataFrame(tableDict["data"], columns=tableDict["columns"])
        if kwargs["dataMethod"] == "PoT":
            try:
                instruction = [
                    {"role": "system", "content": BasePrompt.SystemPrompt},
                    {"role": "user", "content": BasePrompt.POTAssistantPrompt.format(tableString=tableString, questionString=questionString)}
                ]
                dataDict["rawInstruction"] = instruction
                response = llmCaller.infer(instruction, model_name, engine=engine, tabledf=tabledf, question=dataDict["question"],dataMethod=kwargs["dataMethod"])
                dataDict["RawInfer"] = response
                
                python_code = self.extract_python(response)

                df_str = "df="+json.dumps(dataDict["table"],ensure_ascii=False).strip()+"\ndf = pd.DataFrame(df['data'], columns=df['columns'])\n"
                codes = python_code.replace("df = pd.read_csv('table.csv')",df_str).strip()
                print(codes)
                response = self.exec_code(codes.strip()).strip()
                logger.info(f"回答正确：{response}")
                # response = llmCaller.infer(instruction, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
                # print(response)
                dataDict["RawInfer"] = response
            except Exception as e:
                logger.error(f"错误内容为:{e}")
                logger.error("回答错误：{response}".format(response=dataDict["id"]))
                response = "-"
        else:
            basePromptMap = {
                "PoT": BasePrompt.POTAssistantPrompt,
                "DP": BasePrompt.DPAssistantPrompt,
                "SCoT": BasePrompt.SCoTAssistantPrompt,
                "TCoT": BasePrompt.TCoTAssistantPrompt
            }
            instruction = [
                {"role": "system", "content": BasePrompt.SystemPrompt},
                {"role": "user", "content": basePromptMap[kwargs["dataMethod"]].format(tableString=tableString, questionString=questionString)}
            ]
            dataDict["rawInstruction"] = instruction
            response = llmCaller.infer(instruction, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
            dataDict["RawInfer"] = response
            extractAnswer = self.extract_final_annswer(response)
            response = extractAnswer
        return response
    
    def exec_code(self, code):
        import io
        import sys
        output_buffer = io.StringIO()

        # Save the current stdout so we can restore it later
        original_stdout = sys.stdout

        # Redirect stdout to the buffer
        sys.stdout = output_buffer

        # Execute the code
        exec(code)

        # Get the captured output
        captured_output = output_buffer.getvalue()

        # Restore original stdout
        sys.stdout = original_stdout

        # Print the captured output
        return captured_output.replace("\n", " ")

    def _extract_answer(self, dataDict):
        dataDict["answer"] = ", ".join([i["Answer"] for i in dataDict["GeneratedQuery"]["SubQueries"]])
        dataDict["RawAnswer"] = dataDict["answer"]
        dataDict["answer"] = dataDict["answer"].replace(",", " ").replace("，", " ").replace("  "," ").strip("%$.").lower()
        dataDict["FinalResult"] = ""
        tmpresult = []
        for rawanswer in dataDict["answer"].split(" "):
            if rawanswer.lower() in dataDict["ExtractResult"].lower():
                tmpresult.append(rawanswer.lower().strip())
        dataDict["FinalResult"] = " ".join(tmpresult)
        if dataDict["FinalResult"] == "":
            dataDict["FinalResult"] = dataDict["ExtractResult"].lower()
        if dataDict["FinalResult"] == dataDict["answer"]:
            dataDict["score"] = 1
        else:
            dataDict["score"] = 0

testCallLLM = TestCallLLM()