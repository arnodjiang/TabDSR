from models.BaseCallLLM import BaseCallLLM
from dao.LLMCaller import llmCaller
from dao.ToolDao import toolDao
from dao.TableBench.QueryReviseCaller import queryReviseCaller
from dao.TableCleanDao import tableCleanDao
from dao.PotDao import potDao
from dao.Agents.Agent1 import agent1
from dao.Agents.Agent2 import agent2
from dao.Agents.Agent3 import agent3


from dao.Agents.CottableAgent import cotableAgent

from loguru import logger
import logging
import os
import json
import pandas as pd

from tools import *
import ast
import re
from .Converter import TestConverter
from dao.BasePrompt import BasePrompt
import traceback

ConverterMap = {
    "Test": TestConverter
}

numbers = 1
# _logger = logging.getLogger("logger_name")

class TableBenchCallLLM(BaseCallLLM):
    def __init__(self):
        super().__init__()
        self.DatasetName = "TableBench"
        self.RawDataRoot = "./EvaluationCode/EvaluationCode/TableBench/RawData"

        self._infer_qtype = ["NumericalReasoning"]

        self._logger = logging.getLogger(os.environ.get('logger_name'))
        # /raid/share/jiangchangjiang/tablellmPipeline/results/TableBench/qwen2.5/32b/Agents
    def build_first_prompt(self, system_prompt, user_prompt, tool_prompt=None):
        """
        构建多层嵌套 prompt
        """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]


    def inference(self, model_name, to_path, head=None, engine="hf", **kwargs):
        response_result = []
        number = 0

        self._infer_fileType = [kwargs["tablebenchMode"]]
        logger.info(kwargs["dataset_name"])
        if kwargs["dataset_name"] == "TatQa":
            self._infer_qtype = []
            self.RawDataPath = {
                "PoT": "./EvaluationCode/EvaluationCode/TatQa/filtered_TatQa_test_gold.jsonl"
            }
        elif kwargs["dataset_name"] == "CalTab151":
            self._infer_qtype = []
            self.RawDataPath = {
                "PoT": "./EvaluationCode/EvaluationCode/CalTab151/ModifiedRawQTabData.jsonl"
            }
        elif kwargs["dataset_name"] == "TableBench":
            self._infer_qtype = []
            self.RawDataPath = {
                "PoT": "./EvaluationCode/EvaluationCode/TableBench/RawData/TableBenchData.jsonl"
            }

        for dataType, dataMap in self.RawDataPath.items():
            # print(dataType)
            toData_id = {}

            if os.path.exists(to_path):
                with open(to_path, 'r', encoding='utf-8') as file:
                    toData = json.load(file)
                toData_id = {i["id"]:i for i in toData}
            with open(dataMap, 'r', encoding='utf-8') as file:
                for steps, line in enumerate(file):
                    LineData = json.loads(line.strip())
                    logger.info(LineData)
                    if self._infer_qtype == [] and LineData["id"] not in toData_id.keys():
                        
                        pass
                    elif LineData["id"] in toData_id.keys():
                        response_result.append(toData_id[LineData["id"]])
                        continue
                    else:
                        continue

                    if kwargs["dataset_name"] in ["TableBench","CalTab151", "TatQa"]:
                        # logger.info(LineData)
                        LineData["table"] = LineData["Table"]
                        LineData["question"] = LineData["Question"]
                        LineData["answer"] = LineData["Answer"]

                    response = self._build_pipeline(LineData, model_name, tools=[calculate_mathematical_expression], engine=engine, dataMethod=kwargs["tablebenchMode"], agent_mode=kwargs["agent_mode"], dataset_name=kwargs["dataset_name"])
                    LineData.pop("table")
                    LineData.pop("question")
                    LineData.pop("answer")
                    response_result.append(LineData)
                    # if steps==5:
                    with open(to_path, 'w', encoding='utf-8') as f:
                        json.dump(response_result, f, ensure_ascii=False,indent=2)
            with open(to_path, 'w', encoding='utf-8') as f:
                json.dump(response_result, f, ensure_ascii=False,indent=2)

    def _build_pipeline(self, dataDict, model_name, tools=None, engine="hf", head=None, **kwargs):
        
        global numbers
        agent_mode = kwargs["agent_mode"]
        table_dict = json.loads(dataDict["table"]) if isinstance(dataDict["table"], str) else dataDict["table"]
        if agent_mode == "2+rawLLM" and model_name in ["TableGPT2-7B","tablellama:7b","RUCKBReasoning/TableLLM-13b","RUCKBReasoning/TableLLM-7b","TIGER-Lab/StructLM-7B", "microsoft/tapex-large-finetuned-wtq","google/tapas-large-finetuned-wtq","neulab/omnitab-large-finetuned-wtq"]:
            agent2_json = agent2.infer_with_reflection(dataDict, "qwen2.5", engine="hf", dataMethod=kwargs["dataMethod"], agent_mode=agent_mode)
            dataDict["table"] = agent2_json
            try:
                agent3_answer = agent3.raw_rawLLM(dataDict, model_name=model_name, engine=engine, dataMethod=kwargs["dataMethod"], question=dataDict["question"], agent_mode=agent_mode)
            except Exception as e:
                agent3_answer = "-"
        elif agent_mode == "1+2a+3a":
            agent1_json = agent1.infer_with_reflection(dataDict, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
            agent2_json = agent2.infer_with_2a(dataDict, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
            agent3_answer = agent3.infer_with_3a(dataDict, model_name, engine=engine, queries=[dataDict["question"]], table=table_dict, dataMethod=kwargs["dataMethod"])
        elif agent_mode == "1+2+3":
            agent1_json = agent1.infer_with_reflection(dataDict, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
            agent2_json = agent2.infer_with_reflection(dataDict, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
            agent3_answer = agent3.infer(dataDict, model_name, engine=engine, queries=agent1_json, table=agent2_json, dataMethod=kwargs["dataMethod"])
        elif agent_mode == "1_no_ref+2_no_ref+3":
            agent1_json = agent1.infer_with_no_reflection(dataDict, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
            agent2_json = agent2.infer_with_no_reflection(dataDict, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
            agent3_answer = agent3.infer(dataDict, model_name, engine=engine, queries=agent1_json, table=agent2_json, dataMethod=kwargs["dataMethod"])
        elif agent_mode == "1_no_ref+3":
            agent1_json = agent1.infer_with_no_reflection(dataDict, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
            agent3_answer = agent3.infer(dataDict, model_name, engine=engine, queries=agent1_json, table=table_dict, dataMethod=kwargs["dataMethod"])
        elif agent_mode == "2_no_ref+3":
            agent2_json = agent2.infer_with_no_reflection(dataDict, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
            agent3_answer = agent3.infer(dataDict, model_name, engine=engine, queries=[dataDict["question"]], table=agent2_json, dataMethod=kwargs["dataMethod"])
        elif agent_mode == "1+3":
            agent1_json = agent1.infer_with_reflection(dataDict, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
            agent3_answer = agent3.infer(dataDict, model_name, engine=engine, queries=agent1_json, table=table_dict, dataMethod=kwargs["dataMethod"])
        elif agent_mode == "2+3":
            agent2_json = agent2.infer_with_reflection(dataDict, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
            agent3_answer = agent3.infer(dataDict, model_name, engine=engine, queries=[dataDict["question"]], table=agent2_json, dataMethod=kwargs["dataMethod"])
        elif agent_mode == "3":
            agent3_answer = agent3.infer(dataDict, model_name, engine=engine, queries=[dataDict["question"]], table=table_dict, dataMethod=kwargs["dataMethod"])
        elif agent_mode == "raw":
            agent3_answer = self._raw_infer(dataDict, model_name=model_name, engine=engine, dataMethod=kwargs["dataMethod"])
        elif agent_mode == "1_query+3":
            agent1_json = agent1.infer_with_QueryDecomposition(dataDict, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
            agent3_answer = agent3.infer_with_QueryDecomposition(dataDict, model_name, engine=engine, query=dataDict["question"], table=table_dict, dataMethod=kwargs["dataMethod"], QuerySteps=agent1_json)
        elif agent_mode == "1_query+2+3":
            agent1_json = agent1.infer_with_QueryDecomposition(dataDict, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
            agent2_json = agent2.infer_with_reflection(dataDict, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
            agent3_answer = agent3.infer_with_QueryDecomposition(dataDict, model_name, engine=engine, query=dataDict["question"], table=agent2_json, dataMethod=kwargs["dataMethod"], QuerySteps=agent1_json)
        elif agent_mode == "TabSQLify":
            agent3_answer = None
        elif agent_mode == "cotable":
            agent3_answer = cotableAgent.infer(dataDict, model_name=model_name, engine=engine, dataMethod=kwargs["dataMethod"])
        elif agent_mode == "rawLLM" and model_name in ["TableGPT2-7B","tablellama:7b","RUCKBReasoning/TableLLM-13b","RUCKBReasoning/TableLLM-7b","TIGER-Lab/StructLM-7B", "microsoft/tapex-large-finetuned-wtq","google/tapas-large-finetuned-wtq","neulab/omnitab-large-finetuned-wtq"]:
            try:
                agent3_answer = agent3.raw_rawLLM(dataDict, model_name=model_name, engine=engine, dataMethod=kwargs["dataMethod"], question=dataDict["question"], dataset_name=kwargs["dataset_name"])
            except Exception as e:
                agent3_answer = "-"
        else:
            raise ValueError(f"{agent_mode} is not a valid mode, please check it")

        dataDict["ExtractResult"] = agent3_answer
        # self._extract_answer(dataDict)
        return None

    def _raw_infer(self, dataDict, model_name, engine, **kwargs):

        questionString = dataDict["question"]
        tableString = dataDict["table"] if isinstance(dataDict["table"], str) else json.dumps(dataDict["table"], ensure_ascii=False)
        if isinstance(dataDict["table"], str):
            tableDict = json.loads(dataDict["table"])
        else:
            tableDict = dataDict["table"]
        tabledf = pd.DataFrame(tableDict["data"], columns=tableDict["columns"])
        TableTitle = dataDict.get("TableTitle")
        TableTitle = f"\nTable Title:\n{TableTitle}\n" if TableTitle else None

        if kwargs["dataMethod"] in ["PoT"]:
            try:
                rawInstturiocn = BasePrompt.POTAssistantPrompt.format(tableString=tableString, questionString=questionString)+TableTitle if TableTitle else BasePrompt.POTAssistantPrompt.format(tableString=tableString, questionString=questionString)
                instruction = [
                    {"role": "system", "content": BasePrompt.SystemPrompt},
                    {"role": "user", "content": rawInstturiocn}
                ]
                
                dataDict["rawInstruction"] = instruction
                response = llmCaller.infer(instruction, model_name, engine=engine, tabledf=tabledf, question=dataDict["question"],dataMethod=kwargs["dataMethod"])
                dataDict["RawInfer"] = response
                
                python_code = self.extract_python(response)

                df_str = f"df={json.dumps(tableDict, ensure_ascii=False,indent=2)}"+"\ndf = pd.DataFrame(df['data'], columns=df['columns'])\n"
                df_str = df_str.replace("null","None")
                codes = python_code.replace("df = pd.read_csv('table.csv')",df_str).strip().replace("\\n","\n")
                dataDict["RawCodes"] = codes
                response = self.exec_code(codes.strip()).strip()

                dataDict["RawInfer"] = response
            except Exception as e:

                response = "-"
                dataDict["ErrorMessage"] = response
        elif kwargs["dataMethod"] in ["DP","TCoT", "SCoT"]:
            tableString = self.tableGen(data=tableDict["data"],columns=tableDict["columns"])
            basePromptMap = {
                "PoT": BasePrompt.POTAssistantPrompt,
                "DP": BasePrompt.DPAssistantPrompt,
                "SCoT": BasePrompt.SCoTAssistantPrompt,
                "TCoT": BasePrompt.TCoTAssistantPrompt
            }
            if TableTitle is None:
                instruction = [
                    {"role": "system", "content": BasePrompt.SystemPrompt},
                    {"role": "user", "content": basePromptMap[kwargs["dataMethod"]].format(tableString=tableString, questionString=questionString)}
                ]
            else:
                instruction = [
                    {"role": "system", "content": BasePrompt.SystemPrompt},
                    {"role": "user", "content": basePromptMap[kwargs["dataMethod"]].format(tableString=tableString, questionString=questionString)+TableTitle}
                ]
            dataDict["rawInstruction"] = instruction
            response = llmCaller.infer(instruction, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
            dataDict["RawInfer"] = response
            extractAnswer = self.extract_final_annswer(response)
            response = extractAnswer
        return response

    def tableGen(self, data, columns):
        return pd.DataFrame(data=data,columns=columns).to_json(orient="split")

    def extract_final_annswer(self, text):
        extractAnswer = re.findall("Final Answer:(.*)", text, re.IGNORECASE)
        if extractAnswer == []:
            return ""
        else:
            answer = extractAnswer[-1].strip()
            return answer

    def _extract_answer(self, dataDict):
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

    def _Pipline_POT(self, dataDict, model_name, tableString=None, engine="hf", head=None):
        # result = potDao.pipeline(dataDict, model_name, tableString, engine=engine, head=head)
        result = potDao.pipeline_only_one_round(dataDict, model_name, tableString, engine=engine, head=head)

    def _Pipline_QueryRevisePrompt(self, dataDict, model_name, tableString=None, engine="hf"):
        revise_query = queryReviseCaller.pipeline(dataDict, model_name, engine=engine)
        
        ## 1 Raw COT
        # response = self._Pipline_RawCOTPrompt(dataDict, model_name, query=revise_query)

        ## 2 Tool
        response = self._Pipline_ToolPrompt(dataDict, model_name, query=revise_query)

        return response

    def _pipline_RawPOTPrompt(self, dataDict, model_name, tableString=None, query=None, engine="hf", TFormat="JSON"):
        tableString = dataDict["table"] if tableString is None else tableString
        query = dataDict["question"] if query is None else query

        dataDict["RawPrompt"] = {}
        instruction = self.build_first_prompt(
            system_prompt=self.RawPOTSystemPrompt,
            user_prompt=self.RawPOTPrompt.format(TableString=tableString, QuestionString=query)
        )
        output = llmCaller.infer(instruction, model_name, tools=None,engine=engine)
        dataDict["RawPrompt"]["inputs"] = instruction
        dataDict["RawPrompt"]["outputs"] = output

        result_dict = self.exec_python_code(output, tableString)

        if result_dict["ExtractResult"]:
            dataDict["code"] = result_dict["code"]
            dataDict["ExtractResult"] = result_dict["ExtractResult"]
        else:
            dataDict["ExtractResult"] = ""

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
        return output
    def inentify_python_code(self, result):
        matcch_result = re.search(r"```python(.*?)```", result, re.IGNORECASE | re.DOTALL)
        if matcch_result is None:
            return matcch_result.strip()
        else:
            return matcch_result.group(1)

    def exec_python_code(self, response, tablestring):
        """
        从大模型回复中抽取代码
        """
        codes = self.inentify_python_code(response)
        df_str = "df="+tablestring.strip()+"\n"+"\ndf = pd.DataFrame(df['data'], columns=df['columns'])\n"
        codes = codes.replace("df = pd.read_csv('table.csv')",df_str).strip()

        try:
            exec_result = self.exec_code(codes).strip()
        except:
            exec_result = None


        return {
            "code":codes,
            "ExtractResult": exec_result
        }
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
    def _Pipline_RawCOTPrompt(self, dataDict, model_name, tableString=None, query=None, engine="hf", TFormat="JSON"):

        tableString = dataDict["table"] if tableString is None else tableString
        query = dataDict["question"] if query is None else query
        dataDict["RawPrompt"] = {}

        instruction = self.build_first_prompt(
            system_prompt=self.RawSystemPrompt,
            user_prompt=self.RawTCOTPrompt.format(TableString=tableString, QuestionString=query, TFormat=TFormat)
        )
        output = llmCaller.infer(instruction, model_name, tools=None,engine=engine)
        dataDict["RawPrompt"]["inputs"] = instruction
        dataDict["RawPrompt"]["outputs"] = output
        match_answer = re.search(r"Final Answer:(.*)", output, re.IGNORECASE | re.DOTALL)
        if match_answer is None:
            dataDict["ExtractResult"] = "-"
        else:
            dataDict["ExtractResult"] = match_answer.group(1).strip()

        dataDict["RawAnswer"] = dataDict["answer"]
        return output

    def _Pipline_TableClean_RawCOTPrompt(self, dataDict, model_name, engine="hf"):

        tableString = self._build_tableclean_pipeline(dataDict, model_name, engine="hf")

        instruction = self.build_first_prompt(
            system_prompt=self.RawSystemPrompt,
            user_prompt=self.RawTCOTPrompt.format(TableString=tableString, QuestionString=dataDict["question"])
        )
        output = llmCaller.infer(instruction, model_name, tools=None,engine=engine)


        match_answer = re.search(r"Final Answer:(.*)", output, re.IGNORECASE)
        if match_answer is None:
            dataDict["ExtractResult"] = "-"
        else:
            dataDict["ExtractResult"] = match_answer.group(1).strip()

        dataDict["RawAnswer"] = dataDict["answer"]
        dataDict["answer"] = dataDict["answer"].replace(",", "").replace("，", "").strip("%$")
        return output

    def _Pipline_TableCLeanToolPrompt(self, dataDict, model_name, engine="hf", TFormat="JSON"):

        # tableString = self._build_tableclean_pipeline(dataDict, model_name, engine="hf")
        tableString = tableCleanDao.pipeline(dataDict, model_name)

        response = self._build_TOOL_pipeline(dataDict, model_name, tableClean=tableString)

        match_answer = re.search(r"The final answers:.*(\[.*\])", response, re.IGNORECASE | re.DOTALL)
        if match_answer is not None:
            rets = BaseCallLLM.clean_final_answers(match_answer.group(1).replace("]]", "]"))
            try:
                dataDict["ExtractResult"] = ast.literal_eval(rets.strip())
            except:
                dataDict["ExtractResult"] = "Tobr"
        else:
            dataDict["ExtractResult"] = "-"
        # print(dataDict["ExtractResult"])
        dataDict["ExtractResult"] = [str(i).strip() for i in dataDict["ExtractResult"]]
        dataDict["ExtractResult"] = " ".join(dataDict["ExtractResult"]).replace(",", "").replace("，", "").replace("%", "").replace("$","")
        dataDict["RawAnswer"] = dataDict["answer"]
        dataDict["answer"] = dataDict["answer"].replace(",", "").replace("，", "").strip("%$")
        return response


    def _Pipline_ToolPrompt(self, dataDict, model_name, query=None, engine="hf", TFormat="JSON"):
        # tableString = self._build_tableclean_pipeline(dataDict, model_name, dataDict["table"],engine="hf")
        if TFormat == "JSON":
            tableClean = dataDict["table"]
        elif TFormat == "HTML":
            table_df = pd.DataFrame(json.loads(dataDict["table"])['data'], columns=json.loads(dataDict["table"])['columns'])
            tableClean = table_df.to_html(na_rep='-')
        response = self._build_TOOL_pipeline(dataDict, model_name, tableClean=tableClean, query=query, TFormat=TFormat)

        # match_answer = re.search(r"The Final Answer:(.*)", response, re.IGNORECASE)
        # if match_answer is None:
        #     dataDict["ExtractResult"] = "-"
        # else:
        #     dataDict["ExtractResult"] = match_answer.group(1).strip()
        # print(response)

        match_answer = re.search(r"The final answers:.*(\[.*\])", response, re.IGNORECASE | re.DOTALL)
        if match_answer is not None:
            rets = BaseCallLLM.clean_final_answers(match_answer.group(1))
            try:
                dataDict["ExtractResult"] = ast.literal_eval(rets.strip())
            except:
                dataDict["ExtractResult"] = "-"
        else:
            dataDict["ExtractResult"] = "-"
        # print(dataDict["ExtractResult"])
        dataDict["ExtractResult"] = [str(i).strip() for i in dataDict["ExtractResult"]]
        dataDict["ExtractResult"] = " ".join(dataDict["ExtractResult"]).replace(",", "").replace("，", "").replace("%", "").replace("$","")
        dataDict["RawAnswer"] = dataDict["answer"]
        dataDict["answer"] = dataDict["answer"].replace(",", "").replace("，", "").strip("%$")
        return response


    def _build_TableRAG_TXT_pipeline(self, dataDict, model_name, engine="hf"):
        self._logger = logging.getLogger(os.environ.get('logger_name'))

        rawTable = json.loads(dataDict["table"])
        tableDf = self._table_to_df(table=rawTable['data'],columns=rawTable["columns"])

        ## 1 table clean
        
        TableCleanString = self._build_tableclean_pipeline(dataDict, model_name, engine="hf")


        ## 2 call tool

        ## 3 eva
        output = self._build_TOOL_pipeline(dataDict, model_name, TableCleanString, engine="hf")

        dataDict["ExtractResult"] = " ".join(output)
        dataDict["RawAnswer"] = dataDict["answer"]
        dataDict["answer"] = dataDict["answer"].replace(",", "").replace("，", "")

        return output

    # def _Main_Tool_pipeline(self, dataDict, model_name, engine="hf"):
    #     output = self._build_TOOL_pipeline(dataDict, model_name, dataDict["table"], engine="hf")
    #     dataDict["ExtractResult"] = " ".join(output)
    #     dataDict["RawAnswer"] = dataDict["answer"]
    #     dataDict["answer"] = dataDict["answer"].replace(",", "").replace("，", "")

    #     return output


    def _build_TableRAG_pipeline(self, dataDict, model_name, engine="hf"):
        self._logger = logging.getLogger(os.environ.get('logger_name'))

        tableDf = self._table_to_df(table=json.loads(dataDict["table"])['data'],columns=json.loads(dataDict["table"])["columns"])

        SchemaRetrievalPrompt = self.SchemaRetrievalPromptFromTableRAG.format(TableExamplesWithColumns=tableDf.head(3).to_dict(orient='split'), ColumnKeys=tableDf.columns.to_list(), QuestionString=dataDict["question"])
        CellRetrievalPrompt = self.CellRetrievalPromptFromTableRAG.format(TableExamplesWithColumns=tableDf.columns.to_list(),QuestionString=dataDict["question"])
        # CellRetrievalPrompt = self.CellRetrievalPromptFromTableRAG.format(QuestionString=dataDict["question"])

        SchemaRetrievalInput = self.build_first_prompt(
            system_prompt=self.SchemaRetrievalSystemPromptFromTableRAG,
            user_prompt=SchemaRetrievalPrompt
        )

        CellRetrievalInput = self.build_first_prompt(
            system_prompt=self.CellRetrievalSystemPromptFromTableRAG,
            user_prompt=CellRetrievalPrompt
        )

        while(True):
            try:
                output1 = llmCaller.infer(SchemaRetrievalInput, model_name, tools=None, engine=engine)
                self._logger.info("1.1 第一次的列名:{Columns}".format(Columns=output1))
                response = json.loads(output1)
                break
            except json.decoder.JSONDecodeError as je:
                SchemaRetrievalInput.append({
                    "role": "assistant", "content": output1
                })
                SchemaRetrievalInput.append({
                    "role": "user", "content": f"Your response raise 'json.decoder.JSONDecodeError' in python, fix your answer."
                })
                continue
            except TypeError as te:
                SchemaRetrievalInput.append(
                    {"role": "assistant", "content": output1}
                )
                SchemaRetrievalInput.append({
                    {"role": "user", "content": f"Your response raise TypeError: unhashable type: 'dict' in python, fix your answer."}
                })
            except:
                SchemaRetrievalInput.append(
                    {"role": "assistant", "content": output1}
                )
                SchemaRetrievalInput.append(
                    {"role": "user", "content": f"Your response is not formatted list and raise error by json.loads(response), fix your answer."}
                )
        dataDict["SchemaRetrievalOutput"] = json.loads(output1)
        
        while(True):
            try:
                output2 = llmCaller.infer(CellRetrievalInput, model_name, tools=None,engine=engine)
                self._logger.info("1.2 相关索引:{Columns}".format(Columns=output2))
                response = json.loads(output2)
                break
            except json.decoder.JSONDecodeError as je:
                SchemaRetrievalInput.append(
                    {"role": "assistant", "content": output2}
                )
                SchemaRetrievalInput.append({
                    {"role": "user", "content": f"Your response raise 'json.decoder.JSONDecodeError' in python, fix your answer."}
                })
                continue
            except TypeError as te:
                SchemaRetrievalInput.append(
                    {"role": "assistant", "content": output2}
                )
                SchemaRetrievalInput.append(
                    {"role": "user", "content": f"Your response raise TypeError: unhashable type: 'dict' in python, fix your answer."}
                )
                continue
            except:
                SchemaRetrievalInput.append(
                    {"role": "assistant", "content": output1}
                )
                SchemaRetrievalInput.append(
                    {"role": "user", "content": f"Your response is not formatted list and raise error by json.loads(response), fix your answer."}
                )
        dataDict["CellRetrievalOutput"] = json.loads(output2)
        
        CellRetrievalFix = self._filter_CellRetrieval(dataDict["CellRetrievalOutput"], dataDict["SchemaRetrievalOutput"])
        dataDict["CellRetrievalFixOutput"] = CellRetrievalFix
        # self._logger.info("2.过滤后的相关索引:{Columns}".format(Columns=CellRetrievalFix))
        dataDict["SchemaRetrievalOutput"].extend(dataDict["CellRetrievalOutput"])
        table_df_columns = [i for i in dataDict["SchemaRetrievalOutput"] if i.strip() and i != "..." and i in tableDf.columns.to_list()]
        # table_df_columns = [re.sub(r'-+', '-', i.strip().replace("\\n", "-").replace("\n", "-").replace(" ","-")) for i in table_df_columns if i.strip()]
        table_df_columns = list(set(table_df_columns))
        table_df_fix = tableDf[table_df_columns].to_dict(orient='split')

        dataDict["TableCLean"] = table_df_fix
        del dataDict["answer_formatter"]

        # self._logger.info("3.最后的输入:{Columns}".format(Columns=self.TableRAGTCOTPrompt.format(TableString=table_df_fix, CellRetrievalQueries=dataDict["CellRetrievalFixOutput"], QuestionString=dataDict["question"])))

        inputs3 = self.build_first_prompt(
            system_prompt=self.TableRAGSystemPromptWithEquation,
            user_prompt=self.TableRAGTCOTPromptWithEquation.format(TableString=table_df_fix, CellRetrievalQueries=dataDict["CellRetrievalFixOutput"], QuestionString=dataDict["question"])
        )
        dataDict["iutput3_generate_expression"] = inputs3
        output3 = llmCaller.infer(inputs3, model_name, tools=None,engine=engine)
        # self._logger.info("3.1 抽取表格计算公式的结果 output3: {Columns}".format(Columns=output3))
        dataDict["output3_generate_expression"] = output3
        # inputs3_check = self.build_first_prompt(
        #     system_prompt=self.ReviseSystemPrompt,
        #     user_prompt=self.RevisePrompt.format(LastAgent=output3)
        # )
        # dataDict["inputs3_check_generate_expression"] = inputs3_check
        # output3_check = llmCaller.infer(inputs3_check, model_name, tools=None,engine=engine)
        # dataDict["output3_check_generate_expression"] = output3_check
        for _ in range(5):
            expression = ""
            try:
                inputs4 = self.build_first_prompt(
                    system_prompt="You are a table analyst. Your task is to compute the final mathematical expression based on the reasoning content.",
                    user_prompt=f"Compute mathematical expression: {output3}"
                )

                output4 = llmCaller.infer(inputs4, model_name, tools=self.tools, engine=engine)
                # self._logger.info("3.2 计算公式的结果 output4: {Columns}".format(Columns=output4))
                response = BaseCallLLM.parse_tool_calls(output4)

                if response != []:
                    for res_item in response:
                        function_name = self.tools_map[res_item["name"]]
                        expression = res_item["arguments"]["expression"]
                        result = function_name(res_item["arguments"]["expression"])
                        inputs4.append({"role": "assistant", "tool_calls": [{"type": "function", "function": res_item}]})
                        inputs4.append({
                            "role": "tool",
                            "content": result
                        })
                break
            except:
                inputs3 = self.build_first_prompt(
                    system_prompt=self.TableRAGSystemPromptWithEquation,
                    user_prompt=self.TableRAGTCOTPromptWithEquation.format(TableString=table_df_fix, CellRetrievalQueries=dataDict["CellRetrievalFixOutput"], QuestionString=dataDict["question"])
                )
                dataDict["iutput3_generate_expression"] = inputs3
                output3 = llmCaller.infer(inputs3, model_name, tools=None,engine=engine)
                # self._logger.info("3.1 抽取表格计算公式的结果 output3: {Columns}".format(Columns=output3))
                dataDict["output3_generate_expression"] = output3
                inputs3.append({"role": "assistant", "content": output4})
                inputs3.append({"role": "user", "content": f"Do not generate the format of code \"{expression}\". Instead, convert it to a complete math expression like \"1+2+3\"."})
                # messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": res_item}]})

        output5 = llmCaller.infer(inputs4, model_name, tools=None, engine=engine)
        # dataDict["input3__generate_expression"] = inputs3
        dataDict["input4"] = inputs4
        # self._logger.info(":End ======== :End")

        
        return output5
    
    def _build_TableRAG_pipeline_onlyTools(self, dataDict, model_name, engine="hf"):
        self._logger = logging.getLogger(os.environ.get('logger_name'))

        tableDf = self._table_to_df(table=json.loads(dataDict["table"])['data'],columns=json.loads(dataDict["table"])["columns"])

        inputs3 = self.build_first_prompt(
            system_prompt=self.TableRAGSystemPromptWithEquation,
            user_prompt=self.TableRAGTCOTPromptWithEquation.format(TableString=dataDict["table"], QuestionString=dataDict["question"])
        )
        dataDict["iutput3_generate_expression"] = inputs3
        output3 = llmCaller.infer(inputs3, model_name, tools=None,engine=engine)
        # self._logger.info("3.1 抽取表格计算公式的结果 output3: {Columns}".format(Columns=output3))
        dataDict["output3_generate_expression"] = output3
        # inputs3_check = self.build_first_prompt(
        #     system_prompt=self.ReviseSystemPrompt,
        #     user_prompt=self.RevisePrompt.format(LastAgent=output3)
        # )
        # dataDict["inputs3_check_generate_expression"] = inputs3_check
        # output3_check = llmCaller.infer(inputs3_check, model_name, tools=None,engine=engine)
        # dataDict["output3_check_generate_expression"] = output3_check
        for _ in range(5):
            expression = ""
            try:
                inputs4 = self.build_first_prompt(
                    system_prompt="You are a table analyst. Your task is to compute the final mathematical expression based on the reasoning content.",
                    user_prompt=f"Compute mathematical expression: {output3}"
                )

                output4 = llmCaller.infer(inputs4, model_name, tools=self.tools, engine=engine)
                # self._logger.info("3.2 计算公式的结果 output4: {Columns}".format(Columns=output4))
                response = BaseCallLLM.parse_tool_calls(output4)

                if response != []:
                    for res_item in response:
                        function_name = self.tools_map[res_item["name"]]
                        expression = res_item["arguments"]["expression"]
                        result = function_name(res_item["arguments"]["expression"])
                        inputs4.append({"role": "assistant", "tool_calls": [{"type": "function", "function": res_item}]})
                        inputs4.append({
                            "role": "tool",
                            "content": result
                        })
                break
            except:
                inputs3 = self.build_first_prompt(
                    system_prompt=self.TableRAGSystemPromptWithEquation,
                    user_prompt=self.TableRAGTCOTPromptWithEquation.format(TableString=table_df_fix, CellRetrievalQueries=dataDict["CellRetrievalFixOutput"], QuestionString=dataDict["question"])
                )
                dataDict["iutput3_generate_expression"] = inputs3
                output3 = llmCaller.infer(inputs3, model_name, tools=None,engine=engine)
                # self._logger.info("3.1 抽取表格计算公式的结果 output3: {Columns}".format(Columns=output3))
                dataDict["output3_generate_expression"] = output3
                inputs3.append({"role": "assistant", "content": output4})
                inputs3.append({"role": "user", "content": f"Do not generate the format of code \"{expression}\". Instead, convert it to a complete math expression like \"1+2+3\"."})
                # messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": res_item}]})

        output5 = llmCaller.infer(inputs4, model_name, tools=None, engine=engine)
        # dataDict["input3__generate_expression"] = inputs3
        dataDict["input4"] = inputs4
        # self._logger.info(":End ======== :End")

        
        return output5

    def _build_TableRAG_pipeline_with_tableSqify(self, dataDict, model_name, engine="hf"):
        self._logger = logging.getLogger(os.environ.get('logger_name'))
        self._logger.info("正在运行ID - {ids}的表格数据".format(ids=dataDict["id"]))
        self._logger.info("Satrt: ======== : Start")
        tableDf = self._table_to_df(table=json.loads(dataDict["table"])['data'],columns=json.loads(dataDict["table"])["columns"])

        SchemaRetrievalPrompt = self.SchemaRetrievalPromptFromTableRAG.format(TableExamplesWithColumns=tableDf.head(3).to_dict(orient='list'), ColumnKeys=tableDf.columns.to_list(), QuestionString=dataDict["question"])
        CellRetrievalPrompt = self.CellRetrievalPromptFromTableRAG.format(TableExamplesWithColumns=tableDf.columns.to_list(),QuestionString=dataDict["question"])
        # CellRetrievalPrompt = self.CellRetrievalPromptFromTableRAG.format(QuestionString=dataDict["question"])

        SchemaRetrievalInput = self.build_first_prompt(
            system_prompt=self.SchemaRetrievalSystemPromptFromTableRAG,
            user_prompt=SchemaRetrievalPrompt
        )

        CellRetrievalInput = self.build_first_prompt(
            system_prompt=self.CellRetrievalSystemPromptFromTableRAG,
            user_prompt=CellRetrievalPrompt
        )

        while(True):
            try:
                output1 = llmCaller.infer(SchemaRetrievalInput, model_name, tools=None, engine=engine)
                self._logger.info("1.1 第一次的列名:{Columns}".format(Columns=output1))
                response = json.loads(output1)
                break
            except json.decoder.JSONDecodeError as je:
                SchemaRetrievalInput.append({
                    "role": "assistant", "content": output1
                })
                SchemaRetrievalInput.append({
                    "role": "user", "content": f"Your response raise 'json.decoder.JSONDecodeError' in python, fix your answer."
                })
                continue
            except TypeError as te:
                SchemaRetrievalInput.append(
                    {"role": "assistant", "content": output1}
                )
                SchemaRetrievalInput.append({
                    {"role": "user", "content": f"Your response raise TypeError: unhashable type: 'dict' in python, fix your answer."}
                })
            except:
                SchemaRetrievalInput.append(
                    {"role": "assistant", "content": output1}
                )
                SchemaRetrievalInput.append(
                    {"role": "user", "content": f"Your response is not formatted list and raise error by json.loads(response), fix your answer."}
                )
        dataDict["SchemaRetrievalOutput"] = json.loads(output1)
        
        while(True):
            try:
                output2 = llmCaller.infer(CellRetrievalInput, model_name, tools=None,engine=engine)
                self._logger.info("1.2 相关索引:{Columns}".format(Columns=output2))
                response = json.loads(output2)
                break
            except json.decoder.JSONDecodeError as je:
                SchemaRetrievalInput.append(
                    {"role": "assistant", "content": output2}
                )
                SchemaRetrievalInput.append({
                    {"role": "user", "content": f"Your response raise 'json.decoder.JSONDecodeError' in python, fix your answer."}
                })
                continue
            except TypeError as te:
                SchemaRetrievalInput.append(
                    {"role": "assistant", "content": output2}
                )
                SchemaRetrievalInput.append(
                    {"role": "user", "content": f"Your response raise TypeError: unhashable type: 'dict' in python, fix your answer."}
                )
                continue
            except:
                SchemaRetrievalInput.append(
                    {"role": "assistant", "content": output1}
                )
                SchemaRetrievalInput.append(
                    {"role": "user", "content": f"Your response is not formatted list and raise error by json.loads(response), fix your answer."}
                )
        dataDict["CellRetrievalOutput"] = json.loads(output2)
        
        CellRetrievalFix = self._filter_CellRetrieval(dataDict["CellRetrievalOutput"], dataDict["SchemaRetrievalOutput"])
        dataDict["CellRetrievalFixOutput"] = CellRetrievalFix
        self._logger.info("2.过滤后的相关索引:{Columns}".format(Columns=CellRetrievalFix))

        table_df_columns = [i for i in dataDict["SchemaRetrievalOutput"] if i.strip() and i != "..." and i in tableDf.columns.to_list()]
        # table_df_columns = [re.sub(r'-+', '-', i.strip().replace("\\n", "-").replace("\n", "-").replace(" ","-")) for i in table_df_columns if i.strip()]
        table_df_fix = tableDf[table_df_columns].to_dict(orient='list')

        dataDict["TableCLean"] = table_df_fix
        del dataDict["answer_formatter"]

        self._logger.info("3.最后的输入:{Columns}".format(Columns=self.TableRAGTCOTPrompt.format(TableString=table_df_fix, CellRetrievalQueries=dataDict["CellRetrievalFixOutput"], QuestionString=dataDict["question"])))
        inputs3 = self.build_first_prompt(
            system_prompt=self.TableRAGSystemPrompt,
            user_prompt=self.TableRAGTCOTPrompt.format(TableString=table_df_fix, CellRetrievalQueries=dataDict["CellRetrievalFixOutput"], QuestionString=dataDict["question"])
        )

        output3 = llmCaller.infer(inputs3, model_name, tools=None,engine=engine)
        self._logger.info(":End ======== :End")
        return output3

    def _filter_CellRetrieval(self, CellRetrieval, columns):
        res = []
        
        for i in CellRetrieval:
            is_find = 0
            for j in columns:
                if i.lower() in j.lower():
                    is_find = 1
                    break
            if is_find == 0:
                res.append(i)
                
        return res

        
    def _table_to_df(self, table, columns):
        tableColumnFix = [re.sub(r'-+', '-', i.strip().replace("\\n", "-").replace("\n", "-").replace(" ","-")) for i in columns if i.strip()]
        return pd.DataFrame(table, columns=tableColumnFix)


    def _build_four_pipeline(self, dataDict, model_name, engine="hf"):
        """
        三段式的pipeline
          1. 获取简化的行列明
          2. 获取最终结果的计算公式
          3. 调用工具，获取结果
          4. 根据工具的结果获取最终计算结果

        详细步骤：
          1. 使用 `build_first_prompt` 方法构建第一个提示，获取简化的行列明。
          2. 使用 `infer` 方法进行推理，获取简化的行列明结果。
          3. 使用 `build_first_prompt` 方法构建第二个提示，获取最终结果的计算公式。
          4. 使用 `infer` 方法进行推理，获取计算公式结果。
          5. 如果提供了工具，则调用工具解析计算公式并获取结果。
          6. 使用 `infer` 方法进行最终推理，获取最终计算结果。

        参数：
          - dataDict: 包含问题和表格数据的字典。
          - model_name: 使用的模型名称。
          - engine: 使用的推理引擎，默认为 "hf"。

        返回值：
          - 最终的计算结果。
        """
        ## 1.生成简化的行列
        self._logger = logging.getLogger(os.environ.get('logger_name'))
        self._logger.info("正在运行ID{ids}的表格数据".format(ids=dataDict["id"]))

        tableColumnFix = [re.sub(r'-+', '-', i.strip().replace("\\n", "-").replace("\n", "-").replace(" ","-")) for i in json.loads(dataDict["table"])["columns"] if i.strip()]

        inputs1 = self.build_first_prompt(
            system_prompt=self.GenCleanTableSystem1Prompt,
            user_prompt=self.GenCleanTable1Prompt.format(ColumnNames=tableColumnFix,QuestionString=dataDict["question"])
        )
        self._logger.info("第一次的输入为:{Columns}".format(Columns=inputs1))
        output1 = llmCaller.infer(inputs1, model_name, tools=None,engine=engine)
        self._logger.info("第一次的回复为:{Columns}".format(Columns=output1))
        table_df = pd.DataFrame(json.loads(dataDict["table"])['data'], columns=tableColumnFix)
        table_df = table_df.replace(to_replace=r'^(?i)(None|null|)$', value='-', regex=True)
        table_df = table_df.replace(to_replace=[None, ''], value='-', regex=False)
        table_df = table_df.fillna("-")

        table_df_columns = [i.strip(" \"") for i in self._extract_final_output(input_text=output1, final_sentence="Relevant column names:").split(",")]
        self._logger.info("简化后的列名为:{Columns}".format(Columns=table_df_columns))
        table_df_columns = [i for i in table_df_columns if i.strip() and i != "..." and i != "\"" and i in tableColumnFix]
        table_df_fix = table_df[table_df_columns].to_json(orient="split")
        self._logger.info("简化后的表格为:{Columns}".format(Columns=table_df_fix))
        ## 2.生成列
        inputs2 = self.build_first_prompt(
            system_prompt=self.RawSystemPrompt,
            user_prompt=self.RawTCOTPrompt.format(TableString=table_df_fix, QuestionString=dataDict["question"])
        )
        self._logger.info("第二次的输入为:{Columns}".format(Columns=inputs2))
        output2 = llmCaller.infer(inputs2, model_name, tools=None,engine=engine)
        self._logger.info("第二次的回复为:{Columns}".format(Columns=output2))

        return output2

    def _build_tableclean_pipeline(self, dataDict, model_name, engine="hf"):
        dataDict["TableClean"] = {}
        TableCleanPrompt1_DG = BaseCallLLM.load_prompt_from_txt(self.TableCleanPrompt1_DG)
        TableCleanPrompt1_DG = TableCleanPrompt1_DG.replace("{{TableString}}", dataDict["table"])
        TableCleanPrompt1_DG = TableCleanPrompt1_DG.replace("{{QuestionString}}", dataDict["question"])

        inputs1 = self.build_first_prompt(
            system_prompt=self.TableCleanSystemPrompt1,
            user_prompt=TableCleanPrompt1_DG
        )

        output1 = llmCaller.infer(inputs1, model_name, tools=None, engine=engine).replace("，",",")

        dataDict["TableClean"]["input"] = inputs1
        dataDict["TableClean"]["output"] = output1

        json_match = re.search(r"```json\n(.*?)```", output1, re.DOTALL)


        try:
            if json_match:
                json_data_str = json_match.group(1)
                # 解析JSON字符串为Python对象
                # print(json_data_str)
                json_data = json.loads(json_data_str)
                dataDict["TableClean"]["output_Extract"] = json_data
            else:
                json_data = dataDict["table"]
                dataDict["TableClean"]["output_Extract"] = "-"
        except:
            json_data = dataDict["table"]
            dataDict["TableClean"]["output_Extract"] = "-"
            dataDict["TableClean"]["json_Error"] = output1

        return json_data

    def _build_DP_pipeline(self, dataDict, model_name, tableClean, query=None,engine="hf"):
        dataDict["DP"] = {}
        query = dataDict["question"] if query is None else query
        
        TableCleanPrompt1_DP = BaseCallLLM.load_prompt_from_txt(self.TableCleanPrompt1_DP)
        TableCleanPrompt1_DP = TableCleanPrompt1_DP.replace("{{TableString}}", json.dumps(tableClean, indent=2))
        
        TableCleanPrompt1_DP = TableCleanPrompt1_DP.replace("{{QuestionString}}", query)

        inputs1 = self.build_first_prompt(
            system_prompt=self.TableCleanSystemPrompt1_DP,
            user_prompt=TableCleanPrompt1_DP
        )
        rep_times = 0
        while(True):
            output = llmCaller.infer(inputs1, model_name, tools=None, engine=engine)
            if_exist_answer = re.search(r"The Final Answer:.*(\[.*\])",output.split("\n")[-1].strip(),re.IGNORECASE | re.DOTALL)
            if if_exist_answer is None:
                dataDict["DP"]["output_uncomplete"] = output
                inputs1.extend([
                    {"role": "assistant", "content": output},
                    {"role": "user", "content": "Continue to finish your step and answer the question by the format."}
                ])
                # output = llmCaller.infer(inputs1, model_name, tools=None, engine=engine)
                # self._logger.info(output)
                self._logger.debug(f"已经重复运行{rep_times}次未完成对话。")
                rep_times += 1
                continue
            break



        dataDict["DP"]["inputs"] = inputs1
        dataDict["DP"]["output"] = output

        output = [str(i) for i in BaseCallLLM.extract_result(output)]
        
        return output

    def _build_TOOL_pipeline(self, dataDict, model_name, tableClean, query=None, engine="hf", TFormat="JSON"):
        # dataDict["TableClean"] = {}
        self._logger = logging.getLogger(os.environ.get('logger_name'))
        query = dataDict["question"] if query is None else query
        dataDict["TOOL"] = {}
        if TFormat == "JSON":
            TableCleanPrompt1_DP = BaseCallLLM.load_prompt_from_txt(self.TableCleanPrompt1_TOOL)
        elif TFormat == "HTML":
            TableCleanPrompt1_DP = BaseCallLLM.load_prompt_from_txt(self.TableCleanPrompt1_TOOL_HTML)
        TableCleanPrompt1_DP = TableCleanPrompt1_DP.replace("{{TableString}}", tableClean)
        TableCleanPrompt1_DP = TableCleanPrompt1_DP.replace("{{QuestionString}}", query)

        inputs1 = self.build_first_prompt(
            system_prompt=self.TableCleanSystemPrompt1_TOOL,
            user_prompt=TableCleanPrompt1_DP
        )

        rep_times = 0
        while(True):

            output = llmCaller.infer(inputs1, model_name, tools=None, engine=engine)
            self._logger.info(output)

            if_exist_answer = re.search(r"The final answers:.*(\[.*\])",output.strip(), re.IGNORECASE | re.DOTALL)
            if if_exist_answer is None:
                dataDict["TOOL"]["output_uncomplete"] = output
                inputs1.extend([
                    {"role": "assistant", "content": output},
                    {"role": "user", "content": "Continue to finish your step and answer the question by the required format concisely."}
                ])
                # output = llmCaller.infer(inputs1, model_name, tools=None, engine=engine)
                self._logger.debug(f"已经重复运行{rep_times}次未完成对话。")
                rep_times += 1
                if rep_times == 4:
                    break
                continue
            break


        dataDict["TOOL"]["inputs"] = inputs1
        dataDict["TOOL"]["outputs"] = output

        # self._logger.info(BaseCallLLM.extract_result(output, prefix="The final answers:"))
        # output = [str(i) for i in BaseCallLLM.extract_result(output, prefix="The final answers:")]
        output = output.strip()
        match = re.search(f"The final answers:", output, re.IGNORECASE | re.DOTALL)

        if match is not None:
            output_tool = toolDao.tool_pipeline(output, model_name, dataDict=dataDict)
        else:
            output_tool = output


        return output_tool
    
    # def _rep_complete_pipeline(self, output, inputs1, model_name, tools=None, engine=engine)

        






tableBenchCallLLM = TableBenchCallLLM()