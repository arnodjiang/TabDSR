from models.BaseCallLLM import BaseCallLLM
from dao.LLMCaller import llmCaller

import pandas as pd
import json
import re
import os
import io
import sys
import logging

class PotDao:
    def __init__(self) -> None:
        self.prompt_root = BaseCallLLM.prompt_root
        self.SystemPrompt = BaseCallLLM.PotSystemPrompt
        self.UserPrompt = BaseCallLLM.load_prompt_from_txt(BaseCallLLM.PotPrompt)

    def pipeline_step3(self, dataDict, model_name, tablestring=None,engine="hf", **kwargs):
        llmCaller.init_model(model_name)
        dataDict["step3"] = {}

        table = json.loads(dataDict["table"].strip()) if tablestring is None else json.loads(tablestring.strip())
        # print("====")
        # print(table)
        # print(type(table))
        # print(type(json.loads(tablestring)))
        # print("====")
        tablestring = self._compose_prompt_table(table, dataDict["question"])

        # instruction = BaseCallLLM.build_first_prompt(
        #     system_prompt=self.SystemPrompt,
        #     user_prompt=self.UserPrompt.replace(f"{{InputString}}", "table_data="+json.dumps(tablestring,indent=2)+"\ntable_df = pd.DataFrame(table_data['data'], columns=table_data['columns'])")
        # )
        instruction = BaseCallLLM.build_first_prompt(
            system_prompt=self.SystemPrompt,
            user_prompt="1+1=?"
        )
        dataDict["step3"]["input"] = instruction
        print("======")
        print(instruction)
        print(len(instruction))
        print("========")
        response = llmCaller.infer(instruction, model_name, engine=engine)
        print(response)
        print("========2")
        dataDict["step3"]["output"] = response
        # print(response)
        # print("======")
        # print(response)
        # print(type(response))
        # print("========")
        codes = self.inentify_python_code(response)

        dataDict["step3"]["exec_code"] = codes

        try:
            dataDict["ExtractResult"] = self.exec_code(codes).strip()
            dataDict["step3"]["step3_code_success"] = True
        except:
            dataDict["ExtractResult"] = "-"
            dataDict["step3"]["step3_code_success"] = False


        dataDict["RawAnswer"] = dataDict["answer"]
        dataDict["answer"] = dataDict["answer"].replace(",", " ").replace("，", " ").replace("  "," ").strip("%$.")
        dataDict["FinalResult"] = ""
        tmpresult = []
        for rawanswer in dataDict["answer"].split(" "):
            if rawanswer.lower() in dataDict["ExtractResult"].lower():
                tmpresult.append(rawanswer.lower().strip())
        dataDict["FinalResult"] = " ".join(tmpresult)
        if dataDict["FinalResult"] == dataDict["answer"]:
            dataDict["score"] = 1
        else:
            dataDict["score"] = 0

    def pipeline_only_one_round(self, dataDict, model_name, tablestring=None,engine="hf", **kwargs):
        """
        只运行一次代码，不多次重复保证代码准确性
        """
        logger = logging.getLogger(os.environ.get('logger_name'))

        table = json.loads(dataDict["table"]) if tablestring is None else json.loads(tablestring)
        try:
            tablestring = self._compose_prompt_table(table, dataDict["question"])
        except:
            tablestring = self._compose_prompt_table(json.loads(dataDict["table"]), dataDict["question"])

        tablestring["critical_columns"] = table.get("critical_columns", "<All columns is critical>")
        tablestring["critical_rows"] = table.get("critical_rows", "<All rows is critical>")
        instruction = BaseCallLLM.build_first_prompt(
            system_prompt=self.SystemPrompt,
            user_prompt=self.UserPrompt.replace(f"{{InputString}}", "```python\ntable_data="+json.dumps(tablestring,indent=2)+"\ntable_df = pd.DataFrame(table_data['data'], columns=table_data['columns'])\n```")
        )
        response = llmCaller.infer(instruction, model_name, engine=engine)
        codes = self.inentify_python_code(response)
        dataDict["outputs"] = {}
        dataDict["outputs"]["output"] = response
        dataDict["outputs"]["codes"] = codes
        try:
            codes = "table_data="+json.dumps(tablestring,indent=2).strip()+"\n"+codes
            dataDict["ExtractResult"] = self.exec_code(codes).strip()
        except Exception as e:
            data_id = dataDict["id"]
            logger.error(f"An error occurred: {e}, id={data_id}.")
            dataDict["ExtractResult"] = "-"
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

    def pipeline(self, dataDict, model_name, tablestring=None,engine="hf", **kwargs):
        logger = logging.getLogger(os.environ.get('logger_name'))
        # print(tablestring)
        table = json.loads(dataDict["table"]) if tablestring is None else json.loads(tablestring)
        # columns = self.rename_duplicate_columns(table["columns"])
        # table["columns"] = columns
        try:
            tablestring = self._compose_prompt_table(table, dataDict["question"])
        except:
            tablestring = self._compose_prompt_table(json.loads(dataDict["table"]), dataDict["question"])
        # print(table)
        tablestring["critical_columns"] = table.get("critical_columns", "<All columns is critical>")
        tablestring["critical_rows"] = table.get("critical_rows", "<All rows is critical>")
        
        
        
        instruction = BaseCallLLM.build_first_prompt(
            system_prompt=self.SystemPrompt,
            user_prompt=self.UserPrompt.replace(f"{{InputString}}", "```python\ntable_data="+json.dumps(tablestring,indent=2)+"\ntable_df = pd.DataFrame(table_data['data'], columns=table_data['columns'])\n```")
        )
        # print(instruction)
        response = llmCaller.infer(instruction, model_name, engine=engine)


        dataDict["outputs"] = {}
        dataDict["outputs"]["pythoncode"] = response
        # print(response)
        codes = self.inentify_python_code(response)
        # print("codes:")
        # logger.debug(f"codes:\n{codes}\n---------")
        
        # print("codes:")
        # logger.debug(f"codes:\n{codes}\n---------")
        if codes is None:
            instruction.extend([
                {"role": "assistant", "content": response},
                {"role": "user", "content": "Continue to finish your step and answer the question by the required format completely."}
            ])
            response = llmCaller.infer(instruction, model_name, engine=engine)
            codes = self.inentify_python_code(response)
            if codes is None:
                dataDict["ExtractResult"] = "-"
                return response
            
        dataDict["outputs"]["exec"] = codes
        # print(codes)

        try:
            # 尝试执行代码并提取结果
            # print(codes)
            codes = "table_data="+json.dumps(tablestring,indent=2).strip()+"\n"+codes
            dataDict["ExtractResult"] = self.exec_code(codes).strip()
            # res = dataDict["ExtractResult"]
            # print("res:")
            # logger.debug(f"res:\n{res}\n---------")
        except Exception as e:

            instruction.extend([
                {"role": "assistant", "content": f"Error encountered: {e}. There was an issue processing the data. Please check the provided code and try again."},
                {"role": "user", "content": "Continue to fix your python code error and answer the question by the required format completely."}
            ])

            response = llmCaller.infer(instruction, model_name, engine=engine)
            codes = self.inentify_python_code(response)
            if codes is None:
                dataDict["ExtractResult"] = "-"
                return response
            
            dataDict["outputs"]["exec"] = codes
            # print(codes)
            try:
                dataDict["ExtractResult"] = self.exec_code(codes).strip()
            except:
                dataDict["ExtractResult"] = "-"
        try:
            dataDict["ExtractResult"] = self.exec_code(codes).strip()
        except:
            pass
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
        dataDict["outputs"]["pythoncode2"] = response

    def _compose_prompt_table(self, tableString, query):
        # self._logger = logging.getLogger(os.environ.get('logger_name'))
        # print(tableString)
        # table_df = pd.DataFrame(tableString['data'], columns=tableString["columns"])

        # tablestring = json.loads(table_df.to_json(orient="split"))
        tableString["UserQuery"] = query


        return tableString
    

    
    def inentify_python_code(self, result):
        matcch_result = re.search(r"```python(.*)```", result, re.IGNORECASE | re.DOTALL)
        if matcch_result is None:
            return matcch_result
        else:
            return matcch_result.group(1)
        
    def exec_code(self, code):
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
    
    def rename_duplicate_columns(self, columns):
        # 创建一个字典来跟踪每个列名的出现次数
        seen = {}
        fixed_columns = []

        for col in columns:
            # 如果列名已经出现过，添加后缀
            if col in seen:
                seen[col] += 1
                new_col = f"{col}_{seen[col]}"
            else:
                seen[col] = 0
                new_col = col
            
            fixed_columns.append(new_col)

        return fixed_columns
potDao = PotDao()