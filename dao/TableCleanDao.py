from models.BaseCallLLM import BaseCallLLM
from dao.LLMCaller import llmCaller

import pandas as pd
import json
import re
import os
import io, sys


class TableCleanDao:
    def __init__(self) -> None:
        self.prompt_root = BaseCallLLM.prompt_root
        self.SystemPrompt = BaseCallLLM.TableCleanSystemPrompt
        self.UserPrompt = BaseCallLLM.load_prompt_from_txt(BaseCallLLM.TableCleanPromptPath)

        self.Step2Prompt = BaseCallLLM.load_prompt_from_txt(BaseCallLLM.Step2PromptPath)

    def pipeline_step2(self, dataDict, model_name, engine="hf", **kwargs):

        input_json = self._compose_step2(dataDict["table"], dataDict["question"])
        instruction = BaseCallLLM.build_first_prompt(
            system_prompt=self.SystemPrompt,
            user_prompt=input_json
        )
        response = llmCaller.infer(instruction, model_name, engine=engine)
        
        dataDict["step2"] = {}
        try:
            response_code = self._parse_step2_result(response)
            code_result = self.exec_code(response_code)
            table_dict = json.loads(code_result)
            table_df = pd.DataFrame(table_dict['data'], columns=table_dict["columns"])
            SimpleTable = table_dict
            dataDict["step2"]["SimpleSuccess"] = True
        except:
            SimpleTable = dataDict["table"]
            dataDict["step2"]["SimpleSuccess"] = False
        ### output
        dataDict["step2"] = {}
        dataDict["step2"]["input"] = instruction
        dataDict["step2"]["output"] = response
        dataDict["step2"]["SimpleTable"] = SimpleTable
        return SimpleTable
        

    def _compose_step2(self, table_string, question):
        inputs = self.Step2Prompt.replace("{{TableString}}", table_string).replace("{{Question}}", question)
        return inputs

    def _parse_step2_result(self, response):
        result = re.findall(r"```python(.*?)```", response, re.IGNORECASE | re.DOTALL)
        response = ""
        if result:
            response = result[-1]
        else:
            response = None
        return response

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
        return captured_output

    def pipeline(self, dataDict, model_name, engine="hf", **kwargs):

        # table = json.loads(dataDict["table"])
        # table_df = pd.DataFrame(table['data'], columns=table["columns"])
        # row_len, col_len = len(table['data']), len(table["columns"])
        # tablestring = self._compose_prompt_table(table, dataDict["question"])
        # prompt = BaseCallLLM.build_QueryRevise_prompt(tablestring)

        instruction = BaseCallLLM.build_first_prompt(
            system_prompt=self.SystemPrompt,
            user_prompt=self.UserPrompt.replace(f"{{InputString}}", dataDict["table"])
        )

        dataDict["TableClean"] = {}
        dataDict["TableClean"]["outputs"] = []
        dataDict["TableClean"]["inputs"] = instruction
        response = llmCaller.infer(instruction, model_name, engine=engine)
        # print(response)
        dataDict["TableClean"]["outputs"].append(response)
        clean_content = self._get_content_from_json(response)
        dataDict["TableClean"]["clean_table"] = clean_content


        # if clean_content == "":
        #     clean_content = dataDict["table"]

        return clean_content
        # for _ in range(3):
        #     dataDict["TableClean"]["inputs"] = instruction
        #     response = llmCaller.infer(instruction, model_name, engine=engine)
        #     print(instruction)
        #     print(response)

        #     parse_result = BaseCallLLM.parse_json_response(response)
        #     dataDict["TableClean"]["outputs"].append(response)
            
        #     try:
        #         structure = json.loads(parse_result)
        #     except:
        #         return table_df.to_json(orient="split")
        #     # print(structure)
        #     break
            
            # try:
            #     tablesrt = self._get_sub_table_from_idx(table, structure)
            #     break
            # except IndexError as IE:
            #     # instruction.append({"role": "assistant", "content": response})
            #     # instruction.append({"role": "user", "content": f"The valid row indices range from 0 to {row_len - 1} and valid col indices range from 0 to {col_len - 1}. Please recheck the generated indices."})

            #     tablesrt = table_df
        # return tablesrt.to_json(orient="split")
    
    def _compose_prompt_table(self, tableString, query):
        # self._logger = logging.getLogger(os.environ.get('logger_name'))
        table_df = pd.DataFrame(tableString['data'], columns=tableString["columns"])
        tablestring = json.loads(table_df.to_json(orient="split"))
        # tablestring["UserQuery"] = query


        return tablestring
    
    def _get_sub_table_from_idx(self, table, table_indexes):
        tabledf = pd.DataFrame(table['data'], columns=table["columns"])
        row_idx = table_indexes["row_indices"] if table_indexes["row_indices"] != [] else slice(None)  # 使用 slice(None) 作为默认值，表示选取所有行
        col_idx = table_indexes["column_indices"] if table_indexes["column_indices"] != [] else slice(None)  # 使用 slice(None) 作为默认值，表示选取所有列

        return tabledf.iloc[row_idx, col_idx]

    def _get_content_from_json(self, string):
        result = re.findall(r"```json(.*?)```", string, re.IGNORECASE | re.DOTALL)

        if result is not None or result != []:
            return json.loads(result[-1].strip("\n").strip("```json").strip("```").strip("\n"))
        else:
            return None
    
tableCleanDao = TableCleanDao()