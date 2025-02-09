import pandas as pd
import json
from models.BaseCallLLM import BaseCallLLM
from dao.LLMCaller import llmCaller
import os
import json
import logging

class QueryReviseCaller:
    def __init__(self) -> None:
        pass

    def pipeline(self, dataDict, model_name, engine="hf"):
        dataDict["queryRevise"] = {}
        table = json.loads(dataDict["table"])
        tablestring = self._compose_prompt_table(table, dataDict["question"])
        prompt = BaseCallLLM.build_QueryRevise_prompt(tablestring)
        instruction = BaseCallLLM.build_first_prompt(
            system_prompt=BaseCallLLM.QueryReviseSystemPrompt,
            user_prompt=prompt
        )

        output = llmCaller.infer(instruction, model_name, tools=None,engine=engine)
        dataDict["queryRevise"]["output"] = output

        parse_result = BaseCallLLM.parse_json_response(output)
        
        structure = json.loads(parse_result).get("Reformulated_Query", "-")

        dataDict["queryRevise"]["output"] = structure

        return structure

    def _compose_prompt_table(self, tableString, query):
        self._logger = logging.getLogger(os.environ.get('logger_name'))
        table_df = pd.DataFrame(tableString['data'], columns=tableString["columns"])
        tablestring = json.loads(table_df.to_json(orient="split"))
        tablestring["UserQuery"] = query
        del tablestring["index"]


        return tablestring

queryReviseCaller = QueryReviseCaller()