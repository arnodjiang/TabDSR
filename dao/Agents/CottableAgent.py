from dao.Agents.BaseAgent import BaseAgent
from dao.LLMCaller import llmCaller

import json
import os, re
import pandas as pd
from loguru import logger

class CotableAgent:
    def __init__(self):
        self.promptpath = "./prompts/cotable"
        self.GenPromptPath = BaseAgent.load_prompt(os.path.join(self.promptpath, "generate.txt"))
        self.GroupPromptPath = BaseAgent.load_prompt(os.path.join(self.promptpath, "group.txt"))

    def infer(self, dataDict, **kwargs):

        table_dict = dataDict["chainOfTableDf"] if isinstance(dataDict["chainOfTableDf"], dict) else json.loads(dataDict["chainOfTableDf"])
        table_df = pd.DataFrame(table_dict["data"], columns=table_dict["columns"])
        question = dataDict["Question"]
        TableTitle = dataDict.get("TabeTitle", None)
        TableTitle = f"\nTable Title:\n{TableTitle}\n" if TableTitle else ""
        groupcolumn = ""
        if "Group ID" in table_dict["columns"]:
            groupcolumn = self.GroupPromptPath.replace("{{GroupColumn}}", table_dict["columns"][1]).replace("{{CleanTable}}", self.dataframe_to_simple(table_df))
        prompt = self.GenPromptPath.replace("{{RawTable}}",self.dataframe_to_raw_string(table_df)).replace("{{TableTitle}}", TableTitle).replace("{{GroupData}}",groupcolumn).replace("{{Question}}",question)
        prompt = [{"role": "user", "content": prompt}]
        response = llmCaller.infer(prompt, model_name = kwargs["model_name"], engine=kwargs["engine"], dataMethod=kwargs["dataMethod"])
        dataDict["RawInfer"] = response

        extract_answer = self.extract_the_final_annswer(response)
        
        # return self.extract_final_annswer(response)
        return extract_answer

    def extract_final_annswer(self, text):
        extractAnswer = re.findall("Final Answer:(.*)", text, re.IGNORECASE)
        if extractAnswer == []:
            return ""
        else:
            answer = extractAnswer[-1].strip()
            return answer
    def extract_the_final_annswer(self, text):
        text = text.replace("```", "").strip()
        extractAnswer = re.findall("The answer is:\s*(.*)", text, re.IGNORECASE|re.DOTALL)
        if extractAnswer == []:
            text = "\n".join(text.split("\n")[-2:])
        else:
            extractAnswer = extractAnswer[-1].strip()
        if extractAnswer==[] or extractAnswer.strip() == "":
            text = "\n".join(text.split("\n")[-2:])
        else:
            text = extractAnswer
        text = "\n".join(text.split("\n")[-1:]) if len(text) > 100 else text
        return text

    def dataframe_to_simple(self, df):
        result = []
        header = " | ".join(df.columns)
        result.append(header)
        for steps, row in df.iterrows():
            formatted_row = " | ".join([str(row[col]) for col in df.columns])
            result.append(formatted_row)
        return "\n".join(result)
    
    def dataframe_to_raw_string(self, df):
        """
        Converts a DataFrame into a custom structured string with column names included.

        Parameters:
            df (pd.DataFrame): DataFrame with any columns.

        Returns:
            str: The formatted string with the custom structure.
        """
        result = []
        header = "col: " + " | ".join(df.columns)
        result.append(header)
        for steps, row in df.iterrows():
            formatted_row = f"row {steps} : " + " | ".join([str(row[col]) for col in df.columns])
            result.append(formatted_row)
        return "\n".join(result)
cotableAgent = CotableAgent()