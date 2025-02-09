from dao.Agents.BaseAgent import BaseAgent
from dao.LLMCaller import llmCaller
import json

class Agent1:
    def __init__(self):
        self.SystemPrompt="You are a table analyst."
        self.UserPrompt=BaseAgent.load_prompt("./prompts/Agent1PromptSimple.txt")
        
        self.ReflectionSystem="You are tasked with reviewing and reflecting on a previously provided output to ensure its correctness."
        self.ReflectionPrompt=BaseAgent.load_prompt("./prompts/Agent1PromptSimpleWithReflection.txt")

    def infer_with_QueryDecomposition(self, dataDict, model_name, engine="hf", **kwargs):
        dataDict["agent1"] = {}
        instruction=self.replace_prompt_QueryDecomposition(
            Query=dataDict["question"]
        )
        dataDict["agent1"]["input1"] = instruction
        response = llmCaller.infer(instruction, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
        dataDict["agent1"]["output1"] = response
        try:
            subQuery = BaseAgent.extract_json(response)["steps"]

            # subQuery["subQueryCount"] += 1
        except:
            subQuery = {
                "step": 1,
                "description": "Direct answer the question: " + dataDict["question"]
            }
        return self.steps2NL(subQuery)
        
    def steps2NL(self, steps):
        prompts = "\n- Step{step}: {desc}\n"
        toNL = ""
        for step in steps:
            toNL += prompts.format(step=step["step"],desc=step["description"])
        return toNL


    def infer_with_no_reflection(self, dataDict, model_name, engine="hf", **kwargs):
        dataDict["agent1"] = {}
        instruction=self.replace_prompt_with_reflection(
            Query=dataDict["question"]
        )
        dataDict["agent1"]["input1"] = instruction
        response = llmCaller.infer(instruction, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
        dataDict["agent1"]["output1"] = response
        try:
            subQuery = BaseAgent.extract_json(response)
            if subQuery["subQueryCount"] == 1:
                return subQuery["subQueries"]
            else:
                subQuery["subQueries"].append(dataDict["question"])
                return subQuery["subQueries"]
            subQuery["subQueryCount"] += 1
        except:
            subQuery = {
                "subQueryCount": 1,
                "subQueries": [dataDict["question"]]
            }
        return subQuery["subQueries"]
    def infer_with_reflection(self, dataDict, model_name, engine="hf", **kwargs):
        dataDict["agent1"] = {}
        instruction=self.replace_prompt_with_reflection(
            Query=dataDict["question"]
        )
        dataDict["agent1"]["input1"] = instruction
        response = llmCaller.infer(instruction, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
        dataDict["agent1"]["output1"] = response
        try:
            subQuery = BaseAgent.extract_json(response)
            if len(subQuery["subQueries"]) != 1:
                subQuery["subQueries"].append(dataDict["question"])
            subQuery["subQueryCount"] += 1
            return subQuery["subQueries"]
        except:
            subQuery = {
                "subQueryCount": 1,
                "subQueries": [dataDict["question"]]
            }
            

        
        reflectionInstruction = self.replace_prompt_reflection(
            Query=dataDict["question"],
            OutputToReview=subQuery
        )
        dataDict["agent1"]["input2"] = reflectionInstruction

        response = llmCaller.infer(reflectionInstruction, model_name, engine=engine, dataMethod=kwargs["dataMethod"])
        dataDict["agent1"]["output2"] = response
        try:
            ref = self.parse_reflection(response, dataDict)
            # If reflection is valid, set it to the question or parsed sub-queries
            if ref is True:
                ref = [dataDict["question"]]
            dataDict["agent1"]["parseError"] = 0
            
        except Exception as e:
            print(e)
            # Handle any exceptions, fallback to default reflection
            dataDict["agent1"]["parseError"] = 1
            ref = [dataDict["question"]]
        return ref
    
    def infer(self, dataDict, model_name, engine="hf"):
        # print(dataDict["table"])
        datas = json.loads(dataDict["table"])
        instruction=self.replace_prompt(
            columns=datas["columns"], 
            data=datas["data"][1:4], 
            query=dataDict["question"]
        )
        response = llmCaller.infer(instruction, model_name, engine=engine)
        try:
            subQuery = BaseAgent.extract_json(response)
        except:
            subQuery = {
                "SubQuery": dataDict["question"],
                "SubQueryHint": ""
            }
        return subQuery

    def replace_prompt_QueryDecomposition(self, **inputs):
        return BaseAgent.build_first_prompt(system_prompt=self.SystemPrompt,user_prompt=self.UserPrompt.replace("{{QueryString}}", inputs["Query"]))

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
            "query": query
        }
        return BaseAgent.build_first_prompt(system_prompt=self.SystemPrompt,user_prompt=self.UserPrompt.replace("{{Inputs}}", json.dumps(inputs, indent=2, ensure_ascii=False)))
    
    def parse_reflection(self, response, dataDict):
        ex_response = BaseAgent.extract_reflection(response)
        dataDict["agent1"]["Agent1Reflection"] = ex_response["isCorrect"]
        return True if ex_response["isCorrect"] else ex_response["correctedOutput"]["subQueries"]

agent1 = Agent1()