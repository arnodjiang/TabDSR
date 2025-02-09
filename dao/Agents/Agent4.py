from dao.Agents.BaseAgent import BaseAgent
from dao.LLMCaller import llmCaller
from dao.Agents.LlmValidAgent import llmValidAgent
import json
import os
import logging

class Agent4:
    def __init__(self):
        self.SystemPrompt="You are a data analyst proficient in Python. Your task is to write executable Python code to analyze the table and then answer questions."
        self.UserPrompt=BaseAgent.load_prompt("./prompts/Agent4Prompt.txt")


    def infer(self, dataDict, model_name, engine="hf", agent_querys=None):
        logger = logging.getLogger(os.environ.get('logger_name'))
        datas = json.loads(dataDict["table"])
        instruction=self.replace_prompt(
            columns=datas["columns"],
            data=datas["data"],
            index=[i for i in range(len(datas["data"]))],
            query=dataDict["question"],
            SubQueries=agent_querys
        )
        # print(instruction)
        # print("=====")
        response = llmCaller.infer(instruction, model_name, engine=engine)
        # print(BaseAgent.extract_json(response))
        PythonCode = BaseAgent.extract_python(response)
        # res = self.get_code_result(dataDict, PythonCode).strip()
        try:
            res = self.get_code_result(dataDict, PythonCode).strip()
        except TypeError as te:
            logger.error(f"内容报错{te}, 报错内容为{PythonCode}")
            res = "-"
        except NameError as ne:
            logger.error(f"内容报错{ne}, 报错内容为{PythonCode}")
            res = "-"
        except ValueError as ve:
            logger.error(f"内容报错{ve}, 报错内容为{PythonCode}")
            res = "-"
        except SyntaxError as se:
            logger.error(f"内容报错{se}, 报错内容为{PythonCode}")
            res = "-"
        except KeyError as ke:
            logger.error(f"KeyError 内容报错{ke}, 报错内容为{PythonCode}")
            res = "-"
        except:
            res = "-"
        return res

    def replace_prompt(self, columns, data, index, query, **kwargs):
        """
        加载prompt并替换关键位置,{{Inputs}}
        """
        inputs = {
            "columns": columns,
            "data": data,
            "index": index,
            "Query": query
        }
        if kwargs.get("SubQueries"):
            inputs.update(kwargs)
        
        return BaseAgent.build_first_prompt(system_prompt=self.SystemPrompt,user_prompt=self.UserPrompt.replace("{{Inputs}}", "```python\ntable_data="+json.dumps(inputs,indent=2)+"\ntable_df = pd.DataFrame(table_data['data'], columns=table_data['columns'])\n```"))
    
    def get_code_result(self, dataDict, agent3_response):
        table_string = json.dumps(dataDict["table"],ensure_ascii=False,indent=2).strip()

        prefix = [
            "import pandas as pd",
            f"table_data={table_string}",
            "table_df=pd.DataFrame(table_data['data'], columns=table_data['columns'])",
            agent3_response
        ]

        comb_code = "\n".join(prefix)
        return BaseAgent.exec_code("table_data={table_string}\n".format(table_string=dataDict["table"])+agent3_response)

agent4 = Agent4()