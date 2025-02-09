from dao.Agents.BaseAgent import BaseAgent
from dao.LLMCaller import llmCaller
from dao.Agents.LlmValidAgent import llmValidAgent
import json
import os
from loguru import logger
from dao.Tools import Tools
import pandas as pd
from contextlib import redirect_stdout
import io
class Agent3:
    def __init__(self):
        self.SystemPrompt="You are a data analyst proficient in Python. Your task is to write executable Python code to analyze the table and then answer questions."
        self.UserPrompt=BaseAgent.load_prompt("./prompts/Agent3PromptSimple.txt")
        # self.UserPrompt=BaseAgent.load_prompt("./prompts/Agent3PromptQueryDecomposition.txt")

        self.new_system_prompt = BaseAgent.load_prompt("./prompts/Agent3PromptSimpleSystem.txt")
        self.new_user_prompt = BaseAgent.load_prompt("./prompts/Agent3PromptSimpleUser.txt")
        
        self.new_system_prompt3a = BaseAgent.load_prompt("./prompts/Agent3PromptSimpleSystem3a.txt")
        self.new_user_prompt3a = BaseAgent.load_prompt("./prompts/Agent3PromptSimpleUser3a.txt")
        self.test_save_sample_data = 1

    def infer_with_tool(self, dataDict, model_name, engine="hf", tool=False, **kwagrs):
        # logger = logging.getLogger(os.environ.get('logger_name'))
        dataDict["agent3"] = {}
        instruction=self.replace_prompt(
            columns=kwagrs["table"]["columns"],
            data=kwagrs["table"]["data"],
            Query=kwagrs["query"],
            index=[
                i for i in range(len(kwagrs["table"]["data"]))
            ],
            QuerySteps=kwagrs["QuerySteps"]
        )
        response = llmCaller.infer(instruction, model_name, tool=tool, engine=engine)

        result = []
        if tool is True:
            response = Tools.try_parse_tool_calls(content=response, table_data=kwagrs["table"])
            result.extend([i["result"] for i in response if i["status"]=="success"])
        if result==[]:
            result="-"
        return " ".join(result)

    def raw_infer(self, dataDict, model_name, engine="hf", tool=False, **kwagrs):
        instruction = [
            {"role": "system", "content": self.SystemPrompt},
            {"role": "user", "content": dataDict["instruction"]}
        ]
        response = llmCaller.infer(instruction, model_name, engine=engine).replace("\\n","\n")
        # print(response)
        try:
            python_code = BaseAgent.extract_python(response)

            df_str = "df="+dataDict["table"].strip()+"\ndf = pd.DataFrame(df['data'], columns=df['columns'])\n"
            codes = python_code.replace("df = pd.read_csv('table.csv')",df_str).strip().replace("\\n","\n")
            response = BaseAgent.exec_code(codes.strip()).strip()
            print(f"回答正确：{response}")
        except Exception as e:
            print(f"错误内容为:{e}")
            print("回答错误：{response}".format(response=dataDict["id"]))
            response = "-"
        return response

    def infer_with_QueryDecomposition(self, dataDict, model_name, engine="hf", tool=False, **kwagrs):
        # logger = logging.getLogger(os.environ.get('logger_name'))
        dataDict["agent3"] = {}
        instruction=self.replace_prompt(
            columns=kwagrs["table"]["columns"],
            data=kwagrs["table"]["data"],
            Query=kwagrs["query"],
            index=[i for i in range(len(kwagrs["table"]["data"]))],
            QuerySteps=kwagrs["QuerySteps"]
        )
        
        response = llmCaller.infer(instruction, model_name, tool=tool, engine=engine, dataMethod=kwagrs["dataMethod"])
        dataDict["agent3"]["output"] = response
        # print(BaseAgent.extract_json(response))
        PythonCode = BaseAgent.extract_python(response)

        # logger.info(f"Agent3 的提取的代码为：\n{PythonCode}")
        try:
            res = self.get_code_result(dataDict, PythonCode, kwagrs["table"]).strip()
            logger.info(f"Agent3 代码运行结果：\n{res}")
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
        except Exception as e:
            logger.error(f"内容报错{e}, 报错内容为{PythonCode}")
            res = "-"
        dataDict["agent3"]["input"] = instruction
        return res

    def raw_rawLLM(self, dataDict, model_name, engine="hf", tool=False, **kwargs):
        dataDict["agent3"] = {}
        tableDict = json.loads(dataDict["table"]) if isinstance(dataDict["table"], str) else dataDict["table"]
        tabledf = pd.DataFrame(data=tableDict["data"],columns=tableDict["columns"])
        question = dataDict["Question"]
        # TableTitle = dataDict.get("TabeTitle", None)
        # TableTitle = f"{TableTitle}\n" if TableTitle else None
        if model_name == "TableGPT2-7B":
            prompt = self.build_tablegpt_prompt(tabledf=tabledf, question=question, dataset_name=kwargs["dataset_name"])
            
            response = llmCaller.infer(prompt, model_name, tool=tool, engine=engine, dataMethod=kwargs["dataMethod"])
            
            dataDict["agent3"]["RawOutput"] = response
            PythonCode = BaseAgent.extract_python(response)
            prefix = [
                "import pandas as pd",
                "import numpy as np",
                f"table_data={json.dumps(tableDict, ensure_ascii=False,indent=2)}".replace("null","None"),
                "df = pd.DataFrame(table_data['data'], columns=table_data['columns'])",
                PythonCode
            ]

            PythonCode = "\n".join(prefix)

            dataDict["agent3"]["ExtractCode"] = PythonCode
            try:
                res = self.get_code_result(dataDict, PythonCode, tableDict).strip()
                logger.info(f"Agent3 代码运行结果：\n{res}")
            except Exception as e:
                logger.error(f"代码运行报错为:\n{e}")
                res = "-"
        elif model_name == "tablellama:7b":
            tableString = self.dataframe_to_custom_string(tabledf)
            # TableTitle = dataDict.get("TabeTitle", None)
            # TableTitle = f"{TableTitle}\n" if TableTitle else None
            if kwargs["dataset_name"] != "FinQa":
                Prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nThis is a table QA task. The goal of this task is to answer the question given the table.\n\n### Input:\n{input}\n\n### Question:\n{question}\n\n### Response:".format(input=tableString,question=kwargs["question"])
            else:
                Prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nThis is a table QA task. The goal of this task is to answer the question given the table.\n\n### Input:\n{input}\n\n### Table Title:{TableTitle}\n\n### Question:\n{question}\n\n### Response:".format(input=tableString,question=kwargs["question"])
            # message = [
            #     {"role": "user", "content": Prompt}
            # ]
            res = llmCaller.infer(Prompt, model_name, engine=engine, dataMethod=kwargs["dataMethod"], tabledf=tabledf)
        elif model_name == "TIGER-Lab/StructLM-7B":
            prompt = "[INST] <<SYS>>\nYou are an AI assistant that specializes in analyzing and reasoning over structured information. You will be given a task, optionally with some structured knowledge input. Your answer must strictly adhere to the output format, if specified.\n<</SYS>>\n{instruction} [/INST]"
            instruction = "Use the information in the following table to solve the problem, choose between the choices if they are provided. table:\n\n{StructLMTable}\n\nquestion:\n\n{QuestionString}".format(StructLMTable="test",QuestionString=question)
        elif model_name == "neulab/omnitab-large-finetuned-wtq":
            res = llmCaller.infer([], model_name, engine=engine, dataMethod=kwargs["dataMethod"], tabledf=tabledf, question=question)
        elif model_name == "microsoft/tapex-large-finetuned-wtq":
            res = llmCaller.infer([], model_name, engine=engine, dataMethod=kwargs["dataMethod"], tabledf=tabledf, question=question)
        elif model_name == "google/tapas-large-finetuned-wtq":
            res = llmCaller.infer([], model_name, engine=engine, dataMethod=kwargs["dataMethod"], tabledf=tabledf, question=question)
        elif model_name == "RUCKBReasoning/TableLLM-13b":
            # prompt = self.build_tablegpt_prompt(tabledf=tabledf, question=question)
            response = llmCaller.infer("", model_name, tool=tool, engine=engine, dataMethod=kwargs["dataMethod"], tabledf=tabledf, question=question)
            if kwargs["dataMethod"] == "PoT":
                dataDict["agent3"]["RawOutput"] = response
                PythonCode = response
                prefix = [
                    "import pandas as pd",
                    "import numpy as np",
                    f"table_data={json.dumps(tableDict, ensure_ascii=False,indent=2)}".replace("null","None"),
                    PythonCode
                ]

                PythonCode = "\n".join(prefix)

                dataDict["agent3"]["ExtractCode"] = PythonCode.replace("df = pd.read_csv('data.csv')", "df = pd.DataFrame(table_data['data'], columns=table_data['columns'])").strip().replace("\\n","\n")
                try:
                    res = self.get_code_result(dataDict, dataDict["agent3"]["ExtractCode"], tableDict).strip()
                    logger.info(f"Agent3 代码运行结果：\n{res}")
                except Exception as e:
                    logger.error(f"代码运行报错为:\n{e}")
                    res = "-"
            elif kwargs["dataMethod"] == "DP":
                res = response
        return res

    def dataframe_to_custom_string(self,df):
        """
        Converts a DataFrame into a custom structured string with column names included.

        Parameters:
            df (pd.DataFrame): DataFrame with any columns.

        Returns:
            str: The formatted string with the custom structure.
        """
        result = []
        header = "[TAB] col: | " + " | ".join(df.columns) + " |"
        result.append(header)
        for _, row in df.iterrows():
            formatted_row = "[SEP] | " + " | ".join([str(row[col]) for col in df.columns]) + " |"
            result.append(formatted_row)
        return " ".join(result)

    def build_tablellm_prompt(self, **kwargs):
        table_df, question = kwargs["tabledf"], kwargs["question"]
        example_prompt_template = """[INST]Below are the first few lines of a CSV file. You need to write a Python program to solve the provided question.
        
        Header and first few lines of CSV file:
        {csv_data}

        Question: {question}[/INST]
        """
        prompt = example_prompt_template.format(
            csv_data=table_df.head(5).to_csv(index=False),
            question=question,
        )

        messages = [
            {"role": "user", "content": prompt}
        ]
        return messages

    def build_tablegpt_prompt(self,**kwargs):
        table_df, question = kwargs["tabledf"], kwargs["question"]
        if kwargs["dataset_name"] == "FinQa":
            TableTitle = kwargs.get("TabeTitle", None)
            TableTitle = f"\nTable Title:\n{TableTitle}\n" if TableTitle else None
            example_prompt_template = """Given access to several pandas dataframes, write the Python code to answer the user's question.

            /*
            "{var_name}.head(5).to_string(index=False)" as follows:
            {df_info}
            */
            {TableTitle}

            Question: {user_question}
            """
            prompt = example_prompt_template.format(
                var_name="df",
                df_info=table_df.head(5).to_string(index=False),
                user_question=question,
                TableTitle=TableTitle
            )
        else:

            example_prompt_template = """Given access to several pandas dataframes, write the Python code to answer the user's question.

            /*
            "{var_name}.head(5).to_string(index=False)" as follows:
            {df_info}
            */

            Question: {user_question}
            """
            prompt = example_prompt_template.format(
                var_name="df",
                df_info=table_df.head(5).to_string(index=False),
                user_question=question,
            )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        return messages

    def _build_3_prompt(self, questions: list, table: dict, max_sample_rows: int = 5) -> list:
        truncated_data = table['data'][:max_sample_rows]
        compelete_table_data = json.dumps(table['data'], indent=2, ensure_ascii=False)
        system_prompt = f"""You are a table data analyst. Generate Python code using pandas to solve multiple questions.
Follow these STRICT rules:
1. Wrap ALL generated code in ```python markdown blocks
2. For EACH question, generate code followed by print(question + result)
3. Use ONLY these columns: {json.dumps(table['columns'])}
4. Sample data (first {max_sample_rows} rows): 
{json.dumps(truncated_data, indent=2, ensure_ascii=False)}
5. CRITICAL REQUIREMENTS:
- Use existing DataFrame `table_df` (ALREADY LOADED)
- Never create/load DataFrame
- Match exact column names
- Maintain code execution 
- Ignore missing values

**Pandas Version-Specific Rules (2.2.3)**:
1. pd.to_numeric(args) - For type conversion, args: list, tuple, 1-d array, or Series.
2. .loc[] - Label-based indexing
3. .iloc[] - Position-based indexing

**Pandas Deprecated methods**:
1. .ix[]

Example Response Format:
```python
# Question 1: Average salary
result = table_df['salary'].mean()
print(f"1. Average salary: {{result}}")

# Question 2: Department distribution
dept_counts = table_df['department'].value_counts()
print(f"2. Department distribution: {{dept_counts}}")
```"""

        user_prompt = f"Complete table data:\n{compelete_table_data}\nGenerate Python code for questions:\n" + "\n".join(
            [f"{i+1}. {q}" for i, q in enumerate(questions)]
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    # def _extract_code(self, response: str) -> str:
    #     """Extract code block from model response"""
    #     if "```python" in response:
    #         return response.split("```python")[1].split("```")[0]
    #     return response

    def _validate_code(self, code: str, table: dict) -> dict:
        """Execute and validate generated code"""
        df = pd.DataFrame(table['data'], columns=table['columns'])
        try:
            exec_scope = {'table_df': df.copy(), 'pd': pd}
            result = "-"
            with redirect_stdout(io.StringIO()) as output:
                exec(code, exec_scope)
                result = output.getvalue()
            # result = exec_scope.get('result', "-")
            logger.info(f"代码执行成功，结果为: {result}")
            return {
                "executable": True,
                "result": result
            }
        except Exception as e:
            logger.error(f"Code execute error: {e}")
            return {"executable": False, "result": str(e)}

    def infer_with_3_1(self, dataDict, model_name, engine="hf", **kwagrs):
        instruction = self._build_3_prompt(questions=kwagrs["queries"], table=kwagrs["table"])
        # logger.debug(f"instruciton is: \n[\n{instruction}\n]")
        response = llmCaller.infer(instruction, model_name, engine=engine, dataMethod=kwagrs["dataMethod"])
        logger.debug(f"response is: \n[\n{response}\n]")
        code = BaseAgent.extract_python(response)
        logger.debug(f"code is: \n[\n{code}\n]")
        validation = self._validate_code(code, kwagrs["table"])

        return validation["result"]

    def infer_with_3a(self, dataDict, model_name, engine="hf", tool=False, **kwagrs):
        dataDict["agent3"] = {}
        instruction = self.build_3a_prompt(
            columns=kwagrs["table"]["columns"],
            data=kwagrs["table"]["data"],
            Queries=kwagrs["queries"],
            index=[i for i in range(len(kwagrs["table"]["data"]))],
            TabeTitle=dataDict.get("TableTitle", None)
        )
        response = llmCaller.infer(instruction, model_name, tool=tool, engine=engine, dataMethod=kwagrs["dataMethod"])
        dataDict["agent3"]["output"] = response
        # print(BaseAgent.extract_json(response))
        PythonCode = BaseAgent.extract_python(response)

        # logger.info(f"Agent3 的提取的代码为：\n{PythonCode}")
        try:
            res = self.get_code_result(dataDict, PythonCode, kwagrs["table"]).strip()
            logger.info(f"Agent3 代码运行结果：\n{res}")
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
        except Exception as e:
            logger.error(f"内容报错{e}, 报错内容为{PythonCode}")
            res = "-"
        dataDict["agent3"]["input"] = instruction
        return res
    def infer_with_3(self, dataDict, model_name, engine="hf", tool=False, **kwagrs):
        return self.infer_with_3a(dataDict, model_name, engine=engine, **kwagrs)


    def infer(self, dataDict, model_name, engine="hf", tool=False, **kwagrs):
        # logger = logging.getLogger(os.environ.get('logger_name'))
        dataDict["agent3"] = {}
        instruction=self.replace_prompt(
            columns=kwagrs["table"]["columns"],
            data=kwagrs["table"]["data"],
            Queries=kwagrs["queries"],
            index=[i for i in range(len(kwagrs["table"]["data"]))]
        )
        # instruction = self.build_new_prompt(
        #     columns=kwagrs["table"]["columns"],
        #     data=kwagrs["table"]["data"],
        #     Queries=kwagrs["queries"],
        #     index=[i for i in range(len(kwagrs["table"]["data"]))],
        #     TabeTitle=dataDict.get("TableTitle", None)
        # )
        # if self.test_save_sample_data:
        #     with open("./test_sampe_agent3.json", "w", encoding="utf-8") as f:
        #         json.dump(instruction, f, ensure_ascii=False, indent=4)
        #     self.test_save_sample_data = 0
        
        response = llmCaller.infer(instruction, model_name, tool=tool, engine=engine, dataMethod=kwagrs["dataMethod"])
        dataDict["agent3"]["output"] = response
        # print(BaseAgent.extract_json(response))
        PythonCode = BaseAgent.extract_python(response)

        # logger.info(f"Agent3 的提取的代码为：\n{PythonCode}")
        try:
            res = self.get_code_result(dataDict, PythonCode, kwagrs["table"]).strip()
            logger.info(f"Agent3 代码运行结果：\n{res}")
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
        except Exception as e:
            logger.error(f"内容报错{e}, 报错内容为{PythonCode}")
            res = "-"
        dataDict["agent3"]["input"] = instruction
        return res

    def build_new_prompt(self, TableTitle=None, **kwargs):
        

        TableTitle = kwargs.get("TabeTitle", None)
        TableTitle = f"\nTable Title:\n{TableTitle}\n" if TableTitle else None
        sample_data = kwargs['data'][:3] if len(kwargs['data']) >= 3 else kwargs['data']
        TableStructure = TableTitle + f"Table Columns:\n{json.dumps(kwargs['columns'], indent=2)}" if TableTitle else f"Table Columns:\n{json.dumps(kwargs['columns'], indent=2)}"
        system_prompt = self.new_system_prompt.replace("{{TableStructure}}", TableStructure).replace("{{SampleData}}", json.dumps(sample_data, indent=2))
        
        if kwargs.get("TabeTitle"):
            del kwargs["TabeTitle"]
        inputs = "table_data={inputs}\n".format(inputs=json.dumps(kwargs, indent=2, ensure_ascii=False))+"\ntable_df = pd.DataFrame(table_data['data'], columns=table_data['columns'])"
        user_prompt = self.new_user_prompt.replace("{{Inputs}}", inputs)
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    def build_3a_prompt(self, TableTitle=None, **kwargs):
        inputs = "table_data={inputs}\n".format(inputs=json.dumps(kwargs, indent=2, ensure_ascii=False))+"\ntable_df = pd.DataFrame(table_data['data'], columns=table_data['columns'])"
        TableTitle = kwargs.get("TabeTitle", None)
        TableTitle = f"\nTable Title:\n{TableTitle}\n" if TableTitle else None
        sample_data = kwargs['data'][:3] if len(kwargs['data']) >= 3 else kwargs['data']
        TableStructure = TableTitle + f"Table Columns:\n{json.dumps(kwargs['columns'], indent=2)}" if TableTitle else f"Table Columns:\n{json.dumps(kwargs['columns'], indent=2)}"
        system_prompt = self.new_system_prompt3a.replace("{{TableStructure}}", TableStructure).replace("{{SampleData}}", json.dumps(sample_data, indent=2))
        user_prompt = self.new_user_prompt3a.replace("{{Inputs}}", inputs)
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def replace_prompt(self, QuerySteps=None, **kwargs):
        """
        加载prompt并替换关键位置,{{Inputs}}
        """
        inputs = "```python\ntable_data={inputs}\n".format(inputs=json.dumps(kwargs, indent=2, ensure_ascii=False))+"\ntable_df = pd.DataFrame(table_data['data'], columns=table_data['columns'])```"
        if QuerySteps:
            return BaseAgent.build_first_prompt(system_prompt=self.SystemPrompt,user_prompt=self.UserPrompt.replace("{{Inputs}}", inputs).replace("{{QuerySteps}}",QuerySteps))
        else:
            return BaseAgent.build_first_prompt(system_prompt=self.SystemPrompt,user_prompt=self.UserPrompt.replace("{{Inputs}}", inputs))
    
    def get_code_result(self, dataDict, agent3_response ,table):
        # table_string = dataDict["table"]

        prefix = [
            "import pandas as pd",
            "import numpy as np",
            f"table_data={json.dumps(table, ensure_ascii=False,indent=2)}".replace("null","None"),
            "table_df = pd.DataFrame(table_data['data'], columns=table_data['columns'])",
            agent3_response
        ]

        comb_code = "\n".join(prefix)

        return BaseAgent.exec_code(comb_code.strip())

agent3 = Agent3()