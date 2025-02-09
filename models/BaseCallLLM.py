from tools import *
import os
import ast
import re

class BaseCallLLM:

    prompt_root = "./prompts"
    QueryReviseSystemPrompt = "You are tasked with interpreting data in a table format and answering the user’s question based on provided structured information."
    QueryRevisePromptPath = os.path.join(prompt_root, "QueryRevisePrompt.txt")

    toolSystemPrompt = "You are a text-processing assistant designed to analyze, compute, and format outputs based on previous model responses."
    toolUserPromptPath = os.path.join(prompt_root, "TableComputePrompt.txt")

    TableCleanSystemPrompt = "Suppose you are an expert in statistical analysis. You will be given a table described in a special format."
    TableCleanPromptPath = os.path.join(prompt_root, "TableCleanPrompt.txt")
    Step2PromptPath = os.path.join(prompt_root, "POTSImple.txt")

    PotSystemPrompt = "You are a data analyst proficient in Python. Your task is to write executable Python code to analyze the table and then answer questions."
    PotPrompt = os.path.join(prompt_root, "POTPrompt.txt")

    def __init__(self):

        self.model = None
        self.tokenizer = None
        self.ollama_ip = ""
        self.prompt_root = "./prompts"

        self.RawSystemPrompt = "You are a table analyst. Your task is to answer questions based on the table content."
        self.RawTCOTPrompt = "The answer should follow the format below:\n[Answer Format]\nFinal Answer: AnswerName1, AnswerName2...\n\nEnsure the final answer format is the last output line and can only be in the \"Final Answer: AnswerName1, AnswerName2...\" form, no other form. Ensure the \"AnswerName\" is a number or entity name, as short as possible, without any explanation.\n\nLet's think step by step and then give the final answer to the question.\n\nRead the table below in {TFormat} format:\n[TABLE] \n{TableString}\n\nLet's get start!\nQuestion: {QuestionString}"
        
        self.RawPOTSystemPrompt = "You are a data analyst proficient in Python. Your task is to write executable Python code to analyze the table and then answer questions."
        self.RawPOTPrompt = "[Guidelines]\nYou should act following requirements below:\n1. based on the question, write out your analytical approach, and then write Python code according to this approach.\n2. The code needs to be concise and easy to understand, and if necessary, add comments for clarification.\n3. Code blocks need to strictly start with ```python and end with ```\n4. Your analysis must be based entirely on the above data. If the user's question is not related to data analysis, please politely refuse.\n5. You need to generate executable code. If there are results to be presented, please use the print function; if there are charts, please use the matplotlib library to draw them.\n6. Ensure to load the table with command ```df = pd.read_csv('table.csv')```\n\n\nThe answer should follow the format below:\n[Answer Format]\nFinal Answer: AnswerName1, AnswerName2...\n\nEnsure the final answer format is the last output line and can only be in the \"Final Answer: AnswerName1, AnswerName2...\" form, no other form. Ensure the \"AnswerName\" is a number or entity name, as short as possible, without any explanation.\n\n\nLet's think step by step and then generate python code to analyze table and present the final answer to the question.\n\nRead the table below in JSON format:\n[TABLE] \n{TableString}\n\nLet's get start!\nQuestion: {QuestionString}"
        
        self.ReviseSystemPrompt = "You are a table analyst, responsible for reviewing the reasoning process, cell selection, and mathematical expression provided by the previous agent. Your task is to ensure expression's accuracy and consistency with the question requirements."
        self.RevisePrompt = "If the answer is correct: Simply re-state the final expression and answer as provided, without modification (The mathematical expression is: [Expression] (Short explain for expression)). \n\nIf there are any errors: Clearly explain the mistake(s) made by the previous agent. \nProvide the corrected mathematical expression along with a brief explanation of how it addresses the identified errors.\n\nRead the table below in JSON format:\n{TableString}.\n\nQuestion: \"{QuestionString}\"\n\nLast Agent Response: {LastAgent}"
        self.ReviseStep2SystemPrompt = "You are a table analyst. Your task is to fill in answer based on the table content."
        self.ReviseTCOTPrompt = "Read the table below in JSON format:\n[TABLE] \n{TableString}\n\nAnswer:\n{QuestionString}"

        self.GenCleanTableSystem1Prompt = "You are a table analyst. Your task is to identify relevant ColumnName based on the table column names from users' query."
        self.GenCleanTable1Prompt = "Follow this process step by step: \n1.Analyze the user's query to determine which column names from the table are relevant.\n2.Select only those column names that directly relate to the question asked.\nIdentify columns where specific cell values mentioned in the query might appear (e.g., “year” for date-related questions like “When is my birthday?”).\n\nThe answer should include conclusion in the end, the conclusion format below:\nRelevant column names: ColumnName1, ColumnName2...\n\nEnsure the Relevant column names' format is the last output line and can only be in the \"\n\nRelevant column names: ColumnName1, ColumnName2,...\" form, no other form. Ensure the \"ColumnName\" is one of table column names, without any explanation.\n\nRead the table column names below in JSON format:\nTable Column Names: {ColumnNames}\nQuestion: {QuestionString}\n\nLet's think step by step and then give the relevant column names."

        self.SchemaRetrievalSystemPromptFromTableRAG = "Given a table with all column keys and several example cells: \"Table with all column keys and several example cells\", I want to answer a question: \"Question String\". Since I cannot view the table directly, please suggest some column keys that might contain the necessary data to answer this question. Please answer with a list of column keys in JSON format without any additional explanation. Make sure that keys must be one of column keys of input table with json format. Example: [\"key1\", \"key2\", \"key3\"]"
        self.SchemaRetrievalPromptFromTableRAG = "Table with all columns and several example cells: \"{TableExamplesWithColumns}\", column keys: \"{ColumnKeys}\".\n\nQuestion String: \"{QuestionString}\""
        self.CellRetrievalSystemPromptFromTableRAG = "Given a table with all columns: \"Table with all columns\", I want to answer a question: \"Question String\". Since I cannot view the table directly, Please extract filter criteria which might help answer the question and filter and search table content. The filter criteria should be sub string which appear in question. The filter criteria should be contained in the question. Please answer with a list of filter criteria in JSON format without any additional explanation. Example: [\"criteria1\", \"criteria2\", \"criteria3\"]."
        self.CellRetrievalPromptFromTableRAG = "Table with all columns: \"{TableExamplesWithColumns}\".\n\nQuestion String: \"{QuestionString}\""


        # self.TableRAGSystemPromptWithEquation = "You are a table analyst. Your task is to identify relevant cells from the table in json format based on the question and answer the question. You need to convert the final result into mathematical expressions for computation. The result of mathematical expression is the answer to question."
        self.TableRAGSystemPromptWithEquation = "You are a table analyst. Your task is to answer questions based on the table content. First, generate the complete mathematical expression that represents the calculation needed to answer the question. Ensure that all calculations (addition, subtraction, multiplication, division, comparision, etc.) are included in the final expression and use parentheses to indicate the order of operations, without calculating intermediate processes or results. Second, explain how you derived the mathematical expression by identifying the relevant cells based on the constraints specified in the question. At the end, on a new line, provide the complete mathematical expression in the following format: The mathematical expression is: [the complete mathematical expression] (The short explain for the expression). Make sure that the mathematical expression consists of numerical values in the table rather than categorical values."
        self.TableRAGTCOTPromptWithEquation = "Here are tables content and cell retrieval queries retrieved from the table. Read the table below in JSON format:\n{TableString}.\n\nCell Retrieval Queries: {CellRetrievalQueries}\n\nQuestion: \"{QuestionString}\""
        self.TableRAGTCOTPromptWithEquationNoRetrieval = "Read the table below in JSON format:\n{TableString}.\n\nQuestion: \"{QuestionString}\""
        self.TableRAGSystemPromptWithTool = "You are a table analyst. Your task is to answer questions based on the table content in json format. First, identify the cells relevant to the question step by step. According to relevant cells, create a mathematical expression that can be directly computed. For example, if the question is to compute relevant cells' average value, format the result as (1+2+3)/4(average of relevant values). Finally, include the mathematical expression at the end: \"\n\nThe mathematical expression is: 1+2+3+4(Sum up relevant values)\". Ensure The mathematical expression is the last output line and it is the answer to the question."
        self.TableRAGTCOTPromptWithTool = "Here are tables content and cell retrieval queries retrieved from the table. Read the table below in JSON format:\n{TableString}.\n\nCell Retrieval Queries: {CellRetrievalQueries}\n\nQuestion: \"{QuestionString}\""
        self.TableRAGSystemPrompt = "You are a table analyst. Your task is to answer questions based on the table content in json format. The answer should include the format in your conclusion:\n\"\n\nThe final Answer is: Answer1, Answer2\".\n\nEnsure the final answer format is the last output line.\n\nLet's think step by step and then give the final answer to the question."
        self.TableRAGTCOTPrompt = "Here are tables content and cell retrieval queries retrieved from the table. Read the table below in JSON format:\n{TableString}.\n\nCell Retrieval Queries: {CellRetrievalQueries}\n\nQuestion: \"{QuestionString}\""
        # self.CellRetrievalSystemPromptFromTableRAG = "Given a large table, I want to answer a question: \"Question String\". Please extract some keywords which might appear in the table cells and help answer the question. The keywords should be categorical values rather than numerical values. The keywords should be contained in the question. Please answer with a list of keywords in JSON format without any additional explanation. Example: [\"keyword1\", \"keyword2\", \"keyword3\"]"
        # self.CellRetrievalPromptFromTableRAG = "Question String: \"{QuestionString}\""

        self.OnlyToolFunctionSystemPrompt = "You are a table analyst. Your task is to answer questions by mathematical expression based on the table content. First, identify the cells relevant to the question step by step. According to relevant cells, create a mathematical expression that can be directly computed. For example, if the question is to compute relevant cells' average value, format the result as \"(1+2+3)/4(average of relevant values)\". Finally, include the mathematical expression at the end: \"\n\nThe mathematical expression is: 1+2+3+4(Sum up relevant values)\". Ensure The mathematical expression is the last output line and it is the answer to the question."
        self.OnlyToolFunctionUserPrompt = "Table with all columns:\n{TableString}\n\nQuestion:\n{QuestionString}\n\nLet us think step by step."


        
        self.TableCleanSystemPrompt1 = "Suppose you are an expert in statistical analysis. You will be given a table described in a special format. Your task is to extract a sub-table relevant to the question, ensuring that the sub-table matches the answer to the question."
        self.TableCleanPrompt1_DG = os.path.join(self.prompt_root, "TableCleanPrompt1_DG.txt")

        self.TableCleanSystemPrompt1_TOOL = "You are a table analyst. Your task is to answer questions based on the table content."
        self.TableCleanPrompt1_TOOL = os.path.join(self.prompt_root, "TableTOOLPrompt1_TOOL.txt")
        self.TableCleanPrompt1_TOOL_HTML = os.path.join(self.prompt_root, "TableTOOLPrompt1_HTML.txt")

        self.TableCleanSystemPrompt1_DP = "You are a table analyst. Your task is to answer questions based on the sub-table content."
        self.TableCleanPrompt1_DP = os.path.join(self.prompt_root, "TableCleanPrompt1_DP.txt")

        self.TableComputeSystemPrompt = "You are a text-processing assistant designed to analyze, compute, and format outputs based on previous model responses."
        self.TableComputePrompt = os.path.join(self.prompt_root, "TableComputePrompt.txt")

    @staticmethod
    def extract_python(inputs):
        """
        构建多层嵌套 prompt
        """
        matcch_result = re.search(r"```python(.*?)```", inputs, re.IGNORECASE | re.DOTALL)
        # print(matcch_result)
        if matcch_result is None:
            matcch_result = re.search(r"```python(.*)", inputs, re.IGNORECASE | re.DOTALL)
            if matcch_result is None:
                return None
            else:
                return matcch_result.group(1).strip("```").strip()
        else:
            return matcch_result.group(1).strip("```").strip()

    def _extract_final_output(self, input_text, final_sentence):
        last_sentence = input_text.split("\n")[-1].strip()
        pattern = f"{final_sentence}(.*)$"
        search_res = re.search(pattern, last_sentence, re.IGNORECASE)
        if search_res is None:
            return ""
        else:
            return search_res.group(1).strip()

    @staticmethod
    def calculate_mathematical_expression(expression: str):
        """
        A function takes a complete mathematical expression as input and returns the calculated result. Mathematical expression is "1+2+3" or "(45+21+30) / 3". Do not use "avg(45,21,30)". If the expression is python function, you must convert it as Mathematical expression, such as "avg(45,21,30)" as "(45 + 21 + 30) / 3"
        
        Args:
            expression: A valid mathematical expression string to calculate.
        """
        return eval(expression)
    
    @staticmethod
    def load_prompt_from_txt(prompt_path: str):
        """
        从txt文件中加载prompt文本
        """
        inputs = ""

        with open(prompt_path, 'r', encoding='utf-8') as file:
            inputs = file.read()

        return inputs

    @staticmethod
    def extract_result(text, prefix="The Final Answer:"):
        text = text.split("\n")[-1].strip()
        pattern = f"{prefix}.*(\[.*\])"

        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                # text = ":".join(match.group(1).split(":")[1:])
                answer_list = ast.literal_eval(match.group(1).strip())
            except (SyntaxError, ValueError):
                print("解析报错，报错内容为：")
                print(match.group(1).strip())
                answer_list = ["-"]  # 当解析失败时返回一个默认值
        else:
            answer_list = ["-"]
        return answer_list
    
    @staticmethod
    def build_first_prompt(system_prompt, user_prompt):
        """
        构建多层嵌套 prompt
        """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    @staticmethod
    def clean_final_answers(text):
        """
        去除多余的括号解释。

        Examples:
          text = The final answers: ['1572 (Total number of locomotives introduced between 1867 and 1873, considering only the '2 - 4 - 0' and '0 - 6 - 0' types)'].
        """

        matchs = re.sub(r"\(.*\)", "", text, re.IGNORECASE | re.DOTALL)
        if matchs is None or matchs == "":
            return text
        else:
            return matchs
        
    @staticmethod
    def build_QueryRevise_prompt(text):
        """
        去除多余的括号解释。

        Examples:
          text = The final answers: ['1572 (Total number of locomotives introduced between 1867 and 1873, considering only the '2 - 4 - 0' and '0 - 6 - 0' types)'].
        """
        QueryRevisePrompt = BaseCallLLM.load_prompt_from_txt(BaseCallLLM.QueryRevisePromptPath)

        return QueryRevisePrompt.replace("{{InputString}}", json.dumps(text, indent=2))
    
    @staticmethod
    def build_TableClean_prompt(text):
        """
        去除多余的括号解释。

        Examples:
          text = The final answers: ['1572 (Total number of locomotives introduced between 1867 and 1873, considering only the '2 - 4 - 0' and '0 - 6 - 0' types)'].
        """
        QueryRevisePrompt = BaseCallLLM.load_prompt_from_txt(BaseCallLLM.TableCleanPromptPath)

        return QueryRevisePrompt.replace("{{InputString}}", json.dumps(text, indent=2))

    @staticmethod
    def parse_json_response(text):
        match = re.search(r"```json(.*)```", text, re.IGNORECASE | re.DOTALL)
        if match is None:
            return text
        else:
            return match.group(1).strip()
