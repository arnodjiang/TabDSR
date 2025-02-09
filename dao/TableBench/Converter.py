import json


dataset_map = {
    "Test": "./dao/Test/ModifiedRawQTabData.json"
}

POTAssistantPrompt="You are a data analyst proficient in Python. Your task is to write executable Python code to analyze the table and then answer questions.\n\n[Guidelines]\nYou should act following requirements below:\n1. based on the question, write out your analytical approach, and then write Python code according to this approach.\n2. The code needs to be concise and easy to understand, and if necessary, add comments for clarification.\n3. Code blocks need to strictly start with ```python and end with ```\n4. Your analysis must be based entirely on the above data. If the user's question is not related to data analysis, please politely refuse.\n5. You need to generate executable code. If there are results to be presented, please use the print function; if there are charts, please use the matplotlib library to draw them.\n6. Ensure to load the table with command ```df = pd.read_csv('table.csv')```\n\n\nThe answer should follow the format below:\n[Answer Format]\nFinal Answer: AnswerName1, AnswerName2...\n\nEnsure the final answer format is the last output line and can only be in the \"Final Answer: AnswerName1, AnswerName2...\" form, no other form. Ensure the \"AnswerName\" is a number or entity name, as short as possible, without any explanation.\n\n\nLet's think step by step and then generate python code to analyze table and present the final answer to the question.\n\nRead the table below in JSON format:\n[TABLE] \n{tableString}\n\nLet's get start!\nQuestion: {questionString}"
DPAssistantPrompt="Your task is to answer questions based on the table content.\n\n\nThe answer should follow the format below:\n[Answer Format]\nFinal Answer: AnswerName1, AnswerName2...\n\nEnsure the final answer format is the last output line and can only be in the \"Final Answer: AnswerName1, AnswerName2...\" form, no other form. Ensure the \"AnswerName\" is a number or entity name, as short as possible, without any explanation.\n\n\nGive the final answer to the question directly without any explanation.\n\nRead the table below in JSON format:\n[TABLE] \n{tableString}\n\nLet's get start!\nQuestion: {questionString}\n"
SCoTAssistantPrompt="You are a table analyst. Your task is to utilize the Python package 'pandas' to analyze the table and then answer questions.\n\n[Guidelines]\nYou should act in following patterns step by step to analyze the table and then give the final answer:\n[Action Patterns]\nThought: You should always think about what to do to interact with Python code base on Result\nAction: the action can **ONLY** be single line python code\nResult: Simulate the result of the execution of the python code in Action, analyse that result and decide whether to continue or not \n(This thought/Action/Result can repeat N times) \n\n\nThe answer should follow the format below:\n[Answer Format]\nFinal Answer: AnswerName1, AnswerName2...\n\nEnsure the final answer format is the last output line and can only be in the \"Final Answer: AnswerName1, AnswerName2...\" form, no other form. Ensure the \"AnswerName\" is a number or entity name, as short as possible, without any explanation.\n\n\nLet's think step by step and then give the final answer to the question. \nEnsure to have a concluding thought that verifies the table, observations and the question before giving the final answer. \n\nRead the table below in JSON format:\n[TABLE] \n{tableString}\n\nLet's get start!\nQuestion: {questionString}"
TCoTAssistantPrompt="You are a table analyst. Your task is to answer questions based on the table content.\n\n\nThe answer should follow the format below:\n[Answer Format]\nFinal Answer: AnswerName1, AnswerName2...\n\nEnsure the final answer format is the last output line and can only be in the \"Final Answer: AnswerName1, AnswerName2...\" form, no other form. Ensure the \"AnswerName\" is a number or entity name, as short as possible, without any explanation.\n\n\nLet's think step by step and then give the final answer to the question.\n\nRead the table below in JSON format:\n[TABLE] \n{tableString}\n\nLet's get start!\nQuestion: {questionString}"

PromptMap = {
    "DP": DPAssistantPrompt,
    "PoT": POTAssistantPrompt,
    "SCoT": SCoTAssistantPrompt,
    "TCoT": TCoTAssistantPrompt
}

class TestConverter:
    @staticmethod
    def converter(dataList, agent_mode, tablebenchMode):
        table = dataList["table"]
        question = dataList["GeneratedQuery"]["Queries"][0]
        answer = ", ".join([i["Answer"] for i in dataList["GeneratedQuery"]["SubQueries"]])
        instruction = PromptMap[tablebenchMode].format(tableString=json.dumps(table, ensure_ascii=False),questionString=question)
        print(dataList)
        toItm = {
            "id": dataList["id"],
            "qtype": "NumericalReasoning",
            "instruction": instruction,
            "instruction_type": tablebenchMode,
            "table": json.dumps(table, ensure_ascii=False),
            "question": question,
            "answer": answer,
            "answerList": [i["Answer"] for i in dataList["GeneratedQuery"]["SubQueries"]]
        }
        return toItm

if __name__ == "__main__":
    dataset_name = "Test"
    toPath = "./dao/Test/QTab.jsonl"

    Todata = []
    with open(dataset_map[dataset_name], 'r', encoding='utf-8') as file:
        data = json.load(file)

    for dataItm in data:
        print(dataItm)
        Todata.append(TestConverter.converter(dataItm, agent_mode="raw", tablebenchMode="DP"))

    with open(toPath, 'w', encoding='utf-8') as file:
        for item in Todata:
            # 将每个字典对象转换为 JSON 字符串并写入文件，每行一个对象
            json_line = json.dumps(item, ensure_ascii=False)
            file.write(json_line + '\n')