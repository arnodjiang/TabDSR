import os
import re

from evautils import *
from TableBench.TableBenchConfig import TableBenchConfig
from TableBench.TableBenchUtils import tablebenchutils


class TableBenchEvaluator:
    def __init__(self) -> None:
        # self.RawDescriptionTemplate = """This table_section_title is '{table_section_title}' and table_section_text is '{table_section_text}', table_page_title is '{table_page_title}'"""
        self.DatasetName = "TableBench"

    def evaluatePipeline(self, model_name="qwen2.5:7b"):
        """
        对数据集评估
        """
        response_result = []
        number = 0
        for dataType, dataMap in TableBenchConfig.RawDataPath.items():
            if dataType in ["DP","TCoT"]:
                with open(dataMap, 'r', encoding='utf-8') as file:
                    for line in file:
                        # 将每一行的JSON字符串解析为字典，并追加到列表中

                        LineData = json.loads(line.strip())
                        if LineData["qtype"] in ["Visualization"]:
                            continue
                        number += 1
                        response = post_request(LineData["instruction"], model=model_name)
                        # TableBenchConfig.InferPrompt["JSON"]["en"]["COT"].format(TableString=LineData["table"],QuestionString=LineData["question"])

                        response_result.append({
                            "DatasetName": self.DatasetName,
                            "Instruction": LineData["instruction"],
                            "Table": LineData["table"],
                            "Question": LineData["question"],
                            "LLMResponse": response,
                            "Type": dataType,
                            "id": LineData["id"],
                            "label": LineData["answer"],
                            "qtype": LineData["qtype"],
                            "qsubtype": LineData["qsubtype"]
                        })
                        with open(f'./{self.DatasetName}/qwen2-5.json', 'w', encoding='utf-8') as f:
                            json.dump(response_result, f, ensure_ascii=False, indent=4)
        print(f"数据集总数量为{number}")

    async def evaluatePipelineWithAsync(self, model_name="qwen2.5"):
        """
        对数据集评估
        """
        response_result = []
        number = 0
        for dataType, dataMap in TableBenchConfig.RawDataPath.items():
            if dataType in ["TCoT"]:
                with open(dataMap, 'r', encoding='utf-8') as file:
                    for line in file:
                        # 将每一行的JSON字符串解析为字典，并追加到列表中

                        LineData = json.loads(line.strip())
                        # if LineData["qtype"] in ["Visualization", "NumericalReasoning"]:
                        if LineData["qtype"] not in ["NumericalReasoning"]:
                            continue
                        number += 1
                        inputs = TableBenchConfig.InferPrompt["JSON"]["en"]["COT"].format(TableString=LineData["table"],QuestionString=LineData["question"])
                        # inputs = LineData["instruction"]
                        # response = await send_request_with_client(inputs, model=model_name, function_call=False)
                        # TableBenchConfig.InferPrompt["JSON"]["en"]["COT"].format(TableString=LineData["table"],QuestionString=LineData["question"])

                        # 本地
                        response = await self.inference(model_name, local="True", prompt_text=inputs)
                        response_result.append({
                            "DatasetName": self.DatasetName,
                            "Instruction": inputs,
                            "Table": LineData["table"],
                            "Question": LineData["question"],
                            "LLMResponse": response,
                            "Type": dataType,
                            "id": LineData["id"],
                            "label": LineData["answer"],
                            "qtype": LineData["qtype"],
                            "qsubtype": LineData["qsubtype"],
                            "methods": dataType
                        })
                        with open(f'./{self.DatasetName}/qwen2-5.json', 'w', encoding='utf-8') as f:
                            json.dump(response_result, f, ensure_ascii=False, indent=4)
        print(f"数据集总数量为{number}")

    def evaluateResult(self, jsonPath):
        """
        对 LLM 生成结果评估。
        
        该数据集对应的jsonPath结果文件地址。
        """

        data = LoadJson(jsonPath)

        with open(TableBenchConfig.EvaPromptPath, 'r', encoding='utf-8') as file:
            file_content = file.read()


        response_result = []
        for question_item in data:
            inputjson = json.dumps({
                "query": question_item["Question"],
                "model_response": tablebenchutils.ExtractResponse(question_item["LLMResponse"]),
                "ground_truth": question_item['label']
            },indent=4,ensure_ascii=False)


            inputStr = file_content.replace(r"{{inputJson}}", inputjson)

            response = post_request(inputStr, model="qwen2.5:32b")

            match = re.search(r'\d', response[::-1])
            match_text = match.group() if match else None

            if match_text is None:
                question_id = question_item["question_id"]
                print(f"{question_id} 抽取结果为空")

            response_result.append({
                "DatasetName": TableBenchConfig.DatasetName,
                "Instruction": question_item["Instruction"],
                "Question": question_item["Question"],
                "id": question_item["id"],
                "raw_model_response": question_item["LLMResponse"],
                "model_response": tablebenchutils.ExtractResponse(question_item["LLMResponse"]),
                "ground_truth": question_item['label'],
                "EvaResultWithCOT": response,
                "ExtractResult": match_text,
                "qtype": question_item["qtype"],
                "qsubtype": question_item["qsubtype"]
            })

            with open(os.path.join(os.path.dirname(jsonPath), os.path.splitext(os.path.basename(jsonPath))[0]+"_EvaRes.json"), 'w', encoding='utf-8') as f:
                json.dump(response_result, f, ensure_ascii=False, indent=4)

    def extractResultForCompute(self, jsonPath):
        """
        对 LLM 生成结果评估。
        
        该数据集对应的jsonPath结果文件地址。
        """

        data = LoadJson(jsonPath)

        with open(TableBenchConfig.EvaPromptPathForAnswer, 'r', encoding='utf-8') as file:
            file_content = file.read()


        response_result = []
        for question_item in data:
            inputjson = json.dumps({
                "query": question_item["question"],
                "model_response": tablebenchutils.ExtractResponse(question_item["ReviseRes"])
            },indent=4,ensure_ascii=False)


            inputStr = file_content.replace(r"{{inputJson}}", inputjson)
            while(True):
                try:
                    response = post_request(inputStr, model="qwen2.5:32b")
                
                    pattern = r'```json\s*(.*?)\s*```'
                    match_text = re.search(pattern, response, re.DOTALL)
                    print(match_text.group(1))

                    if match_text is None:
                        question_id = question_item["id"]
                        print(f"{question_id} 抽取结果为空")

                    extract = json.loads(match_text.group(1)).get("Answer", "-")
                    question_item.update({
                        "EvaResultWithCOT": response,
                        "ExtractResult": extract if extract else "-",
                        "model_response": tablebenchutils.ExtractResponse(question_item["ReviseRes"])
                    })
                    response_result.append(question_item)
                    break
                except:
                    continue

            with open(os.path.join(os.path.dirname(jsonPath), os.path.splitext(os.path.basename(jsonPath))[0]+"_EvaRes.json"), 'w', encoding='utf-8') as f:
                json.dump(response_result, f, ensure_ascii=False, indent=4)

    def evaluateResultForCompute(self, jsonPath):
        """
        对 LLM 生成结果评估。
        
        该数据集对应的jsonPath结果文件地址。
        """

        data = LoadJson(jsonPath)

        with open(TableBenchConfig.EvaPromptPathZH, 'r', encoding='utf-8') as file:
            file_content = file.read()


        response_result = []
        for question_item in data:
            inputjson = json.dumps({
                "query": question_item["question"],
                "model_response": tablebenchutils.ExtractResponse(question_item["ReviseRes"]),
                "ground_truth": question_item['label']
            },indent=4,ensure_ascii=False)


            inputStr = file_content.replace(r"{{inputJson}}", inputjson)

            response = post_request(inputStr, model="qwen2.5:32b")

            match = re.search(r'\d', response[::-1])
            match_text = match.group() if match else None

            if match_text is None:
                question_id = question_item["id"]
                print(f"{question_id} 抽取结果为空")

            question_item.update({
                "EvaResultWithCOT": response,
                "ExtractResult": match_text,
                "model_response": tablebenchutils.ExtractResponse(question_item["ReviseRes"])
            })
            response_result.append(question_item)

            with open(os.path.join(os.path.dirname(jsonPath), os.path.splitext(os.path.basename(jsonPath))[0]+"_EvaRes.json"), 'w', encoding='utf-8') as f:
                json.dump(response_result, f, ensure_ascii=False, indent=4)

    def inference(self, modelName:str, local: bool):
        """
        大模型推理 by hf

        Args:
            modelName: 模型名，[qwen2.5, qwen2.5:32b]
            local: 是否是本地hf模型
        """
        modelPath = {
            "qwen2.5": "/raid/share/jiangchangjiang/tablellmPipeline/models/Qwen/Qwen2.5-7B-Instruct",
            "qwen2.5:32b": "/raid/share/jiangchangjiang/tablellmPipeline/models/Qwen/qwen/Qwen2.5-32B-Instruct",
            "qwen2.5-math:7b": "/raid/share/jiangchangjiang/tablellmPipeline/models/Qwen/Qwen2.5-Math-7B-Instruct"
        }
        response_result = []
        number = 0
        if local is True:
            model, tokenizer = load_model(modelPath[modelName])
        for dataType, dataMap in TableBenchConfig.RawDataPath.items():
            if dataType in ["TCoT"]:
                with open(dataMap, 'r', encoding='utf-8') as file:
                    for line in file:
                        # 将每一行的JSON字符串解析为字典，并追加到列表中

                        LineData = json.loads(line.strip())
                        # if LineData["qtype"] in ["Visualization", "NumericalReasoning"]:
                        if LineData["qtype"] not in ["NumericalReasoning"]:
                            continue
                        number += 1
                        # inputs = TableBenchConfig.InferPrompt["JSON"]["en"]["COT"].format(TableString=LineData["table"],QuestionString=LineData["question"])
                        # inputs = TableBenchConfig.TestPrompt.format(TableString=LineData["table"],QuestionString=LineData["question"])
                        if local is True:
                            # model, tokenizer = load_model("/raid/share/jiangchangjiang/tablellmPipeline/models/Qwen/Qwen2.5-7B-Instruct")
                            # response = inference_by_hf(model, tokenizer, inputs)
                            # response = post_request(prompt_text=LineData["Instruction"], modelName)
                            messages = [
                                {"role": "system", "content": TableBenchConfig.RawSystemPrompt},
                                {"role": "user", "content": TableBenchConfig.RawTCOTPrompt.format(TableString=LineData["table"],QuestionString=LineData["question"])}
                            ]
                            response = call_llm(model, tokenizer, messages)
                            # print(response)
                            # print(LineData["instruction"])
                            response_result.append({
                                "DatasetName": self.DatasetName,
                                "Instruction": messages,
                                "Table": LineData["table"],
                                "Question": LineData["question"],
                                "LLMResponse": response,
                                "Type": dataType,
                                "id": LineData["id"],
                                "label": LineData["answer"],
                                "qtype": LineData["qtype"],
                                "qsubtype": LineData["qsubtype"],
                                "methods": dataType
                            })
                        with open(f'./{self.DatasetName}/qwen2-5.json', 'w', encoding='utf-8') as f:
                            json.dump(response_result, f, ensure_ascii=False, indent=4)
        print(f"数据集总数量为{number}")
        return response


tableBenchEvaluator = TableBenchEvaluator()