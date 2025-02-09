import json
import requests
import time
from evalogger import logger

import asyncio
import ollama, re
from pprint import pprint

from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def calculate_mathematical_expression(expression: str):
    """
    A function takes a complete mathematical expression as input and returns the calculated result. Mathematical expression is "1+2+3" or "(45+21+30) / 3". Do not use "avg(45,21,30)". If the expression is python function, you must convert it as Mathematical expression, such as "avg(45,21,30)" as "(45 + 21 + 30) / 3"
    
    Args:
        expression: A valid mathematical expression string to calculate.
    """
    return eval(expression)

def parse_tool_calls(model_output):
    """
    解析大模型输出中的多个 <tool_call> 并返回解析后的结果列表.

    Args:
        model_output (str): 大模型生成的包含多个 tool_call 的文本.

    Returns:
        list: 每个元素是解析后的 tool_call 字典.
    """
    # 正则表达式匹配 <tool_call> ... </tool_call>
    tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
    tool_calls = re.findall(tool_call_pattern, model_output, re.DOTALL)

    parsed_tool_calls = []
    
    # 解析每个 <tool_call> 中的 JSON 内容
    for tool_call in tool_calls:
        try:
            # 解析 JSON 数据
            tool_call_json = json.loads(tool_call.strip())
            parsed_tool_calls.append(tool_call_json)
        except json.JSONDecodeError as e:
            print(f"JSON decoding failed: {e}")
    
    return parsed_tool_calls

url = "http://172.17.60.41:11434/api/generate"
_OLLAMA_HOST = "http://172.17.60.41:11434"
client = ollama.AsyncClient(host=_OLLAMA_HOST)

##### 第二次v2
def inference_by_hf(model, tokenizer, prompt_text: str):
    """
    用huggingface库推理.

    Args:
        prompt_text: 完整的提示词
        model: 模型名
        tools：工具函数列表
    """
    messages = [
        {"role": "system", "content": "You are a table analyst. Your task is to identify relevant cells from the table based on the question and convert them into mathematical expressions for computation. The result of mathematical expression is the answer to question."},
        {"role": "user", "content": prompt_text}
    ]

    function_maps = {
        "calculate_mathematical_expression": calculate_mathematical_expression
    }
    # print(response)

    response = call_llm(model, tokenizer, messages)

    messages.append({"role": "assistant", "content": response})
    messages_2 = [
        {"role": "system", "content": "You are a table analyst. Your task is to compute the final mathematical expression based on the reasoning content."},
        {"role": "user", "content": f"Compute mathematical expression: {response}"}
    ]
    response = call_llm(model, tokenizer, messages_2, tools=[calculate_mathematical_expression])
    for _ in range(5):
        expression = ""
        try:
            response = parse_tool_calls(response)

            if response != []:
                for res_item in response:
                    function_name = function_maps[res_item["name"]]
                    expression = res_item["arguments"]["expression"]
                    result = function_name(res_item["arguments"]["expression"])
                    messages_2.append({"role": "assistant", "tool_calls": [{"type": "function", "function": res_item}]})
                    messages_2.append({
                        "role": "tool",
                        "content": result
                    })
            break
        except:
            messages.append({"role": "user", "content": f"Do not generate the format of code \"{expression}\". Instead, convert it to a complete math expression like \"1+2+3\"."})
            # messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": res_item}]})

    return {
        "response":call_llm(model,tokenizer,messages_2),
        "inputs": messages_2
    }

##### 第一次的函数
# def inference_by_hf(model, tokenizer, prompt_text: str):
#     """
#     用huggingface库推理.

#     Args:
#         prompt_text: 完整的提示词
#         model: 模型名
#         tools：工具函数列表
#     """
#     messages = [
#         {"role": "system", "content": "You are a table analyst. Your task is to answer questions based on the table content."},
#         {"role": "user", "content": prompt_text}
#     ]

#     function_maps = {
#         "calculate_mathematical_expression": calculate_mathematical_expression
#     }
#     # print(response)
#     for _ in range(5):
#         expression = ""
#         try:
#             response = call_llm(model, tokenizer, messages, tools=[calculate_mathematical_expression])

#             response = parse_tool_calls(response)

#             if response != []:
#                 for res_item in response:
#                     function_name = function_maps[res_item["name"]]
#                     expression = res_item["arguments"]["expression"]
#                     result = function_name(res_item["arguments"]["expression"])
#                     messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": res_item}]})
#                     messages.append({
#                         "role": "tool",
#                         "content": result
#                     })
#             break
#         except:
#             messages.append({"role": "user", "content": f"Do not generate the format of code \"{expression}\". Instead, convert it to a complete math expression like \"1+2+3\"."})
#             # messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": res_item}]})

#     return {
#         "response":call_llm(model,tokenizer,messages),
#         "inputs": messages
#     }


def call_llm(model, tokenizer, messages, tools=None):
    # print(tools)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools
    )
    # print(text)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.1,
        top_k=20,
        do_sample=True
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


async def send_request_with_client(prompt_text, model="qwen2.5", function_call=True):
    global init_time

    if function_call is False:
        messages = [{'role': 'user', 'content': prompt_text}]
        response = await client.chat(
            model=model,
            messages=messages,
            options={
                "seed": 42,
                "temperature": 0
            })
        # print(f"Response is {response}")
        return {
            "response": response['message']['content'],
            "methods": "direct"
        }
    
    while(True):
        messages = [{'role': 'user', 'content': prompt_text}]
        response = await client.chat(
            model=model,
            messages=messages,
            options={
                "seed": 42,
                "temperature": 0
            },
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "calculate_mathematical_expression",
                        "description": "This function takes a mathematical expression as input and returns the calculated result.",
                        "parameters": {
                            "type": "object",
                            "properties":{
                                "expression": {
                                    "type": "string",
                                    "description": "A valid mathematical expression to calculate."
                                }
                            }
                        },
                        "required": ["expression"]
                    }
                }
            ]
        )
        # print(response)
        
        response_message = response['message']
        # print(f"本次回复的内容为: {response_message}")
        if response['message'].get('tool_calls'):
            # messages.append(response['message'])
            available_functions = {
                'calculate_mathematical_expression': calculate_mathematical_expression
            }
            if not response['message']['tool_calls'][0]["function"]["arguments"].get("expression"):
                # print(f"该请求返回响应错误，需要重新调整")
                # print(response['message']['tool_calls'][0]["function"]["arguments"])
                messages.append(response['message'])
                response = await client.chat(
                    model=model,
                    messages=messages,
                    options={
                        "seed": 42,
                        "temperature": 0
                    })
                # print(f"Response is {response}")
                return {
                    "response": response['message']['content'],
                    "methods": "function_call"
                }
            for tool in response['message']['tool_calls']:
                function_to_call = available_functions[tool['function']['name']]
                # print(tool['function']['arguments']['numbers'])
                try:
                    function_response = function_to_call(tool['function']['arguments']['expression'])
                except:
                    messages = [{'role': 'user', 'content': prompt_text}]
                    response = await client.chat(
                        model=model,
                        messages=messages,
                        options={
                            "seed": 42,
                            "temperature": 0
                        })
                    # print(f"Response is {response}")
                    return {
                        "response": response['message']['content'],
                        "methods": "direct"
                    }
                # Add function response to the conversation
                messages.append(
                    {
                    'role': 'tool',
                    'content': str(function_response)
                    }
                )
        else:
            messages = [{'role': 'user', 'content': prompt_text}]
            response = await client.chat(
                model=model,
                messages=messages,
                options={
                    "seed": 42,
                    "temperature": 0
                })
            print(f"Response is {response}")
            return {
                "response": response['message']['content'],
                "methods": "direct"
            }
        break

    # Second API call: Get final response from the model
    final_response = await client.chat(model=model, messages=messages)
    # print(final_response)
    if final_response['message']['content'].strip() == "":
        response = await client.chat(
            model=model,
            messages=messages,
            options={
                "seed": 42,
                "temperature": 0
            })
        # print(f"Response is {response}")
        return {
            "response": response['message']['content'],
            "methods": "direct"
        }
    return {
        "response": final_response['message']['content'],
        "methods": "function_call"
    }

def post_request(prompt_text, model="qwen2.5"):
    global init_time
    """发送POST请求"""
    api_data = {
        "model": model,
        "prompt":prompt_text,
        "stream":False
    }
    start_time = time.time()
    # print(api_data)
    response = requests.post(url, data=json.dumps(api_data))
    # print(response.json())
    end_time = time.time()
    logger.info(f"单次推理运行时间{end_time-start_time}s")
    return response.json()["response"]

# result = post_request("你好，中国首都在哪里？")

def LoadJson(json_path):
    """
    读取 json 文件
    """
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def save_dict_to_json(data, file_path):
    """
    将包含列表的字典保存为 JSON 文件。

    :param data: 要保存的字典对象
    :param file_path: 保存的文件路径
    """
    try:
        # 打开指定文件路径，并以写入模式保存数据为 JSON
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        print(f"数据已成功保存到 {file_path}")
    except Exception as e:
        print(f"保存数据时出错: {e}")

def list_in_dict_to_dict(dataList, key):
    """将 Dict in List 转换为根据某个字段映射的 Hash 字典。

    Args:
    dataList: List[Dict]
        Description: 数据集字典
    dest: TheXXX
        For example,
    """
    return {d[key]: d for d in dataList}

if __name__ == "__main__":
    print(post_request(prompt_text="计算下列算数：82.8 + 35.9 + 17.1 + 50.6 + 64.0 + 424.0 + 116.4 + 84.3 + 18.6 + 22.5 + 101.3 + 58.5 + 81.4 + 966.1 + 15.3 + 260.6", model="qwen2.5"))