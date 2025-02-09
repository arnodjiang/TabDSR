import json
import os


from loguru import logger
import argparse
import json
import re
from rouge_score import rouge_scorer

class Evaluator:
    @staticmethod
    def remove_symbol(data):
        data = data.replace("%","").replace(",","").replace("$","").replace("-","").strip()
        return data

    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    @staticmethod
    def compare_and_round(s1, s2):
        # print(s2)
        s1, s2 = Evaluator.remove_symbol(s1), Evaluator.remove_symbol(s2)
        # print(s2)
        if not Evaluator.is_number(s1) or not Evaluator.is_number(s2):
            return s1, s2

        num1 = float(s1)
        num2 = float(s2)
        # 获取小数点后的位数
        def get_decimal_places(s):
            if '.' in s:
                return len(s.split('.')[1])
            else:
                return 0
        
        decimal_places1 = get_decimal_places(s1)
        decimal_places2 = get_decimal_places(s2)
        
        # 将小数点位数较多的数四舍五入到小数点位数较少的位数
        if decimal_places1 > decimal_places2:
            rounded_num1 = round(num1, decimal_places2)

            return rounded_num1, s2
        elif decimal_places2 > decimal_places1:

            rounded_num2 = round(num2, decimal_places1)
            return s1, rounded_num2
        else:
            return s1, s2

def load_json(file_path):
    """加载JSON文件并返回数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def index_by_id(data):
    """将数据列表转换为以id为键的字典"""
    return {item['id']: item for item in data}

def compute_right(item):
    answer_list = item.get("Answer", [])
    answer_list = [Evaluator.remove_symbol(i) for i in answer_list]
    extract_result = item.get("ExtractResult", "").lower()
    raw_answer = item.get("RawAnswer", "").lower()
    if len(answer_list) == 1:
        # print(extract_result)
        if answer_list[0] .startswith("."):
            answer_list[0] = "0"+answer_list[0]
        extract_results = re.findall(r"[\.\d]+",extract_result) if Evaluator.is_number(answer_list[0]) else [extract_result]
        new_number_results = []
        for ii_results in extract_results:
            answer_new, fix_result = Evaluator.compare_and_round(answer_list[0], ii_results)
            # logger.info(answer_new)
            # logger.info(fix_result)
            # logger.info(answer_new.strip() == fix_result.strip())
            
            new_number_results.append(str(fix_result))
            if str(answer_new) == str(fix_result):
                answer_list[0] = str(answer_new)
                
        extract_result = " ".join(new_number_results)
        # answer_list[0], extract_result = Evaluator.compare_and_round(answer_list[0], extract_result)
        # print(extract_result)
        answer_list[0], extract_result = str(answer_list[0]), str(extract_result)
        extracted = [ans.lower() for ans in answer_list if ans.lower().strip("0") in extract_result.lower() or (ans.lower() == "yes" and "true" in extract_result.lower()) or (ans.lower() == "no" and "false" in extract_result.lower())]
        # print(extracted)
        extract_result = ", ".join(extracted) if extracted !=[] else extract_result

        if answer_list[0] in extract_result:
            return True
        else:
            return False


def compute_right_id(data):
    data_map = {}
    for i in data:
        isR = compute_right(i)
        data_map[i["id"]] = {
            "right": isR,
            "Answer": i["Answer"],
            "ExtractResult": i["ExtractResult"]
        }
    return data_map

def main(file_a, file_b):
    raw_a = load_json(file_a)
    raw_b = load_json(file_b)

    dic_a = compute_right_id(raw_a)
    dic_b = compute_right_id(raw_b)

    for k,v in dic_a.items():
        # print(dic_b[k])
        if v["right"] is True and k in dic_b and dic_b[k]["right"] is False:
            # pass
            br = dic_b[k]
            logger.debug(f"id 为{k}:")
            logger.debug(f"dic_a 的值为: [\n{v}\n]")
            logger.debug(f"dic_b 的值为: [\n{br}\n]")



if __name__ == "__main__":
    p1 = "/public/home/lab10/jiangchangjiang/P-tablellm-2024-04-31/Archive/openai/gpt-4o_FinQa_raw_TCoT.json"
    p2 = "/public/home/lab10/jiangchangjiang/P-tablellm-2024-04-31/Archive/openai/gpt-4o_FinQa_1+2+3_PoT.json"
    main(p1, p2)