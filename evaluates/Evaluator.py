from loguru import logger
import argparse
import json
import re
from rouge_score import rouge_scorer

class Evaluator:
    @staticmethod
    def remove_symbol(data):
        data = str(data).replace("%","").replace(",","").replace("$","").replace("-","").strip()
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


    @staticmethod
    def calculate_metrics(data):
        total_acc = 0
        acc_count = 0
        results = []
        rougel_acc = 0

        for item in data:
            # 获取必要的数据
            # if item["id"] >= 19:
            #     continue
            answer_list = item.get("Answer", [])
            answer_list = [Evaluator.remove_symbol(i) for i in answer_list]
            extract_result = item.get("ExtractResult", "").lower()
            raw_answer = item.get("RawAnswer", "").lower()

            # for TableSqlify
            # extract_result = item.get("RawAnswer", "").lower()
            # raw_answer = ", ".join(answer_list)

            # 提取 Answer 中存在于 ExtractResult 的部分
            ## 如果是数字，单独处理，取小数点no
            # print(extract_result)
            if len(answer_list) == 1:
                # print(extract_result)
                if answer_list[0].startswith("."):
                    answer_list[0] = "0"+answer_list[0]
                extract_results = re.findall(r"[\.\d]+",extract_result) if Evaluator.is_number(answer_list[0]) else [extract_result]
                new_number_results = []
                for ii_results in extract_results:
                    answer_new, fix_result = Evaluator.compare_and_round(answer_list[0], ii_results)
                    
                    new_number_results.append(str(fix_result))
                    if str(answer_new) == str(fix_result):
                        answer_list[0] = str(answer_new)
                        
                extract_result = " ".join(new_number_results)
                answer_list[0], extract_result = Evaluator.compare_and_round(answer_list[0], extract_result)

                answer_list[0], extract_result = str(answer_list[0]), str(extract_result)

            extracted = [ans.lower() for ans in answer_list if ans.lower().strip("0") in extract_result.lower() or (ans.lower() == "yes" and "true" in extract_result.lower()) or (ans.lower() == "no" and "false" in extract_result.lower())]
            # print(extracted)
            extract_result = ", ".join(extracted) if extracted !=[] else extract_result
            # logger.info("Ref: {answer_list}, extract_result: {extract_result}".format(extract_result=extract_result, answer_list=answer_list))
            # 计算 ROUGE 值
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

            # rouge_score = scorer.score(raw_answer, extract_result.lower())["rougeL"].fmeasure
            # if True: ## raw_answer too many unrelevant answer and not suitable for evaluate
            # if answer_list[0] == "88.81":
            #     print(float(answer_list[0]))
            #     print(float(extract_result)-0.01)
            #     print(str(answer_list[0]) == (str(extract_result-0.01)))
            logger.info("Ref: {answer_list}, extract_result: {extract_result}".format(extract_result=extract_result, answer_list=answer_list))
            rouge_score = scorer.score(", ".join(answer_list), extract_result.lower())["rougeL"].fmeasure
            rougel_acc += rouge_score
            # 计算准确率（ACC）
            if answer_list:
                acc = sum(1 for ans in answer_list if ans.lower() in extracted and (ans != "" and extracted != "")) / len(answer_list)
            else:
                acc = 0
            em = 1 if "".join(answer_list) in item.get("ExtractResult", "").lower() else 0
            # 保存结果
            results.append({
                "ROUGE": rouge_score,
                "ACC": acc,
                "Extracted": extracted,
                "EM": em
            })

            total_acc += acc
            acc_count += 1

        # 计算平均准确率
        avg_acc = total_acc / acc_count if acc_count > 0 else 0
        avg_rouge_score = rougel_acc / len(data)
        # em = sum([i["EM"] for i in results]) / len(results)
        return {
            "acc": round(avg_acc*100,2),
            "rougel": round(avg_rouge_score*100,2)
        }

    @staticmethod
    def evaluate(DataPath, **kwargs):
        with open(DataPath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        data_name = kwargs["data_name"]
        logger.info(f"当前数据{data_name}长度为: {len(data)}")
        if kwargs["data_name"] == "TableBench":
            assert len(data) == 493


        elif kwargs["data_name"] == "CalTab151":
            assert len(data) == 151

        elif kwargs["data_name"] == "TatQa":
            assert len(data) == 736
        
        result = Evaluator.calculate_metrics(data)
        logger.info(f"Eva result: {result}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate LLM infer result', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data_path', '-dp', help='infer result path, json file')
    parser.add_argument('--data_name', '-dn', help='datasetname')

    args = parser.parse_args()

    Evaluator.evaluate(DataPath=args.data_path, data_name=args.data_name)

    # print(Evaluator.compare_and_round("0.385", "0.39"))