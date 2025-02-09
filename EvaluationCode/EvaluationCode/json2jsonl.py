import json

# 输入和输出文件路径
input_file = "/public/home/lab10/jiangchangjiang/P-tablellm-2024-04-31/EvaluationCode/EvaluationCode/TableBench/RawData/TableBenchData.json"
output_file = "/public/home/lab10/jiangchangjiang/P-tablellm-2024-04-31/EvaluationCode/EvaluationCode/TableBench/RawData/TableBenchData.jsonl"

# 转换 JSON 文件为 JSONL 文件
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    data = json.load(infile)  # 加载整个 JSON 文件
    if isinstance(data, list):  # 确保是一个 JSON 数组
        for item in data:
            json.dump(item, outfile, ensure_ascii=False)  # 将 JSON 对象写入一行
            outfile.write("\n")  # 添加换行符
    else:
        print("输入文件不是一个 JSON 数组，无法转换为 JSON Lines 格式。")