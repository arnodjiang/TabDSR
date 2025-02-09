import json

file_path = "/public/home/lab10/jiangchangjiang/P-tablellm-2024-04-31/EvaluationCode/EvaluationCode/TableBench/RawData/TableBench_DP.jsonl"

toDict = {}
output_file = "/public/home/lab10/jiangchangjiang/P-tablellm-2024-04-31/Archive/TableBenchFix/Qwen/2.5/7b/raw_TCoT.json"
# with open("/public/home/lab10/jiangchangjiang/P-tablellm-2024-04-31/EvaluationCode/EvaluationCode/TableBench/RawData/TableBenchData.jsonl", 'r', encoding='utf-8') as file:
#     for line in file:
#         line = json.loads(line.strip())  # 去除多余的空白字符

#         toDict[line["Question"]] = line
with open(output_file, 'r', encoding='utf-8') as file:
    data = json.load(file)
for i in data:
    toDict[i["Question"]] = i

total_n = 0

to_data = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = json.loads(line.strip())  # 去除多余的空白字符

        if line["qtype"] in ["NumericalReasoning", "FactChecking"]:
            total_n += 1
            rawData = toDict[line["question"]]
            rawData["rawid"] = line["id"]
            # print(rawData)
            rawData["TableBenchType"] = line["qtype"]
            to_data.append(rawData)

# output_file = "/public/home/lab10/jiangchangjiang/P-tablellm-2024-04-31/Archive/TableBenchFix/Qwen/2.5/7b/raw_DP.json"
# with open(output_file, "w", encoding="utf-8") as f:
#     for steps,record in enumerate(to_data):
#         record["id"] = steps
#         f.write(json.dumps(record, ensure_ascii=False) + "\n")

# 3. 将修改后的数据保存回 JSON 文件
with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(to_data, file, ensure_ascii=False, indent=4)

print(f"全部总数为{total_n}")
print(f"新数据为{len(to_data)}")