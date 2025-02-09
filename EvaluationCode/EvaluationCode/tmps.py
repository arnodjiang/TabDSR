import json
with open('/public/home/lab10/jiangchangjiang/P-tablellm-2024-04-31/EvaluationCode/EvaluationCode/AllData/data_all.json', 'r') as file:
    data = json.load(file)
# 将数据写入 JSONL 文件
output_file = "/public/home/lab10/jiangchangjiang/P-tablellm-2024-04-31/EvaluationCode/EvaluationCode/AllData/data_all.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for steps,record in enumerate(data):
        record["id"] = steps
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"数据已写入 {output_file}")