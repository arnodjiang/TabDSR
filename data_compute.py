import json
from collections import Counter

# 定义文件路径
file_path = '/raid/share/jiangchangjiang/tablellmPipeline/EvaluationCode/TableBench/RawData/TableBench_PoT.jsonl'

# 读取文件并计算包含 'NumericalReasoning' 键的字典个数，同时统计各个 'qsubtype' 的数量
def count_numerical_reasoning_and_qsubtype(file_path):
    count = 0
    qsubtype_counter = Counter()
    
    with open(file_path, 'r') as file:
        for line in file:
            # 每一行是一个字典，尝试解析JSON
            try:
                record = json.loads(line)
                if record.get("qtype") == "NumericalReasoning":
                    count += 1
                    # 统计 'qsubtype' 的值
                    qsubtype = record.get("qsubtype")
                    if qsubtype:
                        qsubtype_counter[qsubtype] += 1
            except json.JSONDecodeError:
                continue  # 跳过无法解析的行
    
    return count, qsubtype_counter

# 输出结果
numerical_reasoning_count, qsubtype_count = count_numerical_reasoning_and_qsubtype(file_path)
print(f"包含 'NumericalReasoning' 键的字典个数: {numerical_reasoning_count}")
print("各个 'qsubtype' 的数量：")
for qsubtype, count in qsubtype_count.items():
    print(f"{qsubtype}: {count}")
