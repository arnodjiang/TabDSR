import json

def count_jsonl_lines(file_path):
    """
    计算 JSON Lines 文件中的 JSON 对象数量。

    Args:
        file_path (str): .jsonl 文件的路径。

    Returns:
        int: JSON 对象的数量。
    """
    count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()  # 去除多余的空白字符
                if line:  # 忽略空行
                    try:
                        json.loads(line)  # 验证是否为合法的 JSON
                        count += 1
                    except json.JSONDecodeError:
                        print(f"Warning: Invalid JSON on line {count + 1}")
        print(f"Total number of JSON objects: {count}")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return count

# 示例使用
if __name__ == "__main__":
    file_path = "/public/home/lab10/jiangchangjiang/P-tablellm-2024-04-31/EvaluationCode/EvaluationCode/TableBench/RawData/TableBenchData.jsonl"
    count_jsonl_lines(file_path)