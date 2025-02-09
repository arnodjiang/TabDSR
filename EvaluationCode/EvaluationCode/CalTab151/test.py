import json
import argparse
from collections import defaultdict

def count_dataset_names(input_file):
    counts = defaultdict(int)
    with open(input_file, 'r') as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                dataset_name = record.get('DatasetName', None)
                
                if dataset_name is not None:
                    counts[dataset_name] += 1
                else:
                    print(f"Warning: Missing 'DatasetName' in line {line_number}")
                    
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON at line {line_number}: {e.msg}")
                
    return dict(counts)

def main():
    parser = argparse.ArgumentParser(description='统计 JSON Line 文件中 DatasetName 类别的数量')
    parser.add_argument('--input_file', help='输入的 JSON Line 文件路径')
    args = parser.parse_args()

    results = count_dataset_names(args.input_file)
    
    if not results:
        print("未找到有效的 DatasetName 记录")
        return
    
    max_length = max(len(name) for name in results.keys()) if results else 0
    print("统计结果：")
    for name, count in sorted(results.items()):
        print(f"{name.ljust(max_length)} : {count}")

if __name__ == "__main__":
    main()