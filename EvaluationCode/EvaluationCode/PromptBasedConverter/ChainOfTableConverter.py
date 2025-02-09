import json
def process_jsonline(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # 解析 JSON 数据
            record = json.loads(line.strip())
            
            # 提取表格信息
            columns = record["Table"]["columns"]
            data = record["Table"]["data"]
            table_text = [columns] + data  # 合并 columns 和 data
            
            # 构造输出 JSON 数据
            output_record = {
                "statement": record["Question"],
                "table_caption": "",
                "table_text": table_text,
                "cleaned_statement": record["Question"],
                "chain": [],
                "RawData": {}
            }
            output_record["RawData"].update(record)
            
            # 将数据写入输出文件
            outfile.write(json.dumps(output_record, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    process_jsonline(
        input_file="/public/home/lab10/jiangchangjiang/P-tablellm-2024-04-31/EvaluationCode/EvaluationCode/WTQ/data_all_wtq_sikiq.jsonl",
        output_file="/public/home/lab10/jiangchangjiang/P-tablellm-2024-04-31/EvaluationCode/EvaluationCode/WTQ/data_all_wtq_cot.jsonl"
    )