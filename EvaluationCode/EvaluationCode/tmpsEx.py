import json

def extract_tablebench_records(input_file, output_file):
    """
    Extracts records with "DatasetName": "TableBench" from a JSONL file 
    and saves them to a JSON file.

    Args:
        input_file (str): Path to the input JSONL file.
        output_file (str): Path to save the filtered JSON file.
    """
    filtered_data = []

    try:
        # Open the JSONL file and process line by line
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    record = json.loads(line.strip())
                    if record.get("DatasetName") == "TableBench":
                        filtered_data.append(record)
                except json.JSONDecodeError as e:
                    print(f"Error decoding line: {line.strip()} - {e}")

        # Write the filtered records to the output JSON file
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(filtered_data, outfile, ensure_ascii=False, indent=4)

        print(f"Filtered data saved to {output_file}")

    except FileNotFoundError:
        print(f"Input file not found: {input_file}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Specify file paths
input_file_path = "/public/home/lab10/jiangchangjiang/P-tablellm-2024-04-31/EvaluationCode/EvaluationCode/AllData/data_all.jsonl"  # Replace with your input file path
output_file_path = "/public/home/lab10/jiangchangjiang/P-tablellm-2024-04-31/EvaluationCode/EvaluationCode/TableBench/RawData/TableBenchData.json"  # Desired output file name

# Execute the function
extract_tablebench_records(input_file_path, output_file_path)