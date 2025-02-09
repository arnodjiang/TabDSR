from EvaConfig import BasicEvaConfig

import os

class TableBenchConfig(BasicEvaConfig):
    DatasetName = "TableBench"
    RawDataRoot = "/raid/share/jiangchangjiang/tablellmPipeline/EvaluationCode/TableBench/RawData"
    RawDataPath = {
        "DP": os.path.join(RawDataRoot, "TableBench_DP.jsonl"),
        "PoT": os.path.join(RawDataRoot, "TableBench_PoT.jsonl"),
        "SCoT": os.path.join(RawDataRoot, "TableBench_SCoT.jsonl"),
        "TCoT": os.path.join(RawDataRoot, "TableBench_TCoT.jsonl")
    }
    # Instruction = "Please read the following table in {TableFormat} format and then answer the question according to the table.\nRequirements:\nThe output format should be \"The answer is xxx, yyy, zzz\", with answers separated by commas and do not response \"According to the table\".\n\nTitle:\n{TableTitle}\n\nTable:\n{TableString}\n\nQuestion:\n{QuestionString}"