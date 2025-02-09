import os
import json

from ..UnifyData import UnifyData
from ..utils import split_data
from dataclasses import asdict
from ..utils import num_tokens_from_string, isLargeTable

class DataConfig:
    DatasetName = "TableBench"
    RawDataRoot = "/raid/share/jiangchangjiang/tablellmPipeline/EvaluationCode/TableBench/RawData"
    Raw_DP = "TableBench_DP.jsonl"
    Raw_PoT = "TableBench_PoT.jsonl"
    Raw_SCoT = "TableBench_SCoT.jsonl"
    Raw_TCoT = "TableBench_TCoT.jsonl"
    IsNeedSplit = True

class DataProcessor:
    def __init__(self, DataConfig):
        self.DatasetName = DataConfig.DatasetName
        self.RawDataRoot = DataConfig.RawDataRoot
        self.RawDataPath = {
            "DP": os.path.join(self.RawDataRoot, DataConfig.Raw_DP) if DataConfig.Raw_DP is not None else None,
            "PoT": os.path.join(self.RawDataRoot, DataConfig.Raw_PoT) if DataConfig.Raw_PoT is not None else None,
            "SCoT": os.path.join(self.RawDataRoot, DataConfig.Raw_SCoT) if DataConfig.Raw_SCoT is not None else None,
            "TCoT": os.path.join(self.RawDataRoot, DataConfig.Raw_TCoT) if DataConfig.Raw_TCoT is not None else None,
        }
        self.RawDescriptionTemplate = ""
        self.IsNeedSplit = DataConfig.IsNeedSplit

    def main(self, dataRoot):
        to_results = []
        for dataType, dataPath in self.RawDataPath.items():
            result = self.RawDataToJson(DataPath=os.path.join(dataRoot, dataPath), DataType=dataType)
            to_results.extend(result)
        
        if self.IsNeedSplit is True:
            to_results = split_data(to_results)
        return to_results

    def RawDataToJson(self, DataPath, DataType="train"):
        """
        Raw data to json
        """
        data_list = []
        with open(DataPath, 'r', encoding='utf-8') as file:
            for line in file:
                # 将每一行的JSON字符串解析为字典，并追加到列表中
                LineData = json.loads(line.strip())
                
                # RawDescriptionTemplate = self.RawDescriptionTemplate.format(
                #     table_section_title = LineData["table_section_title"],
                #     table_section_text = LineData["table_section_text"],
                #     table_page_title = LineData["table_page_title"]
                # )
                RawDescriptionTemplate = ""
                NumTokens = num_tokens_from_string(LineData["table"].replace("None", "-")+" "+LineData["answer"].replace("None", "-"))
                unifyDataHTML = UnifyData(
                    TableType="Dict('Column','data')",
                    IsRawType=True,
                    Data=LineData["table"].replace("None", "-"),
                    RelevantDescription=RawDescriptionTemplate,
                    DatasetName=self.DatasetName,
                    Question=LineData["question"].replace("None", "-"),
                    Answer=LineData["answer"].replace("None", "-"),
                    Type=DataType,
                    Task="TQA",
                    LargeTable=isLargeTable(NumTokens)
                )
                data_list.append(asdict(unifyDataHTML))
        return data_list
dataProcessor = DataProcessor(DataConfig)