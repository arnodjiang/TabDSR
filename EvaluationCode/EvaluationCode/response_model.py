from dataclasses import dataclass

# @dataclass
# class ResponseModel:
#     TableType: str #html, latex
#     IsRawType: bool # Whether raw data type or converted
#     Data: str
#     RelevantDescription: str
#     DatasetName: str
#     Question: str
#     Answer: str
#     Type: str # train, dev, test
#     Task: str # TQA
#     LargeTable: str # 是否大表格

@dataclass
class ResponseModel:
    DatasetName: str
    Question: str
    Answer: str
    Type: str # train, dev, test