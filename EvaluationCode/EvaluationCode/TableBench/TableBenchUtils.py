class TableBenchUtils:
    def __init__(self) -> None:
        pass

    def ExtractResponse(self, text):
        """
        从文本中提取结果，“The Final answer is XXX”
        """
        return text.split("\n")[-1].strip()

tablebenchutils = TableBenchUtils()