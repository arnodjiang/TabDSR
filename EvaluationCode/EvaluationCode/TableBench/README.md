# https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench

- URL: [arxiv 文章](https://www.arxiv.org/pdf/2408.09174)

## Dataset Summary

TableBench is a dataset that covers 4 major categories and 18 subcategories, focusing on the multi-dimensional capabilities of table question answering.

## Data Fields

|ID|String|Description|
|---|---|---|
|id|string|唯一标识|
|qtype|string|问题类型，包括 (FactChecking, NumericalReasoning, DataAnalysis, Visualization)|
|qsubtype|string|问题的子类型，包括Aggregation等 18 种类型，具体参照[原文](https://www.arxiv.org/pdf/2408.09174)|
|instruction|string|LLM 的 Prompt|
|instruction_type|string|	Three different instruction types in TableBench: TCoT(Textual Chain of Thought),SCoT(Symbolic Chain of Thought) and PoT(Program of Thought)|
|table|string|Table 字典，{"columns": [], "data": [[],[]]}|
|question|string|Question|
|answer|string|Answer|
|answer_formatter|string|输出的限制（很重要）|
