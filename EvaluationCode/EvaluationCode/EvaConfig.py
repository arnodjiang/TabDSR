class BasicEvaConfig:
    DataRoot = "/Users/arnodorian/Desktop/tablellmpipeline/EvaluationCode"
    EvaPromptPath = "./Prompts/EvaResPrompt.txt"
    EvaPromptPathZH = "./Prompts/EvaResPromptZH.txt"
    EvaPromptPathForAnswer = "./Prompts/EvaResPromptForAnswer.txt"

    InferPrompt = {
        "HTML":{
            "en": "Please read the following table in HTML format and then answer the question according to the table.\nRequirements:\nThe output format should be \"The answer is xxx, yyy, zzz\", with answers separated by commas and do not response \"According to the table\".\n\nTable:\n{TableString}\n\nQuestion:\n{QuestionString}\n\nAnswer:\n"
        },
        "JSON": {
            "en": {
                "COT": "Understand the table structure and answer the question step by step. Be sure to include the conclusion in the end, such as \"\n\nThe final answer is: answer1, answer2,...\".\n\nTable:\n{TableString}\n\nQuestion:\n{QuestionString}"
            }
        }
    }
    RawSystemPrompt = "You are a table analyst. Your task is to answer questions based on the table content."
    RawTCOTPrompt = "The answer should follow the format below:\n[Answer Format]\nFinal Answer: AnswerName1, AnswerName2...\n\nEnsure the final answer format is the last output line and can only be in the \"Final Answer: AnswerName1, AnswerName2...\" form, no other form. Ensure the \"AnswerName\" is a number or entity name, as short as possible, without any explanation.\n\n\nLet's think step by step and then give the final answer to the question.\n\nRead the table below in JSON format:\n[TABLE] \n{TableString}\n\nLet's get start!\nQuestion: {QuestionString}"
    TestPrompt = "First, identify the cells relevant to the question step by step. According to relevant cells, create a mathematical expression that can be directly computed. For example, if the question is to compute relevant cells' average value, format the result as (1+2+3)/4(average of relevant values). Finally, include the mathematical expression at the end, such as \"\n\nThe mathematical expression is: 1+2+3+4(Sum up relevant values)\".\n\nTable:\n{TableString}\n\nQuestion:\n{QuestionString}"

    RawSystemPromptWithSelfCorrection = "You are a table analyst capable of complex reasoning, reflection, and self-correction. Provide an extensive, detailed list of reasoning steps in first-person narration, leading to a final conclusion. Your task is to answer questions based on the table content."
    RawTCOTPromptWithSelfCorrection = "Each step should represent a single unit of thought, such as observations, calculations, questions, doubts, realizations, corrections, reflections, discoveries, or decisions. The answer should follow the format below:\n[Answer Format]\nFinal Answer: AnswerName1, AnswerName2...\n\nEnsure the final answer format is the last output line and can only be in the \"Final Answer: AnswerName1, AnswerName2...\" form, no other form. Ensure the \"AnswerName\" is a number or entity name, as short as possible, without any explanation.\n\n\nLet's think step by step and then give the final answer to the question.\n\nRead the table below in JSON format:\n[TABLE] \n{TableString}\n\nLet's get start!\nQuestion: {QuestionString}"