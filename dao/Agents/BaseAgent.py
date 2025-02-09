import re
import json
import io
import sys
import logging
import os
from loguru import logger

class BaseAgent:
    @staticmethod
    def load_prompt(txt_path):
        """
        加载Prompt
        """
        with open(txt_path, 'r', encoding='utf-8') as file:
            inputs = file.read()
        return inputs
    
    @staticmethod
    def build_first_prompt(system_prompt, user_prompt):
        """
        构建多层嵌套 prompt
        """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    @staticmethod
    def extract_json(inputs):
        """
        构建多层嵌套 prompt
        """
        matcch_result = re.findall(r"```json(.*?)```", inputs, re.IGNORECASE | re.DOTALL)
        last_json_block = matcch_result[-1] if matcch_result else None
        if matcch_result is None:
            return matcch_result
        else:
            try:
                resJson = json.loads(last_json_block.strip())
            except Exception as e:
                logger.error(f"解析 JSON 失败：\n{e}\n\n 原 json 格式为{last_json_block}\n\n")
            return json.loads(last_json_block.strip())

    @staticmethod
    def extract_reflection(inputs):
        """
        构建多层嵌套 prompt
        """
        matcch_result = re.findall(r"```json(.*?)```", inputs, re.IGNORECASE | re.DOTALL)
        # Extracting the last JSON block
        last_json_block = matcch_result[-1] if matcch_result else None
        if last_json_block is None:
            return None
        else:
            return json.loads(last_json_block.strip())
    @staticmethod
    def extract_python(inputs):
        """
        构建多层嵌套 prompt
        """
        matcch_result = re.search(r"```python(.*?)```", inputs, re.IGNORECASE | re.DOTALL)
        # print(matcch_result)
        if matcch_result is None:
            matcch_result = re.search(r"```python(.*)", inputs, re.IGNORECASE | re.DOTALL)
            if matcch_result is None:
                return None
            else:
                return matcch_result.group(1).strip("```").strip()
        else:
            return matcch_result.group(1).strip("```").strip()
    @staticmethod
    def exec_code(code):
        logger = logging.getLogger(os.environ.get('logger_name'))
        output_buffer = io.StringIO()

        # Save the current stdout so we can restore it later
        original_stdout = sys.stdout

        # Redirect stdout to the buffer
        sys.stdout = output_buffer

        # Execute the code
        # logger.info("代码为：\n{code}".format(code=code))
        exec(code.strip().strip("```"))

        # Get the captured output
        captured_output = output_buffer.getvalue()

        # Restore original stdout
        sys.stdout = original_stdout

        # Print the captured output
        return captured_output.replace("\n", " ")