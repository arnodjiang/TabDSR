import re, io, sys
import json
import pandas as pd
class Tools:
    @staticmethod
    def execute_python_code(python_code: str, table_data: dict):
        """Execute Python code to print the answers.

        Args:
            python_code: Python code to execute for answering a question, provided as a string and load the table with command ```table_df = pd.DataFrame(table_data['data'], columns=table_data['columns'])```
            table_data: A dictionary containing 'columns' (list of column names) and 'data' (list of rows) that will be loaded into a pandas DataFrame.
        
        Returns:
            None. The function prints the result of executing the Python code on the table data.
        """
        # Load table data into a DataFrame
        table_df = pd.DataFrame(table_data['data'], columns=table_data['columns'])

        # Execute the provided Python code
        try:
            # Safety check: here we would ideally want to ensure the code is safe to execute.
            # You can implement a basic check for allowed commands to mitigate code injection.
            output_buffer = io.StringIO()

            # Save the current stdout so we can restore it later
            original_stdout = sys.stdout

            # Redirect stdout to the buffer
            sys.stdout = output_buffer

            exec(python_code.strip().strip("```"))

            # Get the captured output
            captured_output = output_buffer.getvalue()

            # Restore original stdout
            sys.stdout = original_stdout

            # Print the captured output
            return captured_output.replace("\n", " ")
        except Exception as e:
            print(f"Error executing code: {e}")
        return "-"


    @staticmethod
    def try_parse_tool_calls(content: str, **kwargs):
        """Try parse the tool calls."""
        tool_calls = []
        offset = 0
        results = []
        for i, m in enumerate(re.finditer(r"<tool_call>\n(.+)?\n</tool_call>", content)):
            if i == 0:
                offset = m.start()
            try:
                func = json.loads(m.group(1))
                tool_calls.append({"type": "function", "function": func})
                func_name = func["name"]
                arguments = func["arguments"]
                arguments.update(kwargs)
                result = Tools.execute_python_code(**arguments)
                results.append({"status": "success", "result": result.strip()})
                if isinstance(func["arguments"], str):
                    func["arguments"] = json.loads(func["arguments"])
            except json.JSONDecodeError as e:
                print(f"Failed to parse tool calls: the content is {m.group(1)} and {e}")
                results.append({"status": "fail", "result": e})
                pass
            except Exception as e:
                print(f"Failed to parse tool calls: the content is {m.group(1)} and {e}")
                results.append({"status": "fail", "result": e})
        if tool_calls:
            if offset > 0 and content[:offset].strip():
                c = content[:offset]
            else: 
                c = ""
            return results
        return results