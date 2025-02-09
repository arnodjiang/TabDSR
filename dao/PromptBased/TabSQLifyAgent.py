from dao.Agents.BaseAgent import BaseAgent
from dao.LLMCaller import llmCaller
from sqlalchemy import create_engine

from pandasql import sqldf

import ast
import os
from typing import Dict, List
import pandas as pd
import json

from openai import OpenAI

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

client = OpenAI()

class TabSQLifyAgent:
    def __init__(self):
        self.colRowPrompt = BaseAgent.load_prompt("./prompts/TabSQLify/TabSQLifyColRowPrompt.txt")
        self.answerGeneration = BaseAgent.load_prompt("./prompts/TabSQLify/TabSQLifyAnswerGeneration.txt")

    def infer(self, tabledf, question):
        """
        code from https://github.com/mahadi-nahid/TabSQLify/blob/main/run_wtq_A.py
        """
        colrowInstruction = self.colRowPrompt
        table_json = json.loads(tabledf.to_json(orient='split'))
        T = self.dict2df(table_json)
        T = T.assign(row_number=range(len(T)))
        row_number = T.pop('row_number')
        T.insert(0, 'row_number', row_number)
        col = T.columns

        # print('Table Coll: ', col)
        tab_col = ""
        for c in col:
            tab_col += c + ", "
        tab_col = tab_col.strip().strip(',')

        engine = create_engine('sqlite:///database.db')
        # T = prepare_df_for_neuraldb_from_table(table)
        T = self.convert_df_type(T)


        sql_3 = """select * from T limit 3"""
        three_row = sqldf(sql_3, locals())
        three_row = self.table_linearization(three_row, style='pipe')
        # print('\nThree example rows: \n', str(three_row))

        sql, result, response, output_ans, linear_table = tabsqlify_wtq(T, "", tab_col, question, three_row, selection='rc')

        t_num_cell = T.size
        r_num_cell = result.size
        print('R num_cell: ', r_num_cell, 'T num_cell: ', t_num_cell)


        output_ans = output_ans.lower()
        print('\nResponse: ', response, '\nGen output: ', output_ans, 'Gold: ', answer)

        ## This is not official evaluation. ..... 
        if output_ans.strip() == answer or output_ans.strip().find(answer) != -1 \
                or answer.strip().find(output_ans.strip()) != -1:
            correct += 1
            print("correct: ", correct)

    def convert_df_type(self, df: pd.DataFrame, lower_case=True):
        """
        A simple converter of dataframe data type from string to int/float/datetime.
        """

        def get_table_content_in_column(table):
            if isinstance(table, pd.DataFrame):
                header = table.columns.tolist()
                rows = table.values.tolist()
            else:
                # Standard table dict format
                header, rows = table['header'], table['rows']
            all_col_values = []
            for i in range(len(header)):
                one_col_values = []
                for _row in rows:
                    one_col_values.append(_row[i])
                all_col_values.append(one_col_values)
            return all_col_values

        # Rename empty columns
        new_columns = []
        for idx, header in enumerate(df.columns):
            if header == '':
                new_columns.append('FilledColumnName')  # Fixme: give it a better name when all finished!
            else:
                new_columns.append(header)
        df.columns = new_columns

        # Rename duplicate columns
        new_columns = []
        for idx, header in enumerate(df.columns):
            if header in new_columns:
                new_header, suffix = header, 2
                while new_header in new_columns:
                    new_header = header + '_' + str(suffix)
                    suffix += 1
                new_columns.append(new_header)
            else:
                new_columns.append(header)
        df.columns = new_columns

        # Recognize null values like "-"
        null_tokens = ['', '-', '/']
        for header in df.columns:
            df[header] = df[header].map(lambda x: str(None) if x in null_tokens else x)

        # Convert the null values in digit column to "NaN"
        all_col_values = get_table_content_in_column(df)
        for col_i, one_col_values in enumerate(all_col_values):
            all_number_flag = True
            for row_i, cell_value in enumerate(one_col_values):
                try:
                    float(cell_value)
                except Exception as e:
                    if not cell_value in [str(None), str(None).lower()]:
                        # None or none
                        all_number_flag = False
            if all_number_flag:
                _header = df.columns[col_i]
                df[_header] = df[_header].map(lambda x: "NaN" if x in [str(None), str(None).lower()] else x)

        # Normalize cell values.
        for header in df.columns:
            df[header] = df[header].map(lambda x: str_normalize(x))

        # Strip the mis-added "01-01 00:00:00"
        all_col_values = get_table_content_in_column(df)
        for col_i, one_col_values in enumerate(all_col_values):
            all_with_00_00_00 = True
            all_with_01_00_00_00 = True
            all_with_01_01_00_00_00 = True
            for row_i, cell_value in enumerate(one_col_values):
                if not str(cell_value).endswith(" 00:00:00"):
                    all_with_00_00_00 = False
                if not str(cell_value).endswith("-01 00:00:00"):
                    all_with_01_00_00_00 = False
                if not str(cell_value).endswith("-01-01 00:00:00"):
                    all_with_01_01_00_00_00 = False
            if all_with_01_01_00_00_00:
                _header = df.columns[col_i]
                df[_header] = df[_header].map(lambda x: x[:-len("-01-01 00:00:00")])
                continue

            if all_with_01_00_00_00:
                _header = df.columns[col_i]
                df[_header] = df[_header].map(lambda x: x[:-len("-01 00:00:00")])
                continue

            if all_with_00_00_00:
                _header = df.columns[col_i]
                df[_header] = df[_header].map(lambda x: x[:-len(" 00:00:00")])
                continue

        # Do header and cell value lower case
        if lower_case:
            new_columns = []
            for header in df.columns:
                lower_header = str(header).lower()
                if lower_header in new_columns:
                    new_header, suffix = lower_header, 2
                    while new_header in new_columns:
                        new_header = lower_header + '-' + str(suffix)
                        suffix += 1
                    new_columns.append(new_header)
                else:
                    new_columns.append(lower_header)
            df.columns = new_columns
            for header in df.columns:
                # df[header] = df[header].map(lambda x: str(x).lower())
                df[header] = df[header].map(lambda x: str(x).lower().strip())

        # Recognize header type
        for header in df.columns:

            float_able = False
            int_able = False
            datetime_able = False

            # Recognize int & float type
            try:
                df[header].astype("float")
                float_able = True
            except:
                pass

            if float_able:
                try:
                    if all(df[header].astype("float") == df[header].astype(int)):
                        int_able = True
                except:
                    pass

            if float_able:
                if int_able:
                    df[header] = df[header].astype(int)
                else:
                    df[header] = df[header].astype(float)

            # Recognize datetime type
            try:
                df[header].astype("datetime64")
                datetime_able = True
            except:
                pass

            if datetime_able:
                df[header] = df[header].astype("datetime64")

        return df


    ## from https://github.com/mahadi-nahid/TabSQLify/blob/main/utils/preprocess.py#L8
    def dict2df(self, table: Dict, add_row_id=False, lower_case=True):
        """
        Dict to pd.DataFrame.
        tapex format.
        """
        header, rows = table[0], table[1:]
        # print('header before : ', header)
        header = self.preprocess_columns(header)
        # print('header after: ', header)
        df = pd.DataFrame(data=rows, columns=header)
        return df


    def table_linearization(self,table: pd.DataFrame, style: str = 'pipe'):
        """
        linearization table according to format.
        """
        assert style in ['pipe', 'row_col']
        linear_table = ''
        if style == 'pipe':
            header = ' | '.join(table.columns) + '\n'
            linear_table += header
            rows = table.values.tolist()
            # print('header: ', linear_table)
            # print(rows)
            for row_idx, row in enumerate(rows):
                # print(row)
                line = ' | '.join(str(v) for v in row)
                # print('line: ', line)
                if row_idx != len(rows) - 1:
                    line += '\n'
                linear_table += line

        elif style == 'row_col':
            header = 'col : ' + ' | '.join(table.columns) + '\n'
            linear_table += header
            rows = table.values.tolist()
            for row_idx, row in enumerate(rows):
                line = 'row {} : '.format(row_idx + 1) + ' | '.join(row)
                if row_idx != len(rows) - 1:
                    line += '\n'
                linear_table += line
        return linear_table


    def strip_tokens(self,table_string, sep):
        rows = table_string.strip().split('\n')
        stripped_data = []
        for row in rows:
            tokens = row.split('|')
            stripped_tokens = [token.strip() for token in tokens]
            stripped_data.append(stripped_tokens)

        formatted_table = []
        for row in stripped_data:
            formatted_row = sep.join(row)
            formatted_table.append(formatted_row)

        return '\n'.join(formatted_table)


    def preprocess_columns(self,columns):
        # columns = table.split('\n')[0].split('|')
        # print('preprocessing columns')
        tab_coll = []
        illegal_chars_1 = [' ', '/', '\\', '-', ':', '#', '%']
        illegal_chars_2 = ['.', '(', ')', '[', ']', '{', '}', '*', '$', ',', '?', '!', '\'', '$', '@', '&', '=',
                        '+']
        for x in columns:
            x = x.strip()
            # print(x)
            if x.isnumeric():
                x = "_" + x
            x = x.replace(">", "GT")
            x = x.replace("<", "LT")
            x = x.replace("\\n", "_")
            x = x.replace("\n", "_")
            x = x.replace('\\', '_')
            for char in illegal_chars_1:
                x = x.replace(char, '_')
            for char in illegal_chars_2:
                x = x.replace(char, '')
            tab_coll.append(x.strip())

        counts = {}
        preprocessed_colls = []
        for item in tab_coll:
            if item in counts:
                counts[item] += 1
                preprocessed_colls.append(f"{item}{counts[item]}")
            else:
                counts[item] = 0
                preprocessed_colls.append(item)

        return preprocessed_colls


    def extract_elements_from_string(self,string_list):
        try:
            # Safely evaluate the string as a Python expression
            elements = ast.literal_eval(string_list)

            if isinstance(elements, list):
                return elements
            else:
                return []
        except (SyntaxError, ValueError):
            return []


    def count_cells(self,table: str) -> int:
        rows = table.split('\n')
        cell_count = 0
        for row in rows:
            cells = row.split('|')
            cell_count += len(cells)
        return cell_count
    
    def gen_table_decom_prompt(self, title, tab_col, question, examples, selection='rc'):
        if selection == 'col':
            prompt = "" + p_col_three_wtq
        elif selection == 'row':
            prompt = "" + p_row_three_wtq
        elif selection == 'rc':
            prompt = "" + p_rc_three_wtq
        elif selection == 'sql':
            prompt = "" + p_sql_three_wtq
        prompt += "\nSQLite table properties:\n\n"
        prompt += "Table: " + title + " (" + str(tab_col) + ")" + "\n\n"
        prompt += "3 example rows: \n select * from T limit 3;\n"
        prompt += examples + "\n\n"
        prompt += "Q: " + question + "\n"
        prompt += "SQL:"
        return prompt

    def get_completion(self,prompt, model="gpt-3.5-turbo", temperature=0.7, n=1):
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            n=n,
            stream=False,
            max_tokens=200,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["Table:", "\n\n\n"]
        )
        return response.choices[0].message.content

    def get_sql_3(self,prompt):
        response = None
        while response is None:
            try:
                response = get_completion(prompt, temperature=0)
            except:
                time.sleep(2)
                pass
        return response

    def tabsqlify_wtq(self, T, title, tab_col, question, three_row, selection='rc'):
        # ----------------------------------------------------------------------------------------------
        # selection = ['col', 'row', 'rc', 'sql']
        prompt = self.gen_table_decom_prompt(title, tab_col, question, three_row, selection=selection)
        # print(prompt)
        sql = get_sql_3(prompt)
        # sql = sql.split('where')[0]
        print('\nM1: ', sql, '\n')

        response = ""
        output_ans = ""
        linear_table = ""

        result = pd.DataFrame()
        try:
            result = sqldf(sql, locals())
        except:
            # print('error --> id: ', i, ids)
            # empty_error_ids.append(i)
            output_ans = "error"
            # continue

        if result.shape == (1, 1):
            result_list = result.values.tolist()
            # print('M1 - Result List: ', result_list, type(result_list))

            output_ans = ""
            for row in result_list:
                for coll in row:
                    output_ans += str(coll) + " "
                    # print(coll)
            response = "direct ans"
            output_ans = output_ans.lower()
            print('Direct ans: ', output_ans, 'Gold: ', answer)
            # continue
        elif not result.empty:
            # result_list = [result.columns.values.tolist()] + result.values.tolist()
            # print('M1 - Result List: ', result_list, type(result_list))

            linear_table = table_linearization(result, style='pipe')
            # print('M1 - Linear Table: \n', linear_table)

            prompt_ans = generate_sql_answer_prompt(title, sql, linear_table, question)
            print('promt_ans:\n', prompt_ans)
            response = get_answer(prompt_ans)
            print('response: ', response)
            try:
                output_ans = response.split("Answer:")[1]
                # print('Output answer: ', output_ans)
            except:
                print("Error: Answer generation.")
                output_ans = "" + response
            match = re.search(r'(The|the) answer is ([^\.]+)\.$', output_ans)
            if match:
                output_ans = match.group(2).strip('"')
            print('\nAnswer gen output: ', output_ans, 'Gold: ', answer)

        else:
            print('empty. id --> ', i, id)
            empty_error_ids.append(i)
            prompt = gen_table_decom_prompt(title, tab_col, question, three_row, selection='col')
            sql = get_sql_3(prompt)
            # sql = sql.split('where')[0]
            print('col sql: ', sql)
            try:
                result = sqldf(sql, locals())
            except:
                print('col selection - empty/error')
            if not result.empty and result is not None:
                linear_table = table_linearization(result, style='pipe')
            else:
                sql = "select * from T"
                result = sqldf(sql, locals())
                linear_table = table_linearization(result, style='pipe')

            prompt_ans = generate_sql_answer_prompt(title, sql, linear_table, question)
            print('promt_ans:\n', prompt_ans)
            response = get_answer(prompt_ans)
            print('response: ', response)
            try:
                output_ans = response.split("Answer:")[1]
                # print('Output answer: ', output_ans)
            except:
                print("Error: Answer generation.")
                output_ans = "" + response
            match = re.search(r'(The|the) answer is ([^\.]+)\.$', output_ans)
            if match:
                output_ans = match.group(2).strip('"')
            print('\nAnswer gen output: ', output_ans, 'Gold: ', answer)


        return sql, result, response, output_ans, linear_table