## conda activate tableDepo && python /raid/share/jiangchangjiang/tablellmPipeline/EvaluationCode/TableBench/vsiual_from_id.py --id b71bb2ae2d5e19e17c816355f55ec3d8

import argparse
import json
import pandas as pd
import re
from pprint import pprint

parser = argparse.ArgumentParser(description='tablebench的表格可视化', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--id', help='name 属性，非必要参数')
args = parser.parse_args()

tableResource = "/raid/share/jiangchangjiang/tablellmPipeline/EvaluationCode/TableBench/RawData/TableBench_TCoT.jsonl"

with open(tableResource, 'r', encoding='utf-8') as file:
    data = {json.loads(line.strip())["id"]: json.loads(line.strip()) for line in file}

tableColumnFix = [re.sub(r'-+', '-', i.strip().replace("\\n", "-").replace("\n", "-").replace(" ","-")) for i in json.loads(data[args.id]["table"])["columns"] if i.strip()]
table_df = pd.DataFrame(json.loads(data[args.id]["table"])['data'], columns=tableColumnFix)

pprint(table_df)
pprint(table_df.columns)