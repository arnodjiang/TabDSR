import os
import time
from datetime import datetime

import argparse
import asyncio

from ImTqa.evaluate import imTqaEvaluator
from FetaQa.evaluate import fetaQaEvaluator
from TableBench.evaluate import tableBenchEvaluator
from AitQa.evaluate import aitQaEvaluator
from MultiHiertt.evaluate import multiHierttEvaluator

_DATAMAP = {
    "ImTqa": imTqaEvaluator,
    "FetaQa": fetaQaEvaluator,
    "TableBench": tableBenchEvaluator,
    "AitQa": aitQaEvaluator,
    "MultiHiertt": multiHierttEvaluator
}

def init_args():
    parser = argparse.ArgumentParser(description='Test for argparse, python llm_eva.py -d TableBench -m qwen2.7:32b -a', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--Datasetname', '-d', help=' 数据集名称')
    parser.add_argument('--ModelName', '-m', help='模型名,当前支持：qwen2.5:32B,qwen2.5:7B,qwen2.5-math:7b')
    parser.add_argument('--IsAsync', '-a', action="store_true", help='是否使用 async')
    args = parser.parse_args()
    return args

args = init_args()

def main(args):
    evaluator = _DATAMAP[args.Datasetname]
    if args.IsAsync is False:
        # evaluator.evaluatePipeline(model_name=args.ModelName)
        evaluator.inference(modelName=args.ModelName,local=True)
    else:
        asyncio.run(evaluator.evaluatePipelineWithAsync(model_name=args.ModelName))

if __name__ == "__main__":
    main(args)