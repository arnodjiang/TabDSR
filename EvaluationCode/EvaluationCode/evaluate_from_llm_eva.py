import os
import time
from datetime import datetime

import argparse

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
    parser = argparse.ArgumentParser(description='Test for argparse', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--Datasetname', '-d', help='数据集名称')
    parser.add_argument('--ResPath', '-p', help='数据集评估路径')
    args = parser.parse_args()
    return args

args = init_args()

def main(args):
    evaluator = _DATAMAP[args.Datasetname]
    # evaluator.evaluateResultForCompute(args.ResPath)
    evaluator.extractResultForCompute(args.ResPath)
    

if __name__ == "__main__":
    main(args)