import argparse

def init_args():
    parser = argparse.ArgumentParser(description='Test for argparse', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--logger_name', '-ln', help='name 属性，非必要参数')
    parser.add_argument('--model_name', '-mn', help='year 属性，非必要参数，但是有默认值')
    parser.add_argument('--dataset_name', '-dn', help='year 属性，非必要参数，但是有默认值')
    parser.add_argument('--to_path', '-tp', help='year 属性，非必要参数，但是有默认值')
    parser.add_argument('--engine', '-e', help='ollama or local path')
    parser.add_argument('--head', '-hd', type=int, help='取多少内容')
    parser.add_argument("--agent_mode", type=str, help='name 属性，非必要参数')
    parser.add_argument("--tablebenchMode", type=str, default="PoT")
    args = parser.parse_args()
    return args