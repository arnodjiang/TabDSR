import os
from args.init_args import init_args
from logger import Logger


from dao.TableBench.TableBenchCallLLM import tableBenchCallLLM

import os

current_pid = os.getpid()

print(f"The current process ID is: {current_pid}")
_call_llm_engine = {
    "TableBench": tableBenchCallLLM,
    "CalTab151": tableBenchCallLLM,
    "TatQa": tableBenchCallLLM
}

def main(args):
    if "/" in args.model_name:
        toname = args.model_name.replace("/","_")
    toPath = os.path.join(args.to_path, f"{args.agent_mode}_{args.dataset_name}_{toname}.json")
    print(toPath)
    os.makedirs(args.to_path, exist_ok=True)

    _call_llm_engine[args.dataset_name].inference(model_name=args.model_name, to_path=toPath, engine=args.engine, head=args.head, agent_mode=args.agent_mode, tablebenchMode=args.tablebenchMode, dataset_name=args.dataset_name)

if __name__ == "__main__":
    
    main(init_args())