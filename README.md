# Official Implementation for the Paper: 
## **TabDSR: Decompose, Sanitize, and Reason for Complex Numerical Reasoning in Tabular Data**

We strongly recommend referring to our run example to implement our results.

## Environment
```bash
conda create -n tabdsr python=3.11
conda activate tabdsr
pip install -r requirements.txt
```

## Generate Result of Different Models

Command-Line Arguments:
--agent_mode: Specifies the mode (DP, PoT, or CoT). You can refer to our example.
--tablebenchMode: Same as --agent_mode.
-mn: Model name, which needs to be modified in ./dao/LLMCaller.py.
-dn: Dataset name. Choose from ["TableBenchFix", "TatQa", "CalTab151"].
-tp: Save path.
-e: Load mode. You can refer to our example.
-ln: Logger name.

## Examples

### Running DP Prompt

```bash
python infer.py --agent_mode raw --tablebenchMode DP -mn meta-llama/Llama-2-7b-chat-hf -dn TableBenchFix -tp ./Archive/{DatasetName}/meta-llama/Llama-2-7b-chat-hf -e llama -ln test
```

### Running CoT Prompt

```bash
python infer.py --agent_mode raw --tablebenchMode CoT -mn meta-llama/Llama-2-7b-chat-hf -dn TableBenchFix -tp ./Archive/{DatasetName}/meta-llama/Llama-2-7b-chat-hf -e llama -ln test
```

### Running TabDSR

Local model path

```bash
python infer.py --agent_mode 1+2+3 --tablebenchMode TCoT -mn qwen2.5 -dn TableBenchFix -tp ./Archive/{DatasetName}/qwen2.5 -e qwen -ln test
```

### openai, deepseek or VLLM (recommend)

DeepSeek:

```bash
OPENAI_API_KEY="Your key" OPENAI_API_BASE="https://api.deepseek.com" python infer.py --agent_mode 1+2+3 --tablebenchMode PoT -mn deepseek-chat -dn TableBenchFix -tp ./Archive/deepseek-chat -e openai -ln test
```

```bash
OPENAI_API_KEY="Your key" python infer.py --agent_mode 1+2+3 --tablebenchMode PoT -mn gpt-4o -dn TableBenchFix -tp ./Archive/{DatasetName} -e openai -ln test
```

VLLM:

```bash
OPENAI_API_KEY="Your key" OPENAI_API_BASE="VLLM url" python infer.py --agent_mode 1+2+3 --tablebenchMode PoT -mn Qwen/Qwen2.5-7B-Instruct -dn TableBenchFix -tp ./Archive/{DatasetName} -e openai -ln test
```

## Evaluation

- dp: The path of the result file.
- dn: Dataset name. Choose from ["TableBench", "TatQa", "CalTab151"].

For example:

```bash
python ./evaluates/Evaluator.py -dp ./Archive/deepseek-chat/raw_TatQa_deepseek-ai_DeepSeek-V3.json -dn TableBench
```