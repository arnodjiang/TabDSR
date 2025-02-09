from transformers import AutoModelForCausalLM, AutoTokenizer
from dao.Tools import Tools
from loguru import logger
import transformers, torch

from openai import OpenAI
import os
# import tensorflow as tf
# tf.experimental.numpy.experimental_enable_numpy_behavior()
class LLmCaller:
    def __init__(self) -> None:
        self.ModelMaphf = {}
        self.model = None
        self.tokenizer = None
        self.vllm_sampling_params = None

        self.agent2model = None
        self.agent2tokenizer = None

        

        self.client = OpenAI(base_url=os.environ.get("OPENAI_API_BASE"))

    def init_model(self, model_name, **kwargs):
        if self.model is None or self.tokenizer is None:
            if model_name == "microsoft/tapex-large-finetuned-wtq":
                from transformers import TapexTokenizer, BartForConditionalGeneration
                model_path = self.ModelMaphf[model_name]
                self.tokenizer = TapexTokenizer.from_pretrained(model_path)
                self.model = BartForConditionalGeneration.from_pretrained(model_path)
                return
            elif model_name == "google/tapas-large-finetuned-wtq":
                from transformers import AutoTokenizer, TapasForQuestionAnswering
                model_path = self.ModelMaphf[model_name]
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = TapasForQuestionAnswering.from_pretrained(model_path)
                return
            elif model_name == "neulab/omnitab-large-finetuned-wtq":
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                model_path = self.ModelMaphf[model_name]
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                return
            elif self.model is None or self.tokenizer is None:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                if "phi" in model_name.lower():
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.ModelMaphf.get(model_name, ""),
                        torch_dtype="auto",
                        device_map="auto",
                        trust_remote_code=True
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.ModelMaphf.get(model_name, ""),
                        torch_dtype="auto",
                        device_map="auto"
                    )
                logger.info(f"Model device: {next(self.model.parameters()).device}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.ModelMaphf[model_name])

        if kwargs.get("agent_mode") and self.agent2model is None:
            if model_name == "microsoft/tapex-large-finetuned-wtq":
                from transformers import TapexTokenizer, BartForConditionalGeneration
                model_path = self.ModelMaphf[model_name]
                self.agent2tokenizer = TapexTokenizer.from_pretrained(model_path)
                self.agent2model = BartForConditionalGeneration.from_pretrained(model_path)
                return
            elif model_name == "google/tapas-large-finetuned-wtq":
                from transformers import AutoTokenizer, TapasForQuestionAnswering
                model_path = self.ModelMaphf[model_name]
                self.agent2tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.agent2model = TapasForQuestionAnswering.from_pretrained(model_path)
                return
            elif model_name == "neulab/omnitab-large-finetuned-wtq":
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                model_path = self.ModelMaphf[model_name]
                self.agent2tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.agent2model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                return
            elif self.agent2model is None or self.agent2tokenizer is None:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                if "phi" in model_name.lower():
                    self.agent2model = AutoModelForCausalLM.from_pretrained(
                        self.ModelMaphf.get(model_name, ""),
                        torch_dtype="auto",
                        device_map="auto",
                        trust_remote_code=True
                    )
                else:
                    self.agent2model = AutoModelForCausalLM.from_pretrained(
                        self.ModelMaphf.get(model_name, ""),
                        torch_dtype="auto",
                        device_map="auto"
                    )
                logger.info(f"Model device: {next(self.model.parameters()).device}")
                self.agent2tokenizer = AutoTokenizer.from_pretrained(self.ModelMaphf[model_name])


    def infer(self, messages, model_name, tool=False, engine="hf", **kwagrs):
        """
        主推理接口，有以下功能：
          1.调用 ollama 或者 qwen
        """

        TableTitle=kwagrs.get("TableTitle")
        if engine != "openai":
            self.init_model(model_name)
        if engine == "hf":
            response = self._call_hf(messages, model_name, tool=tool, dataMethod=kwagrs["dataMethod"], agent_mode=kwagrs.get("agent_mode", None))
        elif engine == "openai":
            response = self._call_openai(messages, model_name)
        elif engine == "llama":
            response = self._call_llama(messages, model_name)
        elif engine == "mistral":
            response = self._call_mistral(messages, model_name)
        elif engine == "TableGPT2-7B":
            response = self._call_tablegpt2(messages, model_name)
        elif engine == "tablellm":
            response = self._call_tablellm(messages, model_name, tabledf=kwagrs["tabledf"], question=kwagrs["question"], dataMethod=kwagrs["dataMethod"], TableTitle=TableTitle)
        elif engine == "Phi-3":
            response = self._call_phi3(messages, model_name,dataMethod=kwagrs["dataMethod"], instruction=kwagrs["instruction"])
        elif engine == "tablellama":
            response = self._call_tablellama(messages, model_name, dataMethod=kwagrs["dataMethod"])
        elif engine == "codellama":
            response = self._call_codellama(messages, model_name)
        elif engine == "microsoft/tapex-large-finetuned-wtq":
            response = self.__call_tapex(tabledf=kwagrs["tabledf"], question=kwagrs["question"])
        elif engine == "google/tapas-large-finetuned-wtq":
            response = self.__call_tapas(tabledf=kwagrs["tabledf"], question=kwagrs["question"])
        elif engine == "neulab/omnitab-large-finetuned-wtq":
            response = self.__call_omnitab(tabledf=kwagrs["tabledf"], question=kwagrs["question"])
        else:
            response = self._call_ollama(messages, model_name)
        return response
    
    def __call_tapex(self, tabledf, question):
        tabledf = tabledf.astype(str)

        encoding = self.tokenizer(table=tabledf, query=question, return_tensors="pt")

        outputs = self.model.generate(**encoding)

        return ", ".join([i.strip() for i in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)])

    def __call_tapas(self, tabledf, question):
        tabledf = tabledf.astype(str)
        inputs = self.tokenizer(table=tabledf, queries=[question], padding="max_length", return_tensors="tf")
        outputs = self.model(**inputs)

        logits = outputs.logits
        logits_aggregation = outputs.logits_aggregation
        print(logits_aggregation)
        return logits_aggregation

    def __call_omnitab(self, tabledf, question):
        tabledf = tabledf.astype(str)
        logger.debug(tabledf)
        logger.debug(question)
        encoding = self.tokenizer(table=tabledf, query=question, return_tensors='pt')

        outputs = self.model.generate(**encoding)
        logger.debug(outputs)
        return ", ".join([i.strip() for i in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)])

    def _call_codellama(self, input_text, model_name, **kwargs):
        # self.init_model(model_name)
        user_prompt = [i["content"] for i in input_text if i["role"]=="user"][0]
        template = "[INST] {instruction}\n[/INST]"
        # prompt = template.format(instruction=user_prompt)
        tokenizer = AutoTokenizer.from_pretrained(self.ModelMaphf[model_name])
        # tokenizer = AutoTokenizer.from_pretrained(model)
        pipeline = transformers.pipeline(
            "text-generation",
            model=self.ModelMaphf["CodeLlama-7b-Python-hf"],
            torch_dtype=torch.float16,
            device_map="auto",
        )

        sequences = pipeline(
            user_prompt,
            do_sample=True,
            temperature=0.1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=4096,
            pad_token_id=tokenizer.eos_token_id
        )
        res = ""
        for seq in sequences:
            res += seq['generated_text']
            print(f"Result: {seq['generated_text']}")

        return res


    def _call_tablellama(self, input_text, model_name, **kwargs):
        self.init_model(model_name)
        model_inputs = self.tokenizer([input_text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=4096,
            temperature=0.1,
            do_sample=True
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def _call_phi3(self, input_text, model_name, **kwargs):
        self.init_model(model_name)

        message = [
            {"role": "user", "content": kwargs["instruction"]}
        ]
        return self._call_qwen(message)

    def _call_tablellm(self, input_text, model_name, **kwargs):
        self.init_model(model_name)
        csv_data = kwargs["tabledf"].to_csv(index=False)
        table_df, question = kwargs["tabledf"], kwargs["question"]
        TableTitle = kwargs["TableTitle"]
        if TableTitle:
            if kwargs["dataMethod"] == "PoT":
                prompt = "[INST]Below are the first few lines of a CSV file. You need to write a Python program to solve the provided question.\nHeader and first few lines of CSV file:\n{csv_data}\nTable Title:{TableTitle}\nQuestion: {question}[/INST]".format(csv_data=csv_data,question=question)
            elif kwargs["dataMethod"] == "DP":
                prompt = "[INST]Offer a thorough and accurate solution that directly addresses the Question outlined in the [Question]. The answer should follow the format below:\n[Answer Format]\nFinal Answer: AnswerName1, AnswerName2...\n\nEnsure the final answer format is the last output line and can only be in the \"Final Answer: AnswerName1, AnswerName2...\" form, no other form. Ensure the \"AnswerName\" is a number or entity name, as short as possible, without any explanation.\n### [Table Text]\nThere is a table with no title.\n### [Table]\n```\n{table_in_csv}\n```\n\n### [Question]\n{question}\n\n### [Solution][INST/]".format(table_in_csv=csv_data,question=question)
        else:
            if kwargs["dataMethod"] == "PoT":
                prompt = "[INST]Below are the first few lines of a CSV file. You need to write a Python program to solve the provided question.\nHeader and first few lines of CSV file:\n{csv_data}Question: {question}[/INST]".format(csv_data=csv_data,question=question)
            elif kwargs["dataMethod"] == "DP":
                prompt = "[INST]Offer a thorough and accurate solution that directly addresses the Question outlined in the [Question]. The answer should follow the format below:\n[Answer Format]\nFinal Answer: AnswerName1, AnswerName2...\n\nEnsure the final answer format is the last output line and can only be in the \"Final Answer: AnswerName1, AnswerName2...\" form, no other form. Ensure the \"AnswerName\" is a number or entity name, as short as possible, without any explanation.\n### [Table Text]\nThere is a table with no title.\n### [Table]\n```\n{table_in_csv}\n```\n\n### [Question]\n{question}\n\n### [Solution][INST/]".format(table_in_csv=csv_data,question=question)
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        # print(text)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=4096,
            temperature=0.1,
            do_sample=True
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def _call_tablegpt2(self, input_text, model_name, **kwargs):
        self.init_model(model_name)
        logger.info(f"加载{input_text}")

        text = self.tokenizer.apply_chat_template(
            input_text, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


    def _call_mistral(self, input_text, model_name):
        self.init_model(model_name)
        model_inputs = self.tokenizer.apply_chat_template(
                    input_text,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
        ).to(self.model.device)
        # model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
        # inputs.to(self.model.device)
        # print(model_name)
        # print(input_text)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=4096,
            temperature=0.1,
            do_sample=True
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def _call_llama(self, input_text, model_name):
        self.init_model(model_name)

        if "Llama-3.1" in model_name:
            input_ids = self.tokenizer.apply_chat_template(
                input_text,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)
            decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
            logger.info(f"Decoded input text: {decoded_text}")
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=4096,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            return self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        elif model_name == "LLM-Research/Llama-3.3-70B-Instruct":
            input_ids = self.tokenizer.apply_chat_template(
                input_text,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=4096,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            return self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        else:
            input_ids = self.tokenizer.apply_chat_template(
                input_text,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            # model_inputs = self.tokenizer([input_ids], return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=4096,
                eos_token_id=terminators,
                temperature=0.1,
                do_sample=True
            )

            response = outputs[0][input_ids.shape[-1]:]
            return self.tokenizer.decode(response, skip_special_tokens=True)

    def _call_openai(self, input_text, model_name):
        response = self.client.chat.completions.create(
            model=model_name,
            messages=input_text,
            stream=False,
            temperature=0.1
        )
        return response.choices[0].message.content

    def _call_ollama(self, input_text, model_name):
        pass

    def init_vllm_model(self, model_name):
        if self.model is None:
            from vllm import LLM, SamplingParams
            self.vllm_sampling_params = SamplingParams(temperature=0.1,max_tokens=4096)
            self.model = LLM(model=model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _call_hf(self, input_text, model_name, tool=False, **kwargs):
        call_response = None
        # print("input_text",input_text)
        if "qwen" in model_name.lower():
            self.init_model(model_name)
            # self.init_vllm_model(self.ModelMaphf[model_name])
            # call_response = self._call_vllm(input_text, tool=tool, dataMethod=kwargs["dataMethod"])
            
            call_response = self._call_qwen(input_text, tool=tool, dataMethod=kwargs["dataMethod"], agent_mode=kwargs.get("agent_mode", None))
            # print("call_response",call_response)
        return call_response
    def _call_vllm(self, input_text, **kwargs):
        # print(input_text[1]["content"])
        text = self.tokenizer.apply_chat_template(
            input_text,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        # logger.info(self.model.generate("hellp, tell me a strot?"))
        # Print the outputs.
        outputs = self.model.generate(model_inputs, self.vllm_sampling_params)
        print(outputs)
        
        return outputs[0].outputs[0].text

    def _call_qwen(self, input_text, tool=False, **kwargs):
        agent_model = kwargs.get("agent_mode")
        if kwargs.get("agent_mode"):
            logger.info(f"Agent_model 是{agent_model}, 调用 agent2model")
            Mytokenizer = self.agent2tokenizer
            model = self.agent2model
        else:
            logger.info(f"Agent_model 是{agent_model}, 正常调用")
            Mytokenizer = self.tokenizer
            model = self.model

        # print(tools)
        if tool is True:
            text = Mytokenizer.apply_chat_template(
                input_text,
                tokenize=False,
                add_generation_prompt=True,
                tools=[Tools.execute_python_code]
            )
        else:
            text = Mytokenizer.apply_chat_template(
                input_text,
                tokenize=False,
                add_generation_prompt=True
            )
        # print(text)
        model_inputs = Mytokenizer([text], return_tensors="pt").to(model.device)
        # print(text)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=4096,
            temperature=0.1,
            do_sample=True
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = Mytokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

llmCaller = LLmCaller()