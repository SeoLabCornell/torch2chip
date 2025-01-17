"""
Evaluator 
"""
import os
import re
import torch
import json

from tqdm import tqdm
from src.stage.base import Execute
from src.trainer.llm.metrics import Perplexity
from src.data.llm import DATA_STAGE_MAP
from src.trainer.llm.utils import stop_sequences_criteria

class WikiText(Execute):
    def __init__(self, config_dir, model, tokenizer):
        super().__init__(config_dir)

        self.model = model
        self.tokenizer = tokenizer

        # prepare dataset
        data_name = self.config["dataset"]["name"]
        self.datastage = DATA_STAGE_MAP[data_name](config_dir)
        trainset, testset = self.datastage.run()

        self.testset = tokenizer(
            "\n\n".join(testset["text"]), return_tensors="pt"
        ).input_ids.to(self.device)

        # metric
        chunk_size = self.config["eval"]["chunk_size"]
        n_samples = self.config["eval"]["n_samples"]
        self.metric = Perplexity(chunk_size, n_samples)

    def __name__(self):
        return "WikiText"

    @torch.no_grad
    def run(self):
        self.logger.info(f"Start evaluating {self.__name__()}...")
        self.model.eval()

        for i in tqdm(range(self.metric.n_samples)):
            batch = self.testset[:, (i * 2048) : ((i + 1) * 2048)].to(self.device)

            # one shot inference
            with torch.no_grad():
                logits = self.model(batch).logits

            
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = self.testset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]

            self.metric.update(shift_logits, shift_labels)

        ppl = self.metric.reduce()
        self.logger.info(f"Perpleixty = {ppl.item():.3f}")

    @torch.no_grad
    def export_run(self, export_samples:int):
        self.logger.info("Export Samples!")

        for i in tqdm(range(export_samples)):
            batch = self.testset[:, (i * 2048) : ((i + 1) * 2048)].to(self.device)
            
            # one shot inference
            with torch.no_grad():
                logits = self.model(batch).logits

class GSM8K(Execute):
    def __init__(self, config_dir, model, tokenizer):
        super().__init__(config_dir)

        self.model = model
        self.tokenizer = tokenizer

        # prepare dataset
        data_name = self.config["dataset"]["name"]
        self.datastage = DATA_STAGE_MAP[data_name](config_dir, tokenizer)
        self.trainset, self.testset = self.datastage.run()

        # condition for end of generation
        self.max_gen_toks = self.config["eval"]["max_gen_toks"]
        self.gen_until = ['<|eot_id|>', '<|start_header_id|>user<|end_header_id|>', 'Q:', '</s>', '<|im_end|>']

    def __name__(self):
        return "GSM8K"
    
    def tokenize(self, prompt:str, truncation=False):
        encoding = self.tokenizer(
            prompt,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False
        )

        # TODO: add left_truncate_len (if necessary)
        return encoding["input_ids"], encoding["attention_mask"]
    
    def generate(self, input_ids, attention_mask):
        max_length = input_ids.shape[1] + self.max_gen_toks
        stop_criteria = stop_sequences_criteria(
                self.tokenizer, self.gen_until, input_ids.shape[1], input_ids.shape[0]
        )

        out = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            stopping_criteria=stop_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            attention_mask=attention_mask,
            do_sample=False
        )

        return out

    def metric(self, model_pred, gt):
        ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
        match = ANS_RE.search(gt)

        gt_str = match.group(1).strip()
        gt_str = gt_str.replace(",", "")

        # extract numerical answer from 
        preds = model_pred.split(self.datastage.ans_trigger.lower())
        valid_ans = True if len(preds) > 1 else False

        if valid_ans:
            pred = preds[1]
        else:
            pred = preds[-1]

        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

        if len(pred) == 0:
            return "[invalid]"

        if valid_ans:
            pred = pred[0]
        else:
            # choose the last element in list
            pred = pred[-1]

        if pred[-1] == ".":
            pred = pred[:-1]
        
        return gt_str == pred

    @torch.no_grad
    def run(self):
        self.logger.info(f"Start evaluating {self.__name__()}...")
        output = []
        self.model.eval()

        for idx, sample in enumerate(tqdm(self.testset["dataset"])):
            input_ids, attn_mask = self.tokenize(sample)
            
            input_ids = input_ids.to(self.device)
            attn_mask = attn_mask.to(self.device)

            # label context
            gt = self.testset["label"][idx]

            # generate
            tok = self.generate(input_ids, attn_mask)
            tok_list = tok.tolist()

            # decoded tokens
            tok = tok_list[0][input_ids.shape[1] :]

            # decode tokens
            dec_tok = self.tokenizer.decode(tok, skip_special_tokens=True)
            
            correctness = self.metric(dec_tok, gt)
            output.append(int(correctness))
            

        avg = sum(output) / len(self.testset["dataset"])
        self.logger.info(f"Average Score (exact match) = {avg:.2f}")
        return output

def compute_metric(output_filename):
    with open(output_filename, 'r') as f:
        run_results = json.load(f)
    total_acc = 0
    total_num = 0
    for task in run_results:
        acc = 0
        pred_answers = run_results[task]['pred_answers']
        gold_answers = run_results[task]['gold_answers']
        for pred, gold in zip(pred_answers, gold_answers):
            if pred == gold: acc += 1
        print("ACC-%s: %.4f" % (task, acc/len(gold_answers)))
        total_acc += acc
        total_num += len(gold_answers)
    print("ACC-all: %.4f" % (total_acc/total_num))


class MMLU(Execute):
    def __init__(self, config_dir, model, tokenizer):
        super().__init__(config_dir)

        self.model = model
        self.tokenizer = tokenizer

        # condition for end of generation
        self.max_gen_toks = self.config["eval"]["max_gen_toks"]

        # dataset stage
        data_name = self.config["dataset"]["name"]
        self.datastage = DATA_STAGE_MAP[data_name](config_dir, self.tokenizer)

        self.sub_task_list = [
            'abstract_algebra',
            'anatomy',
            'astronomy',
            'business_ethics',
            'clinical_knowledge',
            'college_biology',
            'college_chemistry',
            'college_computer_science',
            'college_mathematics',
            'college_medicine',
            'college_physics',
            'computer_security',
            'conceptual_physics',
            'econometrics',
            'electrical_engineering',
            'elementary_mathematics',
            'formal_logic',
            'global_facts',
            'high_school_biology',
            'high_school_chemistry',
            'high_school_computer_science',
            'high_school_european_history',
            'high_school_geography',
            'high_school_government_and_politics',
            'high_school_macroeconomics',
            'high_school_mathematics',
            'high_school_microeconomics',
            'high_school_physics',
            'high_school_psychology',
            'high_school_statistics',
            'high_school_us_history',
            'high_school_world_history',
            'human_aging',
            'human_sexuality',
            'international_law',
            'jurisprudence',
            'logical_fallacies',
            'machine_learning',
            'management',
            'marketing',
            'medical_genetics',
            'miscellaneous',
            'moral_disputes',
            'moral_scenarios',
            'nutrition',
            'philosophy',
            'prehistory',
            'professional_accounting',
            'professional_law',
            'professional_medicine',
            'professional_psychology',
            'public_relations',
            'security_studies', 
            'sociology',
            'us_foreign_policy',
            'virology',
            'world_religions'
        ]

    def tokenize(self, prompt):
        encoding = self.tokenizer.batch_encode_plus([prompt], return_tensors="pt", padding=True)
        return encoding["input_ids"], encoding["attention_mask"]
    
    def generate(self, input_ids, attention_mask):
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1, 
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,
            temperature=1.0,
            top_p=1.0
        )

        return out

    def metric(self, model_pred, gt):
        answers = model_pred[-1]
        return answers == gt

    def run(self):
        output = []
        self.model.eval()

        acc_avg = []
        run_results = {}

        for task in self.sub_task_list:
            self.logger.info(f"\nStart Evaluating Task: {task}")
            testset = self.datastage.run(task)

            pred, golden_output = [], []

            pbar = tqdm(testset["dataset"])
            for idx, sample in enumerate(pbar):
                input_ids, attn_mask = self.tokenize(sample)

                input_ids = input_ids.to(self.device)
                attn_mask = attn_mask.to(self.device)

                # label context
                gt = testset["label"][idx]

                # generate
                tok = self.generate(input_ids, attn_mask)
                dec_tok = self.tokenizer.batch_decode(tok, skip_special_tokens=True)

                pred.append(dec_tok[0][-1])
                golden_output.append(gt)

                correctness = self.metric(dec_tok[0], gt)
                output.append(int(correctness))

                acc = sum(output) / len(output)
                pbar.set_description(f"Accuracy: {acc:.4f}")

                acc_avg.append(acc)

            run_results[task] = {'pred_answers':pred, 'gold_answers':golden_output}

        output_filename = os.path.join(self.run_dir, "accuracy.json")
        
        with open(output_filename, 'w') as f:
            json.dump(run_results, f, ensure_ascii=False, indent=2)
        
        compute_metric(output_filename)

        return output