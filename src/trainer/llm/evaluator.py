"""
Evaluator 
"""
import re
import torch
import transformers

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