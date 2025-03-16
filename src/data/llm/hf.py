"""
HuggingFace dataset
"""
import os
import json
import gzip
import random
import datasets
import pandas as pd

from datasets import load_dataset
from tqdm import tqdm
from src.data.base import DataStage
from src.data.data_utils import load_json_data


class WikiText(DataStage):
    def __init__(self, config_dir):
        super().__init__(config_dir)

    def __name__(self):
        return "WikiText"

    def load_dataset(self):
        trainset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        testset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return (trainset, testset)
    
    def prepare_transform(self):
        raise NotImplementedError("WikiText dataset is not compatible with PyTorch transform")
    
    def run(self):
        trainset, testset = self.load_dataset()
        
        try:
            print(f"Number of samples in the training set = {len(trainset)}, tests set = {len(testset)}")
        except:
            print("[WARNING] Cannot fetch the length of the dataset!")
        
        return trainset, testset

class BaseQA(DataStage):
    def __init__(self, config_dir):
        super().__init__(config_dir)

        # data directory
        self.data_root = self.config["dataset"]["root"]
        self.dataset_name = self.config["dataset"]["name"]
        self.data_path = os.path.join(self.data_root, self.dataset_name)

        # batch size
        self.batch_size = self.config["dataset"]["batch_size"]

    def __name__(self):
        return "BaseQA"
    
    def load_dataset(self):
        trainset = load_json_data(self.data_path, "train")
        testset = load_json_data(self.data_path, "test")

        return (trainset, testset)

    def prepare_transform(self, dataset):
        batches = []
        num_batch = len(dataset)//self.batch_size if len(dataset) % self.batch_size == 0 else len(dataset)//self.batch_size + 1
        for i in range(num_batch):
            batch = dataset[i*self.batch_size: min((i+1)*self.batch_size, len(dataset))]
            batches.append(batch)
        return batches
    
    def run(self):
        trainset, testset = self.load_dataset()
        
        # iterable training batches
        train_batches = self.prepare_transform(trainset)
        test_batches = self.prepare_transform(testset)

        return train_batches, test_batches
    

class PiQA(BaseQA):
    def __init__(self, config_dir):
        super().__init__(config_dir)

    def __name__(self):
        return "PiQA"

class WinoGrande(BaseQA):
    def __init__(self, config_dir):
        super().__init__(config_dir)

    def __name__(self):
        return "WinoGrande"

class BoolQ(BaseQA):
    def __init__(self, config_dir):
        super().__init__(config_dir)

    def __name__(self):
        return "BoolQ"


class OpenBookQA(BaseQA):
    def __init__(self, config_dir):
        super().__init__(config_dir)

    def __name__(self):
        return "OpenBookQA"


class HellaSwag(BaseQA):
    def __init__(self, config_dir):
        super().__init__(config_dir)

    def __name__(self):
        return "HellaSwag"

class ARCe(BaseQA):
    def __init__(self, config_dir):
        super().__init__(config_dir)

    def __name__(self):
        return "ARC-e"
    
class ARCc(BaseQA):
    def __init__(self, config_dir):
        super().__init__(config_dir)

    def __name__(self):
        return "ARC-c"
    
class PileSubset(DataStage):
    """
    Subset of Pile dataset: 10K Samples for PTQ calibration
    """
    def __init__(self, config_dir):
        super().__init__(config_dir)

    def load_dataset(self):
        trainset = datasets.load_dataset("mit-han-lab/pile-val-backup", split="validation")
        trainset = trainset.shuffle(seed=42)

        return (trainset, None)

    def run(self):
        trainset, testset = self.load_dataset()
        
        try:
            print(f"Number of samples in the training set = {len(trainset)}, tests set = {len(testset)}")
        except:
            print(f"[WARNING] Cannot fetch the length of the dataset! {type(trainset)}")

        return trainset, testset

class GSM8K(DataStage):
    def __init__(self, config_dir, tokenizer):
        super().__init__(config_dir)
        self.trainset_path = self.config["dataset"]["train"]
        self.validset_path = self.config["dataset"]["test"]
        self.tokenizer = tokenizer

        # prompt template
        self.prompt_head = "Given the following problem, reason and give a final answer to the problem. \nProblem:"
        self.prompt_tail = "\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n"
        self.ans_trigger = "The final answer is"

        self.cot_flag = self.config["eval"]["cot"]
        self.chain_of_thoughts = self.prepare_cot()

    def wrap_text(self, inputs, is_cot:bool=False):
        inst = self.prompt_head + inputs["instruction"] + self.prompt_tail
        instruction = {"role": "user", "content": inst}

        if is_cot:
            assert "chain" in inputs.keys()
            chain = inputs["chain"]

            cot = {"role": "assistant", "content": chain}
            
            return [instruction, cot]
        else:
            return [instruction]

    def prepare_cot(self):
        """
        8-shot COT.
        """
        chain_of_thoughts = []

        chain_of_thoughts.append({
            "instruction": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
            "chain": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6"
        })
        
        chain_of_thoughts.append({
            "instruction": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            "chain": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5"
        })

        chain_of_thoughts.append({
            "instruction": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
            "chain": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39"
        })

        chain_of_thoughts.append({
            "instruction": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
            "chain": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8"
        })

        chain_of_thoughts.append({
            "instruction": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
            "chain": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9"
        })

        chain_of_thoughts.append({
            "instruction": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
            "chain": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29"
        })

        chain_of_thoughts.append({
            "instruction": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
            "chain": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33"
        })

        chain_of_thoughts.append({
            "instruction": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
            "chain": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8"
        })

        return chain_of_thoughts

    def load_jsonl(self, split):
        is_gzip = ".zip" in self.trainset_path
        
        if split == "train":
            file_path = self.trainset_path
        elif split == "test":
            file_path = self.validset_path
        else:
            raise ValueError("[GSM8K] Unknown dataset split")

        # data list
        collected_data = []

        open_func = open if not is_gzip else gzip.open

        with open_func(file_path, "r") as f:
            for line in f:
                item = json.loads(line)

                # access the data item
                new_item = dict(
                    instruction = item["question"] if "question" in item else None,
                    output = item["answer"] if "answer" in item else None, 
                )

                collected_data.append(new_item)
    
        return collected_data

    def load_dataset(self):
        trainset = self.load_jsonl(split="train")
        validset = self.load_jsonl(split="test")

        return trainset, validset
    
    def wrap_cot(self):
        shuffled_cot = list(range(len(self.chain_of_thoughts)))
        random.shuffle(shuffled_cot)

        instruction = []

        for cot in self.chain_of_thoughts:
            wrapped_cot = self.wrap_text(cot, is_cot=True)
            instruction += wrapped_cot
        
        return instruction
    
    def few_shot_dataset(self, dataset):
        cot_datasets = []
        cot_targets = []
        
        pbar = tqdm(dataset)
        for sample in pbar:
            # chain of thoughts
            cot = self.wrap_cot()

            # context of question
            prepare_text = self.wrap_text(sample, is_cot=False)

            cot += prepare_text
            cot = self.tokenizer.apply_chat_template(cot, tokenize=False, add_generation_prompt=True)
            
            # fetch labels and datasets
            cot_datasets.append(cot)
            cot_targets.append(sample["output"])

        return {
            "dataset": cot_datasets,
            "label": cot_targets
        }
    
    def zero_shot_dataset(self, dataset):
        inputs = []
        targets = []
        
        pbar = tqdm(dataset)
        for sample in pbar:
            
            # context of question
            prepare_text = self.wrap_text(sample, is_cot=False)
            prepare_text = self.tokenizer.apply_chat_template(prepare_text, tokenize=False, add_generation_prompt=True)

            # fetch labels and datasets
            inputs.append(prepare_text)
            targets.append(sample["output"])

        return {
            "dataset": inputs,
            "label": targets
        }

    def run(self):
        trainset, validset = self.load_dataset()
        
        trainset = self.zero_shot_dataset(trainset)
        validset = self.few_shot_dataset(validset)
        return trainset, validset

class MMLU(DataStage):
    """
    MMLU benchmark with 5-shot chain-of-thoughts

    Basic benchmark setup is adopted from: https://github.com/FranxYao/chain-of-thought-hub
    """
    def __init__(self, config_dir, tokenizer):
        super().__init__(config_dir)

        self.trainset_path = self.config["dataset"]["train"]
        self.validset_path = self.config["dataset"]["test"]
        self.tokenizer = tokenizer

        # prompt template
        self.cot_path = self.config["dataset"]["cot"]
        self.nshot = self.config["dataset"]["nshot"]

        # multi-choice
        self.choices = ["A", "B", "C", "D"]
        
    def prepare_cot(self, task:str):
        """
        5-shot CoT
        """
        filename = task + "_dev.csv"
        cot_task_path = os.path.join(self.cot_path, filename)

        chain_of_thoughts = pd.read_csv(cot_task_path, header=None)[:self.nshot]
        
        return chain_of_thoughts
    
    def load_dataset(self, task:str):
        filename = task + "_test.csv"
        valid_task_path = os.path.join(self.validset_path, filename)
        
        validset = pd.read_csv(valid_task_path, header=None)
        return validset
    
    def wrap_cot(self, dataset, idx, include_answer=True):
        prompt = dataset.iloc[idx, 0]
        k = dataset.shape[1] - 2
        for j in range(k):
            prompt += "\n{}. {}".format(self.choices[j], dataset.iloc[idx, j+1])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(dataset.iloc[idx, k + 1])
        return prompt
    
    def format_subject(self, subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s

    def prompt_template(self, cot, subject, k=-1):
        subject_txt = self.format_subject(subject)
        prompt = f"The following are multiple choice questions (with answers) about {subject_txt}.\n\n"

        if k == -1:
            k = cot.shape[0]
        for i in range(k):
            prompt += self.wrap_cot(cot, i)
        return prompt

    def few_shot_dataset(self, task:str):
        cot = self.prepare_cot(task)
        validset = self.load_dataset(task)
        
        cot_datasets = []
        cot_targets = []

        num_samples = validset.shape[0]
        for i in tqdm(range(num_samples)):
            prompt_end = self.wrap_cot(validset, i, include_answer=False)
            train_prompt = self.prompt_template(cot, task, self.nshot)

            prompt = train_prompt + prompt_end
            while len(self.tokenizer.tokenize(prompt)) + 1> 2048: # bos token
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = '\n\n'.join(prompt_split)
            label = validset.iloc[i, validset.shape[1]-1]

            cot_datasets.append(prompt)
            cot_targets.append(label)

        return {
            "dataset": cot_datasets,
            "label": cot_targets
        }
    
    def run(self, task):
        validset = self.few_shot_dataset(task)
        return validset

class MATH500(DataStage):
    def __init__(self, config_dir):
        super().__init__(config_dir)
        self.dataset_name = "HuggingFaceH4/MATH-500"
        self.dataset = load_dataset(self.dataset_name, split="test")

    def wrap_text(self, inputs):
        instruction = [
            {"role": "system", "content": "Let's think step by step. And put your answer after \"Answer: \", and you need to use a \\boxed{} command to wrap your answer."},
            {"role": "user", "content": f"{inputs} Remember to put your final answer after \"Answer: \""}
        ]
        return [instruction]

    def run(self):
        print(f"Check dataset...")
        problems = self.dataset["problem"]
        solution = self.dataset["solution"]
        answers = self.dataset["answer"]

        # dataset placeholder
        dataset, target = [], []

        pbar = tqdm(problems)
        for i, prob in enumerate(pbar):
            prepared_text = self.wrap_text(prob)
            dataset.append(prepared_text)

        return {
            "dataset": dataset,
            "solution": solution,
            "target": answers
        }

class MATH500Step(MATH500):
    def __init__(self, config_dir):
        super().__init__(config_dir)
        # system prompt template
        self.prompt_head: str = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always put your answer after \"Answer: \", and you need to use a \\boxed{} command to wrap your answer."

    def wrap_text(self, inputs):
        inst = self.prompt_head + inputs
        instruction = {"role": "user", "content": inst}
        return [instruction]