"""
Register stage and import all the necessary packages / dependencies. 

Fetch arguments, define models, call necessary trainers, executors, etc.
"""

import os
import yaml
import logging
import torch

from src.models.auto_map import ModelMap

class Execute:
    """
    Costruct the starting point of all the executions for Torch2Chip, including model configuration and necessary module fetching (e.g., trainer)

    Args:
        config: configuration defined in an external .yaml file.
        pretrained_checkpoint: pre-trained checkpoint of the target model
    """
    def __init__(
        self,
        config_dir,
        ):

        self.config_dir = config_dir
        self.config = self.prepare_config()
        self.run_dir = self.config["save"]["run_dir"]

        # detect device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # run dir
        self.register_run_dir()

        # initialize logging
        self.logger = self.initialize_logger()
    
    def __name__(self):
        return "Execute"
    
    def register_run_dir(self):
        if not os.path.isdir(self.run_dir):
            os.makedirs(self.run_dir, exist_ok=True)

    def prepare_config(self):
        with open(self.config_dir, 'r') as f:
            config = yaml.full_load(f)
        return config
    
    def initialize_logger(self):
        logname = self.config["save"]["logger"]
        logpath = os.path.join(self.run_dir, logname)

        logger = logging.getLogger(logname)
        logger.setLevel(logging.DEBUG)

        if not logger.handlers:
            file_handler = logging.FileHandler(logpath, mode="w")
            console_handler = logging.StreamHandler()
            
            file_handler.setLevel(logging.DEBUG)
            console_handler.setLevel(logging.INFO)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        return logger

    def create_model(self):
        # mapper
        model_type = self.config["model"]["model_type"]
        print(f"Creating model {model_type}...")

        model_func = ModelMap(model_type)
        model = model_func.fetch()

        # map to device
        model.to(self.device)
        return model

    def output(self):
        """
        Output of the execution stage. Default: Save model state_dict
        """
        model_dict = self.model.state_dict()
        model_path = os.path.join(self.run_dir, "latest_model.pth.tar")

        torch.save(model_dict, model_path)

    def print_arch(self, model:torch.nn.Module, name:str):
        path = os.path.join(self.run_dir, name+".txt")
        
        with open(path, "w") as file:
            print(model, file=file)

    
    def run(self):
        """
        Entrance of execution
        """
        self.logger.info(f"Start stage {self.name}")