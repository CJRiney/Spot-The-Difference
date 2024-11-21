'''
CUDA_VISIBLE_DEVICES=1,2 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ../configs/train_config.yaml train.py ../configs/snn_config.yaml
'''

import logging
import sys, os
import timm

from alignment import H4ArgumentParser, ModelArguments, DataArguments

from pathlib import Path

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from SNNTrainer import SNNTrainer
from SNNDataset import SNNDataset

from transformers import Trainer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.configuration_SNN import SNN
from configs.dataclasses import SNNTrainingArguments

from configs.log_config import logger

logger.setLevel(logging.INFO)

def main():
    ########################
    # Parse configurations #
    ########################
    parser = H4ArgumentParser((ModelArguments, DataArguments, SNNTrainingArguments))
    model_args, data_args, trainer_args = parser.parse()
    
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {trainer_args}")
    
    base_model = timm.create_model(trainer_args.model_name, pretrained=True) 
    data_config = resolve_data_config(base_model.pretrained_cfg, model=base_model)
    transform = create_transform(**data_config)
    
    logger.info(f'Training SNN for {trainer_args.model_name}')
    
    model = SNN(trainer_args)
    trainer = SNNTrainer(model=model)
    
    train_root_path = Path(trainer_args.train_dataset_path)
    len_train_dataset = sum(1 for path in train_root_path.iterdir() if path.is_dir())
    
    eval_root_path = Path(trainer_args.eval_dataset_path)
    len_eval_dataset = sum(1 for path in eval_root_path.iterdir() if path.is_dir())
    
    logger.info(f'Training on {len_train_dataset} datapoints from {trainer_args.train_dataset_path}')
    logger.info(f'Evaluating on {len_eval_dataset} datapoints from {trainer_args.eval_dataset_path}')
    
    ##################################
    # Create train and eval datasets #
    ##################################
    train_image_paths = [
        {
            'input_images': Path(train_root_path) / f'set_{idx}' / 'input.jpg',
            'target_images': Path(train_root_path) / f'set_{idx}' / 'positive.jpg',
            'negative_images': list((Path(train_root_path) / f'set_{idx}' / 'negatives').iterdir()),
        } for idx in range(len_train_dataset)
    ]
    
    eval_image_paths = [
        {
            'input_images': Path(train_root_path) / f'set_{idx}' / 'input.jpg',
            'target_images': Path(train_root_path) / f'set_{idx}' / 'positive.jpg',
            'negative_images': list((Path(train_root_path) / f'set_{idx}' / 'negatives').iterdir()),
        } for idx in range(len_eval_dataset)
    ]
    
    train_dataset = SNNDataset(image_paths=train_image_paths, transform=transform)
    eval_dataset = SNNDataset(image_paths=eval_image_paths, transform=transform)
    
    ################################
    # Initialize Trainer and Train #
    ################################
    trainer = SNNTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    train_result = trainer.train()
    
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ####################################
    # Save model and create model card #
    ####################################
    logger.info("*** Save model ***")
    trainer.save_model('./output')
    logger.info(f"Model saved to {'./output'}")

    ############
    # Evaluate #
    ############
    if trainer_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("*** Training complete! ***")
    
if __name__ == "__main__":
    main()
    
    
    
    