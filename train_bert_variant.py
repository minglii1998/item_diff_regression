import argparse
import json
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig
from transformers import DataCollatorWithPadding
from datasets import Dataset
from transformers import TrainerCallback
import random

class MyModelForSequenceClassification(AutoModelForSequenceClassification):
    """
    Custom model class that ensures tensors are contiguous before saving.
    """
    def save_pretrained(self, save_directory, save_config=True):
        # Ensure all parameters are contiguous before saving
        for name, param in self.named_parameters():
            if not param.data.is_contiguous():
                param.data = param.data.contiguous()
        super().save_pretrained(save_directory, save_config=save_config)

class CustomTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        """
        Save the model after making all parameters contiguous.
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        self.model.eval()
        
        # Make sure all parameters are contiguous
        for name, param in self.model.named_parameters():
            if not param.data.is_contiguous():
                param.data = param.data.contiguous()
        
        # Proceed with the usual save
        self.model.save_pretrained(output_dir)
        
        # Save the tokenizer too if necessary
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        
        self.model.train()

class LossLogger(TrainerCallback):
    """
    A custom callback to log the training and evaluation losses.
    """
    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.eval_loss = []
        self.train_steps = []
        self.eval_steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if 'loss' in logs:
            self.train_loss.append(logs['loss'])
            self.train_steps.append(state.global_step)  # Record the global step for training loss
        if 'eval_loss' in logs:
            print(f"Logging eval_loss: {logs['eval_loss']} at step {state.global_step}")  # Debug print to ensure eval_loss is being logged
            self.eval_loss.append(logs['eval_loss'])
            self.eval_steps.append(state.global_step)  # Record the global step for evaluation loss

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = args.model_name
    if args.dropout_rate != 0:
        config = AutoConfig.from_pretrained(model_name, num_labels=1)
        config.hidden_dropout_prob = args.dropout_rate
        config.attention_probs_dropout_prob = args.dropout_rate
        model = MyModelForSequenceClassification.from_pretrained(model_name, config=config)
    else:
        model = MyModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)

    target = args.learning_target
    if args.train_data_path != args.test_data_path:
        with open(args.train_data_path) as f:
            data_train = json.load(f)
        with open(args.test_data_path) as f:
            data_test = json.load(f)
    else:
        with open(args.train_data_path) as f:
            data_all = json.load(f)
        data_test = random.sample(data_all, int(len(data_all)*args.split_ratio))
        data_train = [item for item in data_all if item not in data_test]

    def prepare_dataset(args, data):
        dataset = {
            "text": [],
            "label": []
        }
        for item in data:
            dataset["text"].append(item["text"])
            dataset["label"].append(item[target]*args.scale)
        return dataset

    data_train = prepare_dataset(args, data_train)
    data_test = prepare_dataset(args, data_test)

    data_train = Dataset.from_dict(data_train).shuffle(seed=42)
    data_test = Dataset.from_dict(data_test).shuffle(seed=42)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

    tokenized_datasets_train = data_train.map(tokenize_function, batched=True)
    tokenized_datasets_test = data_test.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    total_steps = len(tokenized_datasets_train) // args.batch_size * args.num_epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.evaluation_strategy,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        logging_dir=args.log_dir,
        logging_steps=args.logging_steps,
        fp16=torch.cuda.is_available(),
        eval_steps=args.eval_steps if args.evaluation_strategy == "steps" else None,
        save_steps=args.eval_steps if args.evaluation_strategy == "steps" else None,
        load_best_model_at_end=True,
        report_to=["none"],
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=2,
        save_total_limit=1
    )

    loss_logger = LossLogger()
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets_train,
        eval_dataset=tokenized_datasets_test,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[loss_logger]
    )

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for string to value prediction.")
    parser.add_argument('--train_data_path', type=str, default='train_final_type1.json', help="Path to the training data JSON file.")
    parser.add_argument('--test_data_path', type=str, default='test_final_type1.json', help="Path to the testing data JSON file.")
    parser.add_argument('--split_ratio', type=float, default=0.2, help="Proportion of the training data to use for validation if train and test data are the same.")
    parser.add_argument('--learning_target', type=str, default='Difficulty', help="Target variable for learning (e.g., Difficulty, Response_Time).")
    parser.add_argument('--model_name', type=str, default='roberta-base', help="roberta-base, bert-base-uncased")
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="Learning rate for training.")
    parser.add_argument('--dropout_rate', type=float, default=0, help="Dropout rate for the model.")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay for optimizer.")
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="Proportion of total training steps to use for warmup.")
    parser.add_argument('--scale', type=float, default=1, help="Scale the target.")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training.")
    parser.add_argument('--num_epochs', type=int, default=15, help="Number of epochs for training.")
    parser.add_argument('--evaluation_strategy', type=str, choices=['no', 'steps', 'epoch'], default='steps', help="Evaluation strategy to use during training.")
    parser.add_argument('--eval_steps', type=int, default=10, help="Number of update steps between two evaluations if evaluation_strategy is 'steps'.")
    parser.add_argument('--output_dir', type=str, default="./results", help="Directory to save the model output.")
    parser.add_argument('--log_dir', type=str, default="./logs", help="Directory to save the logs.")
    parser.add_argument('--logging_steps', type=int, default=10, help="Logging steps for training.")

    args = parser.parse_args()
    main(args)
