import argparse
import json
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig
from transformers import DataCollatorWithPadding
from datasets import Dataset
from transformers import TrainerCallback
from sklearn.model_selection import train_test_split
import random


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
    # Step 1: Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 2: Load pre-trained model and tokenizer using Auto classes
    model_name = args.model_name

    if args.dropout_rate != 0:
        # Load model configuration
        config = AutoConfig.from_pretrained(model_name, num_labels=1)
        config.hidden_dropout_prob = args.dropout_rate  # Applies to the hidden layers
        config.attention_probs_dropout_prob = args.dropout_rate  # Applies to the attention layers

        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)  # Move model to the GPU if available

    # Step 3: Prepare your dataset (replace with your actual dataset)
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
            if isinstance(item["text"], list):
                item["text"] = item["text"][0]
            dataset["text"].append(item["text"])
            dataset["label"].append(item[target]*args.scale)
        return dataset

    data_train = prepare_dataset(args, data_train)
    data_test = prepare_dataset(args, data_test)

    # Shuffle datasets for better training
    data_train = Dataset.from_dict(data_train).shuffle(seed=42)
    data_test = Dataset.from_dict(data_test).shuffle(seed=42)

    # Step 4: Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=args.max_len)

    tokenized_datasets_train = data_train.map(tokenize_function, batched=True)
    tokenized_datasets_test = data_test.map(tokenize_function, batched=True)

    # Step 5: Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Calculate total training steps
    total_steps = len(tokenized_datasets_train) // args.batch_size * args.num_epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    # Step 6: Define the training arguments with recommended parameters
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy=args.evaluation_strategy,  # Frequent evaluation
        save_strategy=args.evaluation_strategy,  # Ensure save strategy matches evaluation strategy
        learning_rate=args.learning_rate,  # Lower learning rate
        per_device_train_batch_size=args.batch_size,  # Smaller batch size
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,  # More epochs to learn effectively from small data
        weight_decay=args.weight_decay,  # Regularization to prevent overfitting
        logging_dir=args.log_dir,  # Use a single log directory
        logging_steps=args.logging_steps,
        fp16=torch.cuda.is_available(),  # Use mixed precision if a GPU is available
        eval_steps=args.eval_steps if args.evaluation_strategy == "steps" else None,  # Frequent evaluations
        save_steps=args.eval_steps if args.evaluation_strategy == "steps" else None,  # Ensure save steps match eval steps if using 'steps'
        load_best_model_at_end=True,  # Load the best model at the end of training
        report_to=["none"],  # Disable reporting to any tracker like WandB or Tensorboard
        warmup_steps=warmup_steps,  # Adjustable warmup steps based on the warmup ratio
        gradient_accumulation_steps=2,  # Gradient accumulation to simulate larger batch size
        save_total_limit=1  # Limit the number of checkpoints to keep
    )

    # Step 7: Initialize the Trainer
    loss_logger = LossLogger()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets_train,
        eval_dataset=tokenized_datasets_test,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[loss_logger],  # Add custom callback
    )

    # Step 8: Train the model
    trainer.train()

    # Step 9: Save the model
    trainer.save_model(args.output_dir)

    # # Step 10: Evaluate the model
    # eval_results = trainer.evaluate()
    # print(f"Evaluation results: {eval_results}")

    # Step 11: Plot and Save Loss Curves
    plot_loss_curves(loss_logger, args.output_dir, args.learning_target)

def plot_loss_curves(loss_logger, output_dir, learning_target):
    # Extract loss values and steps from the logger
    train_loss = loss_logger.train_loss
    eval_loss = loss_logger.eval_loss
    train_steps = loss_logger.train_steps
    eval_steps = loss_logger.eval_steps

    # Find minimum loss values and their corresponding steps
    min_train_loss = min(train_loss)
    min_train_step = train_steps[train_loss.index(min_train_loss)]

    min_eval_loss = min(eval_loss) if eval_loss else None
    min_eval_step = eval_steps[eval_loss.index(min_eval_loss)] if eval_loss else None

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(train_steps, train_loss, label='Training Loss')
    if eval_loss:
        plt.plot(eval_steps, eval_loss, label='Validation Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)

    if learning_target == 'Difficulty':
        plt.ylim(0, 0.3)

    # Annotate minimum loss values
    plt.scatter(min_train_step, min_train_loss, color='blue')
    plt.text(min_train_step, min_train_loss, f'Min Train Loss: {min_train_loss:.4f}', fontsize=9, verticalalignment='bottom')

    if min_eval_loss is not None:
        plt.scatter(min_eval_step, min_eval_loss, color='orange')
        plt.text(min_eval_step, min_eval_loss, f'Min Eval Loss: {min_eval_loss:.4f}', fontsize=9, verticalalignment='bottom')

    # Save plot to the output directory
    plot_path = f"{output_dir}/loss_curves.png"
    plt.savefig(plot_path)
    print(f"Loss curves saved to {plot_path}")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Train a model for string to value prediction.")

    parser.add_argument('--train_data_path', type=str, default='combined_bin_train_bin_dict_Difficulty_aug_target_5_simple_training.json', help="Path to the training data JSON file.")
    parser.add_argument('--test_data_path', type=str, default='combined_bin_validation_set_Difficulty.json', help="Path to the testing data JSON file.")
    parser.add_argument('--split_ratio', type=float, default=0.2, help="Proportion of the training data to use for validation if train and test data are the same.")
    parser.add_argument('--learning_target', type=str, default='Difficulty', help="Target variable for learning (e.g., Difficulty, Response_Time).")
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help="roberta-base, bert-base-uncased")
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="Learning rate for training.")  # Lower learning rate
    parser.add_argument('--dropout_rate', type=float, default=0, help="")  # Lower learning rate
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay for optimizer.")  # Regularization
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="Proportion of total training steps to use for warmup.")  # Adjustable warmup ratio
    parser.add_argument('--scale', type=float, default=1, help="scale the target") 
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training.")  # Smaller batch size
    parser.add_argument('--num_epochs', type=int, default=15, help="Number of epochs for training.")
    parser.add_argument('--max_len', type=int, default=512, help="Max Len")
    parser.add_argument('--evaluation_strategy', type=str, choices=['no', 'steps', 'epoch'], default='steps', help="Evaluation strategy to use during training.")
    parser.add_argument('--eval_steps', type=int, default=10, help="Number of update steps between two evaluations if evaluation_strategy is 'steps'.")  # Frequent evaluations
    parser.add_argument('--output_dir', type=str, default="./results", help="Directory to save the model output.")
    parser.add_argument('--log_dir', type=str, default="./logs", help="Directory to save the logs.")  # Use a single log directory
    parser.add_argument('--logging_steps', type=int, default=10, help="Logging steps for training.")
    
    args = parser.parse_args()
    main(args)
