import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from sklearn.metrics import root_mean_squared_error
import numpy as np


def main(args):
    # Step 1: Load pre-trained model and tokenizer
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, num_labels=1)
    
    # Ensure model is in evaluation mode
    model.eval()

    # Step 2: Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Step 3: Load test data
    with open(args.test_data_path) as f:
        test_data = json.load(f)
    
    def prepare_dataset(data):
        dataset = {
            "text": [],
            "label": []
        }
        for item in data:
            dataset["text"].append(item["text"])
            dataset["label"].append(item[args.learning_target])
        return dataset

    data_test = prepare_dataset(test_data)
    tokenized_datasets_test = Dataset.from_dict(data_test)

    # Step 4: Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=args.max_len)

    tokenized_datasets_test = tokenized_datasets_test.map(tokenize_function, batched=True)

    # Step 5: Predict and calculate distances
    results = []
    all_labels = []
    all_predictions = []

    for i in range(len(tokenized_datasets_test)):
        input_ids = torch.tensor(tokenized_datasets_test[i]['input_ids']).unsqueeze(0).to(device)
        attention_mask = torch.tensor(tokenized_datasets_test[i]['attention_mask']).unsqueeze(0).to(device)
        label = data_test['label'][i]
        
        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)
            prediction = output.logits.squeeze().item() / args.scale
        
        distance = abs(prediction - label)
        results.append({
            'text': data_test['text'][i],
            'true_value': label,
            'predicted_value': prediction,
            'distance': distance
        })

        all_labels.append(label)
        all_predictions.append(prediction)

    # Calculate Root Mean Squared Error for additional insights
    mse = root_mean_squared_error(all_labels, all_predictions)
    print(f"Root Mean Squared Error (RMSE): {mse}")
    
    results = ['Root Mean Squared Error (RMSE): '+str(mse)] + results

    # Step 6: Save results to JSON file
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a model for string to value prediction and save the results.")

    parser.add_argument('--test_data_path', type=str, default='test_final_type1.json', help="Path to the testing data JSON file.")
    parser.add_argument('--learning_target', type=str, default='Difficulty', help="Target variable for learning (e.g., Difficulty, Response_Time).")
    parser.add_argument('--model_dir', type=str, default='results/Difficulty_type1_bert_5e6_warm3_decay01_bs16_epo20', help="Directory of the saved model.")
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help="Name of the pre-trained model to use (e.g., roberta-base, bert-base-uncased).")
    parser.add_argument('--output_file', type=str, default='prediction_results.json', help="File path to save the results JSON file.")
    parser.add_argument('--scale', type=float, default=1, help="scale the target") 
    parser.add_argument('--max_len', type=int, default=512, help="Max Len")

    args = parser.parse_args()
    main(args)
