import argparse
import json
import torch
from transformers import T5Tokenizer, T5EncoderModel, AutoConfig
from datasets import Dataset
from sklearn.metrics import mean_squared_error
import numpy as np
from torch import nn
import os


import torch.nn as nn

class T5EncoderClassifier(nn.Module):
    def __init__(self, encoder_model, hidden_layers=[512], dropout_rate=0.1):
        super(T5EncoderClassifier, self).__init__()
        self.t5_encoder = encoder_model
        self.config = self.t5_encoder.config  # Store a reference to the config
        
        # Freeze the parameters of the T5 encoder
        for param in self.t5_encoder.parameters():
            param.requires_grad = False
        
        # Build the regressor with additional hidden layers
        layers = []
        input_size = self.t5_encoder.config.hidden_size
        for layer_size in hidden_layers:
            layers.append(nn.Linear(input_size, layer_size))
            layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = layer_size
        layers.append(nn.Linear(input_size, 1))  # Final layer to output the prediction
        self.regressor = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.t5_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        predictions = self.regressor(pooled_output).squeeze(-1)
        
        loss = None
        if labels is not None:
            loss = nn.MSELoss()(predictions, labels.float())
        return (loss, predictions) if loss is not None else predictions



def main(args):
    # Step 1: Load pre-trained model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.model_dir)

    config = AutoConfig.from_pretrained(args.model_dir)
    base_model = T5EncoderModel(config)
    model = T5EncoderClassifier(base_model)

    # Ensure model is in evaluation mode
    model.eval()

    # Step 2: Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    model.load_state_dict(torch.load(os.path.join(args.model_dir,'pytorch_model.bin'), map_location=device))

    # Step 3: Load test data
    with open(args.test_data_path) as f:
        test_data = json.load(f)
    
    def prepare_dataset(data):
        return tokenizer(data["text"], truncation=True, padding=True, return_tensors="pt", max_length=1024)

    # Step 5: Predict and calculate distances
    results = []
    all_labels = []
    all_predictions = []

    for item in test_data:
        inputs = prepare_dataset(item)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the correct device
        label = float(item[args.learning_target])  # Ensure labels are floats
        
        with torch.no_grad():
            prediction = model(**inputs).item() / args.scale
        
        distance = abs(prediction - label)
        results.append({
            'text': item["text"],
            'true_value': label,
            'predicted_value': prediction,
            'distance': distance
        })

        all_labels.append(label)
        all_predictions.append(prediction)

    # Calculate Root Mean Squared Error for additional insights
    mse = mean_squared_error(all_labels, all_predictions, squared=False)
    print(f"Root Mean Squared Error (RMSE): {mse}")
    
    results = [{'Root Mean Squared Error (RMSE)': mse}] + results

    # Step 6: Save results to JSON file
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a model for string to value prediction and save the results.")
    parser.add_argument('--test_data_path', type=str, required=True, help="Path to the testing data JSON file.")
    parser.add_argument('--learning_target', type=str, required=True, help="Target variable for learning (e.g., Difficulty, Response_Time).")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory of the saved model.")
    parser.add_argument('--model_name', type=str, default='t5-base', help="Name of the pre-trained model to use.")
    parser.add_argument('--output_file', type=str, default='prediction_results.json', help="File path to save the results JSON file.")
    parser.add_argument('--scale', type=float, default=1, help="Scale the target value.") 
    args = parser.parse_args()
    main(args)
