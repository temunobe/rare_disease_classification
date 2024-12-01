# Install Libraries
import pandas as pd
import torch
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from config import config

# Prepare the data for training
project_path = config['project_path']
file_path = os.path.join(project_path, 'dataset', 'symbipredict_2022.csv')
data = pd.read_csv(file_path)

data.head()

# Load the model
model_path = os.path.join(project_path, 'models', 'OpenBioLLM-Llama3-70B')

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Convert all desired to numerical labels
# label_encoder = LabelEncoder()
# data['prognosis'] = label_encoder.fit_transform(data['prognosis'])

# train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
# train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

# print(train_data.head())

# Tokenization
def tokenization(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_data = data.map(tokenization, batched=True)
    
# tokenized_train_data = train_data.apply(tokenization, axis=1)
# tokenized_test_data = test_data.apply(tokenization, axis=1)
# tokenized_val_data = val_data.apply(tokenization, axis=1)

# Training
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
)

# Train the model
trainer.train()

# Evaluating and Testing the Model
eval_results = trainer.evaluate()
print(f'Validation results: {eval_results}')

# test_results = trainer.predict(tokenized_test_data)
# print(f'Test results: {test_results.metrics}')

# Validation
# scores = cross_val_score(model, tokenized_train_data, cv=5)
# print(f'Cross Validation scores: {scores}')

# Deploy
model.save_pretrained('./fine_tuned_OpenBioLLM-Llama3-70B')
tokenizer.save_pretrained('./fine_tuned_OpenBioLLM-Llama3-70B')

print("Model fine-tuning and evaluation complete. The fine-tuned model is saved in './fine_tuned_OpenBioLLM_Llama3_70B'.")
