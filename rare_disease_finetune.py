#!/usr/bin/env python
# coding: utf-8

# # Libraries

import os
import torch
import requests
from PIL import Image

from sklearn.metrics import classification_report
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText, MllamaForConditionalGeneration, TrainingArguments, Trainer, IntervalStrategy, EarlyStoppingCallback
from datasets import load_dataset, Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Meta Llama 3.2 3B Instruct
model_id = "/data/user/bsindala/PhD/CS762-NaturalLanguageProcessing/disease-classification-generation/models/Meta-Llama-3.2-3B-Instruct"

# Load model directly
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

def read_files(data_dir):
    texts = []
    annotations = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), 'r') as f:
                texts.append(f.read())
        elif filename.endswith(".ann"):
            with open(os.path.join(data_dir, filename), 'r') as f:
                annotations.append(f.read())
    return texts, annotations

train_texts, train_annotations = read_files("/data/user/bsindala/PhD/CS762-NaturalLanguageProcessing/disease-classification-generation/dataset/RareDis-v1/train")
dev_texts, dev_annotations = read_files("/data/user/bsindala/PhD/CS762-NaturalLanguageProcessing/disease-classification-generation/dataset/RareDis-v1/dev")
test_texts, test_annotations = read_files("/data/user/bsindala/PhD/CS762-NaturalLanguageProcessing/disease-classification-generation/dataset/RareDis-v1/test")

# Debugging: Print the contents of train_texts and train_annotations
print("Train Texts:", train_texts)
print("Train Annotations:", train_annotations)

# Debugging: Print the contents of train_texts and train_annotations
print("Dev Texts:", dev_texts)
print("Dev Annotations:", dev_annotations)

# Debugging: Print the contents of train_texts and train_annotations
print("Test Texts:", test_texts)
print("Test Annotations:", test_annotations)

# PreProcess
def preprocessing(texts, annotations):
    processed_data = []
    for text, ann in zip(texts, annotations):
        processed_data.append({"text": text, "annotations": ann})
    return processed_data

train_data = preprocessing(train_texts, train_annotations)
dev_data = preprocessing(dev_texts, dev_annotations)
test_data = preprocessing(test_texts, test_annotations)

print("Processed Training Data:", train_data)
print("Processed Dev Data:", dev_data)
print("Processed Test Data:", test_data)

# Convert to dictionary
def dictionary_converter(data):
    dict_data = {"text": [], "annotations": []}
    for item in data:
        dict_data["text"].append(item["text"])
        dict_data["annotations"].append(item["annotations"])
    return dict_data

# Convert the processed data to the required format
train_dictionary = dictionary_converter(train_data)
dev_dictionary = dictionary_converter(dev_data)
test_dictionary = dictionary_converter(test_data)

# Dataset Convertion
train_dataset = Dataset.from_dict(train_dictionary)
dev_dataset = Dataset.from_dict(dev_dictionary)
test_dataset = Dataset.from_dict(test_dictionary)

print(train_dataset)
print(dev_dataset)
print(test_dataset)

# Tokenization
# Use the End-of-Sequence Token as the Padding Token
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_dev = dev_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Fine Tuning the Model
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy=IntervalStrategy.EPOCH,
    save_strategy=IntervalStrategy.EPOCH,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    logging_steps=500,
    dataloader_num_workers=1,
    dataloader_pin_memory=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train
trainer.train()

# Evaluate
eval_results = trainer.evaluate()
print(eval_results)

# Test
test_results = trainer.predict(tokenized_test)
print(test_results)

# Validation
predictions = test_results.predictions.argmax(-1)
labels = test_results.label_ids
report = classification_report(labels, predictions)
print(report)

# Save the fine-tuned model and tokenizer
model.save_pretrained("/data/user/bsindala/PhD/CS762-NaturalLanguageProcessing/disease-classification-generation/fine_tuned_model")
tokenizer.save_pretrained("/data/user/bsindala/PhD/CS762-NaturalLanguageProcessing/disease-classification-generation/fine_tuned_model")