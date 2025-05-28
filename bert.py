import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.metrics import accuracy_score
import sys
import os

def main(input_dir, output_dir):
    data = pd.read_csv(os.path.join(input_dir, 'fake_reviews_dataset.csv'))

    def encode_label(df):
        labels = {
            'CG': 0,
            'OR': 1
        }
        
        df['target'] = df['label'].map(labels)
        
        return df


    data = encode_label(data)
    data = data.drop(columns=['category', 'label', 'rating'])

    def preprocess_text(text):
        stop_words = set(stopwords.words('english'))
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        return tokens

    data['text'] = data['text_'].apply(preprocess_text)
    data['text'] = data['text'].apply(lambda x: ' '.join(x))
    data['label'] = data['target']
    data = data.drop(columns=['text_', 'target'])

    train_data = data.sample(frac=0.7, random_state=42)
    val_data = data.drop(train_data.index).sample(frac=0.5, random_state=42)
    test_data = data.drop(train_data.index).drop(val_data.index)
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    print(train_data.shape)
    print(val_data.shape)
    print(test_data.shape)

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=512)

    train_data = train_data.sample(frac=1, random_state=42)

    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)
    test_dataset = Dataset.from_pandas(test_data)

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    columns_to_return = ['input_ids', 'attention_mask', 'label']
    train_dataset.set_format(type='torch', columns=columns_to_return)
    val_dataset.set_format(type='torch', columns=columns_to_return)
    test_dataset.set_format(type='torch', columns=columns_to_return)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {'accuracy': accuracy_score(labels, predictions)}

    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=2)

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, 'results_fake_reviews'),
        eval_strategy="epoch",           # Evaluate after each epoch
        save_strategy="epoch",                 # Save checkpoint after each epoch
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, 'logs_fake_reviews'),
        logging_steps=100,
        save_total_limit=1,                    # Keep only the best model checkpoint
        load_best_model_at_end=True,           # Load best model after training
        metric_for_best_model="eval_accuracy", # Use eval_accuracy to determine the best model
        greater_is_better=True                 # Higher accuracy is better
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate on the test set using the best model
    test_results = trainer.evaluate(test_dataset)
    print("Test Accuracy: ", test_results['eval_accuracy'])
    # Save the model
    model.save_pretrained(os.path.join(output_dir, 'fake_review_detection_model'))
    tokenizer.save_pretrained(os.path.join(output_dir, 'fake_review_detection_model'))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_data.py <input_dir> <output_dir>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    main(input_dir, output_dir)