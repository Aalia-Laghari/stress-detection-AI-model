import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import random
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import evaluate

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

stress_lysis = pd.read_csv("Stress-Lysis.csv")

dreaddit_ds = load_dataset("andreagasparini/dreaddit")
goemotions_ds = load_dataset("SetFit/go_emotions")

dreaddit_df = dreaddit_ds["train"].to_pandas()
goemotions_df = goemotions_ds["train"].to_pandas()

print("Stress-Lysis shape:", stress_lysis.shape)
print(stress_lysis.head())
print("Missing values:", stress_lysis.isnull().sum().sum())
print("Duplicates:", stress_lysis.duplicated().sum())
print("Unique Stress Levels:", stress_lysis["Stress_Level"].unique())
print("\nFeature Skewness:")
print(stress_lysis.skew(numeric_only=True))

plt.figure(figsize=(4, 4))
stress_lysis.boxplot()
plt.title("Feature Outliers")
plt.tight_layout()
plt.show()

sns.histplot(data=stress_lysis, x="Stress_Level", bins=10)
plt.title("Stress Level Distribution")
plt.tight_layout()
plt.show()

correlation = stress_lysis.corr(numeric_only=True)
sns.heatmap(correlation, annot=True, cmap="crest", linewidths=0.2)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

sns.scatterplot(x="Humidity", y="Temperature", hue="Stress_Level", data=stress_lysis)
plt.title("Humidity vs Temperature vs Stress")
plt.tight_layout()
plt.show()

q1, q3 = stress_lysis["Step_count"].quantile([0.25, 0.75])
stress_lysis["Step_count"] = np.clip(stress_lysis["Step_count"], q1, q3)

X = stress_lysis.drop(["Stress_Level"], axis=1)
y = stress_lysis["Stress_Level"]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_s)
X_test_s = scaler.transform(X_test_s)

lr = LogisticRegression(max_iter=300, random_state=SEED)
lr.fit(X_train_s, y_train_s)
print("Logistic Regression Accuracy:", accuracy_score(y_test_s, lr.predict(X_test_s)))

rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=SEED)
rf.fit(X_train_s, y_train_s)
print("Random Forest Accuracy:", accuracy_score(y_test_s, rf.predict(X_test_s)))

svm = SVC(kernel="linear", random_state=SEED)
svm.fit(X_train_s, y_train_s)
print("SVM Accuracy:", accuracy_score(y_test_s, svm.predict(X_test_s)))

pickle.dump(svm, open("stress_trained.sav", "wb"))
print("SVM model saved as stress_trained.sav")

stress_lysis["text"] = stress_lysis.apply(
    lambda row: f"Humidity: {row['Humidity']}, Temp: {row['Temperature']}, Steps: {row['Step_count']}",
    axis=1,
)
stress_lysis = stress_lysis.rename(columns={"Stress_Level": "stress"})
stress_lysis = stress_lysis[["text", "stress"]]

negative_emotions = ["anger", "annoyance", "fear", "nervousness", "sadness", "disgust", "remorse"]
goemotions_df["stress"] = goemotions_df[negative_emotions].sum(axis=1)
goemotions_df["stress"] = goemotions_df["stress"].apply(
    lambda x: 2 if x >= 2 else (1 if x == 1 else 0)
)
goemotions_df = goemotions_df[["text", "stress"]]

dreaddit_df["stress"] = dreaddit_df["label"].apply(lambda x: 2 if x == 1 else 0)
dreaddit_df = dreaddit_df[["text", "stress"]]

combined_df = pd.concat([stress_lysis, goemotions_df, dreaddit_df], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
combined_df.to_csv("combined_dataset.csv", index=False)
print("Combined dataset shape:", combined_df.shape)
print(combined_df["stress"].value_counts())

label_encoder = LabelEncoder()
combined_df["stress"] = label_encoder.fit_transform(combined_df["stress"])

train_texts, test_texts, train_labels, test_labels = train_test_split(
    combined_df["text"].tolist(),
    combined_df["stress"].tolist(),
    test_size=0.2,
    random_state=SEED,
    stratify=combined_df["stress"],
)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)


class StressDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


train_dataset = StressDataset(train_encodings, train_labels)
test_dataset = StressDataset(test_encodings, test_labels)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3
)

accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="./stress_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
    weight_decay=0.01,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

model.save_pretrained("./final_stress_model")
tokenizer.save_pretrained("./final_stress_tokenizer")
print("DistilBERT model saved to ./final_stress_model")
