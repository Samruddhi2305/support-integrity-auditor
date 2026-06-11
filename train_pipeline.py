import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- STAGE 1: PSEUDO-LABEL GENERATION ---

def get_rule_score(text, ticket_type):
    text = str(text).lower()
    
    # Define keywords representing different levels of urgency
    critical_keywords = [
        'data loss', 'deleted', 'disappeared', 'lost', 'locked', 'invalid credentials', 
        'password reset not working', 'forgotten my password', 'reset option is not working', 
        'not turning on', 'does not respond', 'no response', 'charging properly', 
        'flickering', 'crashes', 'crash', 'strange noises', 'hardware problem', 
        'outage', 'hacked', 'compromise', 'stolen', 'smoke', 'fire', 'original charger'
    ]
    
    medium_keywords = [
        'intermittent', 'firmware', 'software bug', 'software update', 'error message', 
        'freezes', 'disconnects', 'internet connection', 'wi-fi network', 'wi-fi', 
        'battery life', 'cannot connect', 'fail to connect', 'unresolved', 'glitch'
    ]
    
    low_keywords = [
        'guide me', 'steps', 'how can i', 'find the option', 'desired action', 'product inquiry'
    ]
    
    score = 0.5  # Default medium
    
    # Check frequency of keyword matches
    crit_count = sum(1 for kw in critical_keywords if kw in text)
    med_count = sum(1 for kw in medium_keywords if kw in text)
    low_count = sum(1 for kw in low_keywords if kw in text)
    
    if crit_count > 0:
        score = 0.85 + 0.05 * min(crit_count, 3)
    elif med_count > 0:
        score = 0.55 + 0.05 * min(med_count, 3)
    elif low_count > 0:
        score = 0.25 - 0.05 * min(low_count, 3)
        
    # Check punctuation (exclamation points indicate frustration/urgency)
    if '!' in text:
        score += 0.05
        
    return min(max(score, 0.0), 1.0)

def generate_pseudo_labels(df_path):
    print("Loading data for pseudo-labeling...")
    df = pd.read_csv(df_path)
    
    # Signal 1: Rule-Based NLP Urgency Score
    print("Computing Rule-Based NLP Urgency Scores...")
    df['rule_score'] = df.apply(lambda row: get_rule_score(row['Ticket Description'], row['Ticket Type']), axis=1)
    
    # Signal 2: Embedding-Based Clustering Urgency Score
    print("Computing sentence embeddings for descriptions...")
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    descriptions = df['Ticket Description'].fillna("").tolist()
    embeddings = st_model.encode(descriptions, show_progress_bar=True)
    
    print("Clustering descriptions using K-Means...")
    num_clusters = 10
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(embeddings)
    
    # Compute the average rule score of each cluster to assign cluster-level urgency
    cluster_rule_means = df.groupby('cluster')['rule_score'].mean().to_dict()
    df['cluster_score'] = df['cluster'].map(cluster_rule_means)
    
    # Fused Urgency Score (Linear Fusion)
    df['fused_score'] = 0.5 * df['rule_score'] + 0.5 * df['cluster_score']
    
    # Map Fused Score to Inferred Severity classes
    def map_to_severity(score):
        if score < 0.35:
            return 'Low'
        elif score < 0.55:
            return 'Medium'
        elif score < 0.75:
            return 'High'
        else:
            return 'Critical'
            
    df['inferred_severity'] = df['fused_score'].apply(map_to_severity)
    
    # Calculate Binary Mismatch Label
    # Mismatch is 1 if Inferred Severity differs from Ticket Priority, else 0
    df['mismatch'] = (df['inferred_severity'] != df['Ticket Priority']).astype(int)
    
    # Print signal agreement (correlation between individual signals)
    corr = df['rule_score'].corr(df['cluster_score'])
    print(f"Signal Fusion Stats - Correlation between Signal 1 and Signal 2: {corr:.4f}")
    print("Mismatch Class Distribution:")
    print(df['mismatch'].value_counts(normalize=True))
    
    return df

# --- STAGE 2: CLASSIFIER TRAINING ---

class TicketSeverityDataset(Dataset):
    def __init__(self, texts, severity_labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = severity_labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_model():
    # 1. Generate Pseudo-Labels
    df = generate_pseudo_labels("customer_support_tickets.csv")
    df.to_csv("customer_support_tickets_with_labels.csv", index=False)
    
    # Map severity labels to integers for classification
    severity_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
    df['severity_label'] = df['inferred_severity'].map(severity_map)
    
    # We will split on df directly so we retain priority and mismatch labels for evaluation
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['severity_label']
    )
    
    train_texts = train_df['Ticket Description'].fillna("").tolist()
    train_labels = train_df['severity_label'].tolist()
    
    val_texts = val_df['Ticket Description'].fillna("").tolist()
    val_labels = val_df['severity_label'].tolist()
    
    # 2. Tokenizer & Dataset Loader
    model_name = "distilbert-base-uncased"
    print(f"Loading Tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_dataset = TicketSeverityDataset(train_texts, train_labels, tokenizer)
    val_dataset = TicketSeverityDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # 3. Initialize Model for 4 classes
    print(f"Loading Model: {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
    
    device = torch.device('cpu')
    model = model.to(device)
    
    # 4. Address Class Imbalance via Weighted Loss
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # 5. Freeze lower layers of DistilBERT to speed up training on CPU
    print("Freezing lower layers of the transformer to optimize for CPU training...")
    for name, param in model.named_parameters():
        if "transformer.layer.5" not in name and "pre_classifier" not in name and "classifier" not in name:
            param.requires_grad = False
            
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=0.01)
    
    epochs = 2
    
    # 6. Training Loop
    print("Starting fine-tuning...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_tensor = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels_tensor)
            
            total_loss += loss.item()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if (step + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        avg_train_loss = total_loss / len(train_loader)
        print(f"Average Training Loss for Epoch {epoch+1}: {avg_train_loss:.4f}")
        
    # 7. Model Evaluation on Validation Split
    model.eval()
    val_severity_preds = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            val_severity_preds.extend(preds)
            
    # Convert predictions back to labels
    rev_severity_map = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}
    val_df = val_df.copy()
    val_df['predicted_severity'] = [rev_severity_map[p] for p in val_severity_preds]
    
    # Derive binary mismatch predictions
    val_df['predicted_mismatch'] = (val_df['predicted_severity'] != val_df['Ticket Priority']).astype(int)
    
    # Evaluate binary mismatch classification performance
    val_mismatch_true = val_df['mismatch'].values
    val_mismatch_pred = val_df['predicted_mismatch'].values
    
    acc = accuracy_score(val_mismatch_true, val_mismatch_pred)
    f1 = f1_score(val_mismatch_true, val_mismatch_pred, average='macro')
    rec_class_0 = recall_score(val_mismatch_true, val_mismatch_pred, pos_label=0)
    rec_class_1 = recall_score(val_mismatch_true, val_mismatch_pred, pos_label=1)
    
    print("\n--- Validation Performance (Binary Mismatch Detection) ---")
    print(f"Accuracy: {acc:.4f} (Target >= 83%)")
    print(f"Macro F1: {f1:.4f} (Target >= 0.82)")
    print(f"Recall (Consistent): {rec_class_0:.4f} (Target >= 0.78)")
    print(f"Recall (Mismatched): {rec_class_1:.4f} (Target >= 0.78)")
    print("----------------------------------------------------------\n")
    print(classification_report(val_mismatch_true, val_mismatch_pred, target_names=['Consistent', 'Mismatched']))
    
    # 8. Save Fine-Tuned Model
    output_dir = "models/sia_classifier"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Saving fine-tuned model and tokenizer to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved successfully!")

if __name__ == "__main__":
    train_model()
