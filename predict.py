import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define priority values for delta calculation
PRIORITY_MAP = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
REV_PRIORITY_MAP = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}

# Grounding check keyword mappings for feature evidence
CRITICAL_KEYWORDS = [
    'data loss', 'deleted', 'disappeared', 'lost', 'locked', 'invalid credentials', 
    'password reset not working', 'forgotten my password', 'reset option is not working', 
    'not turning on', 'does not respond', 'no response', 'charging properly', 
    'flickering', 'crashes', 'crash', 'strange noises', 'hardware problem', 
    'outage', 'hacked', 'compromise', 'stolen', 'smoke', 'fire', 'original charger'
]

MEDIUM_KEYWORDS = [
    'intermittent', 'firmware', 'software bug', 'software update', 'error message', 
    'freezes', 'disconnects', 'internet connection', 'wi-fi network', 'wi-fi', 
    'battery life', 'cannot connect', 'fail to connect', 'unresolved', 'glitch'
]

LOW_KEYWORDS = [
    'guide me', 'steps', 'how can i', 'find the option', 'desired action', 'product inquiry'
]

def load_classifier(model_dir="models/sia_classifier"):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Classifier model directory not found at {model_dir}. Please run train_pipeline.py first.")
    
    print(f"Loading model and tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

def predict_severity_batch(texts, tokenizer, model, batch_size=32):
    device = torch.device('cpu')
    model = model.to(device)
    
    predictions = []
    confidences = []
    
    # Run predictions in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = [str(t) for t in texts[i:i+batch_size]]
        encodings = tokenizer(
            batch_texts,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            for pred, prob in zip(preds, probs):
                predictions.append(REV_PRIORITY_MAP[pred])
                confidences.append(float(prob[pred]))
                
    return predictions, confidences

def generate_dossier(row, inferred_severity, confidence):
    ticket_id = str(row.get('Ticket ID', ''))
    assigned = str(row.get('Ticket Priority', ''))
    description = str(row.get('Ticket Description', ''))
    channel = str(row.get('Ticket Channel', ''))
    ticket_type = str(row.get('Ticket Type', ''))
    
    # Severity numeric map
    assigned_val = PRIORITY_MAP.get(assigned, 1)
    inferred_val = PRIORITY_MAP.get(inferred_severity, 1)
    
    # Mismatch Type and Delta
    delta = inferred_val - assigned_val
    delta_str = f"+{delta}" if delta > 0 else str(delta)
    mismatch_type = "Hidden Crisis" if delta > 0 else "False Alarm"
    
    # Feature Evidence (Strictly grounded in text keywords)
    feature_evidence = []
    desc_lower = description.lower()
    
    # Extract matched keywords
    matched_crit = [kw for kw in CRITICAL_KEYWORDS if kw in desc_lower]
    matched_med = [kw for kw in MEDIUM_KEYWORDS if kw in desc_lower]
    matched_low = [kw for kw in LOW_KEYWORDS if kw in desc_lower]
    
    if matched_crit:
        feature_evidence.append({
            "signal": "keyword",
            "value": matched_crit[0],
            "weight": "Critical Impact"
        })
    elif matched_med:
        feature_evidence.append({
            "signal": "keyword",
            "value": matched_med[0],
            "weight": "Medium Impact"
        })
    elif matched_low:
        feature_evidence.append({
            "signal": "keyword",
            "value": matched_low[0],
            "weight": "Low Impact"
        })
        
    # Resolution time extraction (if closed)
    res_time_str = "N/A (Open/Pending)"
    interpretation = "N/A"
    if pd.notna(row.get('Time to Resolution')) and pd.notna(row.get('First Response Time')):
        try:
            frt = pd.to_datetime(row['First Response Time'])
            rt = pd.to_datetime(row['Time to Resolution'])
            duration = abs((rt - frt).total_seconds() / 3600)
            res_time_str = f"{duration:.2f} hours"
            
            # Map duration to interpretation
            if duration > 12:
                interpretation = "Extended resolution time indicates high complexity."
            elif duration > 4:
                interpretation = "Standard operational resolution time."
            else:
                interpretation = "Rapid resolution typical of low-impact issues."
        except:
            pass
            
    feature_evidence.append({
        "signal": "resolution_time",
        "value": res_time_str,
        "interpretation": interpretation
    })
    
    # Grounded constraint analysis
    if mismatch_type == "Hidden Crisis":
        explanation = (
            f"The customer submitted a ticket under Category '{ticket_type}' reporting an issue with characteristics of '{inferred_severity}' severity. "
            f"Specifically, text indicators include '{feature_evidence[0]['value']}' (mismatched with the assigned '{assigned}' priority). "
            f"Under-triaging this issue creates a severe SLA violation and customer dissatisfaction risk."
        )
    else:
        explanation = (
            f"The support ticket was logged under Category '{ticket_type}' with an assigned priority of '{assigned}'. "
            f"However, objective semantic analysis indicates a true severity of '{inferred_severity}', as evidenced by search keywords like '{feature_evidence[0]['value'] if feature_evidence else 'general inquiry'}'. "
            f"Over-triaging this case leads to queue inflation and support agent fatigue."
        )
        
    dossier = {
        "ticket_id": ticket_id,
        "assigned_priority": assigned,
        "inferred_severity": inferred_severity,
        "mismatch_type": mismatch_type,
        "severity_delta": delta_str,
        "feature_evidence": feature_evidence,
        "constraint_analysis": explanation,
        "confidence": f"{confidence * 100:.2f}%"
    }
    
    return dossier

def run_inference(csv_path, output_csv_path="predictions.csv", output_json_path="dossiers.json"):
    if not os.path.exists(csv_path):
        print(f"Error: Input CSV file not found at {csv_path}")
        sys.exit(1)
        
    print(f"Reading tickets from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Load model
    tokenizer, model = load_classifier()
    
    # Run prediction on Ticket Description
    descriptions = df['Ticket Description'].fillna("").tolist()
    print("Running sequence classifier predictions...")
    inferred_severities, confidences = predict_severity_batch(descriptions, tokenizer, model)
    
    df['inferred_severity'] = inferred_severities
    df['confidence'] = confidences
    
    # Derive mismatch label
    df['mismatch'] = (df['inferred_severity'] != df['Ticket Priority']).astype(int)
    
    # Save output CSV
    print(f"Saving predictions to {output_csv_path}...")
    df.to_csv(output_csv_path, index=False)
    
    # Generate Evidence Dossiers for all mismatches
    print("Generating Evidence Dossiers for priority mismatches...")
    dossiers = []
    mismatch_df = df[df['mismatch'] == 1]
    
    for idx, row in mismatch_df.iterrows():
        dossier = generate_dossier(row, row['inferred_severity'], row['confidence'])
        dossiers.append(dossier)
        
    # Save output JSON
    print(f"Saving dossiers to {output_json_path}...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dossiers, f, indent=2)
        
    print(f"Inference completed! Processed {len(df)} tickets.")
    print(f"Identified {len(mismatch_df)} priority mismatches ({len(mismatch_df)/len(df)*100:.2f}%).")
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <input_csv_path> [output_csv_path] [output_json_path]")
        sys.exit(1)
        
    input_csv = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "predictions.csv"
    out_json = sys.argv[3] if len(sys.argv) > 3 else "dossiers.json"
    
    run_inference(input_csv, out_csv, out_json)
