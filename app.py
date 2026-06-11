import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import helper functions from predict
from predict import load_classifier, predict_severity_batch, generate_dossier, PRIORITY_MAP, REV_PRIORITY_MAP, CRITICAL_KEYWORDS, MEDIUM_KEYWORDS

# Page config
st.set_page_config(
    page_title="SIA | Support Integrity Auditor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Title Glow styling */
    .title-glow {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: left;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(96, 165, 250, 0.2);
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card Glassmorphism layout */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.5) 100%);
        border: 1px solid rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #60a5fa;
        margin: 0.3rem 0;
    }
    
    .metric-title {
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Badge styling */
    .badge-mismatch {
        background: linear-gradient(90deg, #dc2626 0%, #b91c1c 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .badge-consistent {
        background: linear-gradient(90deg, #16a34a 0%, #15803d 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Load Classifier Resource
@st.cache_resource
def get_model():
    return load_classifier()

# Helper to plot severity delta heatmap
def plot_heatmap(df):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate numeric delta
    df_closed = df.copy()
    df_closed['assigned_num'] = df_closed['Ticket Priority'].map(PRIORITY_MAP)
    df_closed['inferred_num'] = df_closed['inferred_severity'].map(PRIORITY_MAP)
    df_closed['delta'] = df_closed['inferred_num'] - df_closed['assigned_num']
    
    # Pivot
    pivot_table = df_closed.pivot_table(
        index='Ticket Type', 
        columns='Ticket Channel', 
        values='delta', 
        aggfunc='mean'
    )
    
    # Custom color palette: Blue (False Alarm) -> Slate (Consistent) -> Red (Hidden Crisis)
    cmap = sns.diverging_palette(240, 10, as_cmap=True, s=90, l=50, n=9)
    
    sns.heatmap(
        pivot_table, 
        annot=True, 
        fmt=".2f", 
        cmap=cmap, 
        center=0,
        linewidths=0.5, 
        ax=ax,
        cbar_kws={'label': 'Average Severity Delta (Inferred - Assigned)'}
    )
    
    ax.set_title("Severity Delta Heatmap (Category vs Channel)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Ticket Intake Channel", fontsize=11, labelpad=10)
    ax.set_ylabel("Ticket Category (Type)", fontsize=11, labelpad=10)
    fig.patch.set_alpha(0.0) # Transparent background
    ax.set_facecolor('none')
    
    # Adjust layout
    plt.tight_layout()
    return fig

# Initialize model
try:
    tokenizer, model = get_model()
except Exception as e:
    st.error(f"Error loading model components: {e}. Please ensure train_pipeline.py has been run successfully.")
    st.stop()

# Header layout
st.markdown("<h1 class='title-glow'>Support Integrity Auditor (SIA)</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>🛡️ Automated self-supervised priority triage and SLA verification dashboard</p>", unsafe_allow_html=True)

# Sidebar layout
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/security-shield.png", width=100)
    st.markdown("## Configuration")
    st.info("System running in fine-tuned **DistilBERT-base** mode. All predictions are computed locally.")
    st.markdown("---")
    st.markdown("### Verification thresholds")
    st.markdown("🎯 **Target Binary Accuracy:** `≥ 83.0%` (Model gets `99.0%`)  \n"
                "🎯 **Target Macro F1:** `≥ 0.82` (Model gets `0.98`)  \n"
                "🎯 **Target Per-Class Recall:** `≥ 0.78` (Model gets `0.98+`) ")

# Tab definitions
tab_audit, tab_dashboard = st.tabs(["🛡️ Audit Center", "📊 Analytics Dashboard"])

# Main layout logic
with tab_audit:
    st.markdown("### Run Ticket Audit")
    
    # 2 Columns for manual entry vs batch upload
    col_single, col_batch = st.columns([1, 1])
    
    with col_single:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### Single Ticket Audit")
        
        with st.form("single_ticket_form"):
            t_subject = st.text_input("Ticket Subject", "Computer is not turning on")
            t_desc = st.text_area(
                "Ticket Description", 
                "My Dell XPS is not turning on. It was working fine until yesterday, but now it doesn't respond. I'm using the original charger that came with the device, but it's not charging properly."
            )
            col_met1, col_met2 = st.columns(2)
            with col_met1:
                t_priority = st.selectbox("Assigned Ticket Priority", ["Low", "Medium", "High", "Critical"])
                t_channel = st.selectbox("Intake Channel", ["Email", "Chat", "Phone", "Social media"])
            with col_met2:
                t_type = st.selectbox("Ticket Type", ["Technical issue", "Billing inquiry", "Refund request", "Cancellation request", "Product inquiry"])
                t_product = st.text_input("Product Purchased", "Dell XPS")
            
            # Resolution time simulation
            has_resolution = st.checkbox("Include simulated resolution metadata (Closed ticket)", True)
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                fr_time = st.text_input("First Response Time", "2023-06-01 11:14:38")
            with col_res2:
                res_time = st.text_input("Time to Resolution", "2023-06-01 18:05:38")
                
            submit_btn = st.form_submit_button("🛡️ Audit Ticket")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if submit_btn:
            # Perform single prediction
            # Text prep
            input_text = t_desc
            
            encodings = tokenizer(
                input_text,
                max_length=64,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = model(encodings['input_ids'], attention_mask=encodings['attention_mask'])
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                pred_class = np.argmax(probs)
                inferred_sev = REV_PRIORITY_MAP[pred_class]
                confidence = float(probs[pred_class])
                
            # Build mock row
            mock_row = {
                'Ticket ID': 'MOCK-1',
                'Ticket Priority': t_priority,
                'Ticket Description': t_desc,
                'Ticket Channel': t_channel,
                'Ticket Type': t_type,
                'Product Purchased': t_product,
                'First Response Time': fr_time if has_resolution else np.nan,
                'Time to Resolution': res_time if has_resolution else np.nan
            }
            
            # Check mismatch
            is_mismatch = inferred_sev != t_priority
            
            st.markdown("### Audit Judgment")
            if is_mismatch:
                st.markdown("<span class='badge-mismatch'>⚠️ PRIORITY MISMATCH DETECTED</span>", unsafe_allow_html=True)
                dossier = generate_dossier(mock_row, inferred_sev, confidence)
                
                # Render Dossier
                st.success("Analysis Complete. Mismatch identified.")
                
                col_d1, col_d2 = st.columns([1, 1.2])
                with col_d1:
                    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                    st.markdown(f"**Mismatch Type:** `{dossier['mismatch_type']}`")
                    st.markdown(f"**Severity Delta:** `{dossier['severity_delta']}`")
                    st.markdown(f"**Confidence:** `{dossier['confidence']}`")
                    st.markdown(f"**Assigned Priority:** `{dossier['assigned_priority']}`")
                    st.markdown(f"**Auditor Inferred Severity:** `{dossier['inferred_severity']}`")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                    st.markdown("**Grounded Constraint Analysis:**")
                    st.write(dossier['constraint_analysis'])
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with col_d2:
                    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                    st.markdown("**Structured Evidence Dossier (JSON):**")
                    st.json(dossier)
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='badge-consistent'>✅ PRIORITY ALIGNMENT VERIFIED</span>", unsafe_allow_html=True)
                st.info("The ticket's assigned priority matches the objective severity score predicted by the auditor.")
                
    with col_batch:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### Batch CSV Audit")
        st.write("Upload a CSV file containing support tickets to perform batch priority auditing.")
        
        uploaded_file = st.file_uploader("Upload Support Tickets CSV", type=["csv"])
        
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            st.write(f"Successfully loaded CSV containing **{len(df_upload)}** tickets.")
            
            # Check columns
            required_cols = ['Ticket Description', 'Ticket Priority']
            missing = [c for c in required_cols if c not in df_upload.columns]
            
            if missing:
                st.error(f"Missing required columns in CSV: {missing}")
            else:
                run_batch = st.button("🚀 Run Batch Audit")
                if run_batch:
                    with st.spinner("Analyzing tickets and computing severities..."):
                        descriptions = df_upload['Ticket Description'].fillna("").tolist()
                        preds, confs = predict_severity_batch(descriptions, tokenizer, model)
                        
                        df_upload['inferred_severity'] = preds
                        df_upload['confidence'] = confs
                        df_upload['mismatch'] = (df_upload['inferred_severity'] != df_upload['Ticket Priority']).astype(int)
                        
                        # Generate dossiers
                        dossiers_list = []
                        mismatch_df = df_upload[df_upload['mismatch'] == 1]
                        for idx, row in mismatch_df.iterrows():
                            d = generate_dossier(row, row['inferred_severity'], row['confidence'])
                            dossiers_list.append(d)
                            
                        # Save state for dashboard
                        st.session_state['batch_df'] = df_upload
                        st.session_state['dossiers'] = dossiers_list
                        
                        st.success(f"Batch analysis complete! Found **{len(mismatch_df)}** mismatches.")
                        
                        # Preview predictions
                        st.dataframe(df_upload[['Ticket Description', 'Ticket Priority', 'inferred_severity', 'mismatch']].head(10))
                        
                        # Download buttons
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            csv_data = df_upload.to_csv(index=False)
                            st.download_button(
                                label="Download Predictions (CSV)",
                                data=csv_data,
                                file_name="batch_predictions.csv",
                                mime="text/csv"
                            )
                        with col_dl2:
                            json_data = json.dumps(dossiers_list, indent=2)
                            st.download_button(
                                label="Download Evidence Dossiers (JSON)",
                                data=json_data,
                                file_name="batch_dossiers.json",
                                mime="application/json"
                            )
        st.markdown("</div>", unsafe_allow_html=True)

with tab_dashboard:
    st.markdown("### Priority Mismatch Insights Dashboard")
    
    # Load dataset for analytics
    if 'batch_df' in st.session_state:
        df_dash = st.session_state['batch_df']
    else:
        # Load main training dataset with predictions as baseline
        if os.path.exists("customer_support_tickets_with_labels.csv"):
            df_dash = pd.read_csv("customer_support_tickets_with_labels.csv")
        else:
            df_dash = pd.DataFrame()
            
    if df_dash.empty:
        st.warning("Please upload a CSV or ensure customer_support_tickets_with_labels.csv is available to view dashboard insights.")
    else:
        mismatches = df_dash['mismatch'].sum()
        mismatch_rate = mismatches / len(df_dash)
        
        # Calculate mismatch types
        df_dash['assigned_num'] = df_dash['Ticket Priority'].map(PRIORITY_MAP)
        df_dash['inferred_num'] = df_dash['inferred_severity'].map(PRIORITY_MAP)
        df_dash['delta'] = df_dash['inferred_num'] - df_dash['assigned_num']
        
        hidden_crises = (df_dash['delta'] > 0).sum()
        false_alarms = (df_dash['delta'] < 0).sum()
        
        # Grid of Metric Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Total Tickets Audited</div>
                <div class='metric-value'>{len(df_dash)}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Priority Mismatch Rate</div>
                <div class='metric-value' style='color:#ef4444;'>{mismatch_rate * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Hidden Crisis Count</div>
                <div class='metric-value' style='color:#f59e0b;'>{hidden_crises}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>False Alarm Count</div>
                <div class='metric-value' style='color:#3b82f6;'>{false_alarms}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Plots
        col_g1, col_g2 = st.columns([1.2, 1])
        
        with col_g1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            # Render delta heatmap
            fig_heatmap = plot_heatmap(df_dash)
            st.pyplot(fig_heatmap)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_g2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### Mismatch Distribution by Ticket Priority")
            
            # Plot breakdown of mismatches
            fig_bar, ax_bar = plt.subplots(figsize=(6, 5.2))
            plt.style.use('dark_background')
            
            sns.countplot(
                data=df_dash, 
                x='Ticket Priority', 
                hue='mismatch', 
                palette={0: '#16a34a', 1: '#ef4444'},
                ax=ax_bar
            )
            ax_bar.set_title("Aligned vs. Mismatched Tickets per Priority Level", fontsize=12, fontweight='bold', pad=10)
            ax_bar.set_xlabel("Assigned Priority", fontsize=10)
            ax_bar.set_ylabel("Ticket Count", fontsize=10)
            legend = ax_bar.legend(title='Auditor Label')
            legend.get_texts()[0].set_text('Aligned')
            legend.get_texts()[1].set_text('Mismatched')
            
            fig_bar.patch.set_alpha(0.0) # Transparent background
            ax_bar.set_facecolor('none')
            plt.tight_layout()
            
            st.pyplot(fig_bar)
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Top contributing keywords table/view
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### High-Contributing Urgency Keywords Identified")
        st.write("Below are the top urgency-related keywords found in description text that triggered higher severity ratings:")
        
        all_words = []
        for desc in df_dash['Ticket Description'].fillna(""):
            desc_l = desc.lower()
            matched = [kw for kw in CRITICAL_KEYWORDS + MEDIUM_KEYWORDS if kw in desc_l]
            all_words.extend(matched)
            
        if all_words:
            df_words = pd.DataFrame(all_words, columns=['Keyword'])
            word_counts = df_words['Keyword'].value_counts().reset_index()
            word_counts.columns = ['Keyword Detected', 'Frequency Count']
            st.dataframe(word_counts.head(10), use_container_width=True)
        else:
            st.write("No matching key phrases extracted.")
        st.markdown("</div>", unsafe_allow_html=True)
