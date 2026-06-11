import os
import urllib.request
import pandas as pd

def download_dataset():
    url = "https://huggingface.co/datasets/chiapudding/kaggle-customer-service/resolve/main/customer_support_tickets.csv"
    output_filename = "customer_support_tickets.csv"
    
    if os.path.exists(output_filename):
        print(f"Dataset already exists at {output_filename}. Checking content...")
    else:
        print(f"Downloading dataset from {url}...")
        try:
            urllib.request.urlretrieve(url, output_filename)
            print("Download completed successfully!")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False

    try:
        df = pd.read_csv(output_filename)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        print("Columns present:")
        print(df.columns.tolist())
        return True
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return False

if __name__ == "__main__":
    download_dataset()
