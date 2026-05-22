import requests
import time
import pandas as pd
import os
import random

# KONFIGURASI PATH DATASET

paths_to_try = [
    '../Membangun_model/loan_preprocessing/loan_processed.csv',
    'loan_preprocessing/loan_processed.csv',
    'loan_processed.csv'
]

dataset_path = None
for path in paths_to_try:
    if os.path.exists(path):
        dataset_path = path
        break

if not dataset_path:
    print("❌ Error: Berkas 'loan_processed.csv' tidak ditemukan!")
    print("Pastikan path foldernya benar.")
    exit(1)

# KONFIGURASI API

url = "http://127.0.0.1:8000/predict"

print(f"✓ Membaca data riil dari dataset: {dataset_path}")

try:
    df = pd.read_csv(dataset_path)
    
    if 'loan_status' in df.columns:
        X = df.drop(columns=['loan_status'])
    else:
        X = df.copy()
    
    print(f"📊 Dataset loaded: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"🚀 Memulai pengiriman {min(100, len(X))} baris data ke API server...")
    print("="*60)
    
    max_rows = min(100, len(X))
    
    for i in range(max_rows):
        row = X.iloc[i]
        payload = row.to_dict()
        
        try:
            response = requests.post(url, json=payload, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                approval_score = result.get('approval_score', 0)
                approval_status = result.get('approval_status', 'Unknown')
                
                print(f"📋 Data {i+1}: Approval Score = {approval_score:.1f}% | Status: {approval_status}")
            else:
                print(f"❌ Data {i+1}: Error {response.status_code} - {response.text[:50]}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Data {i+1}: Cannot connect to server. Make sure server is running on port 8000")
            break
        except Exception as e:
            print(f"❌ Data {i+1}: {str(e)[:100]}")
        
        time.sleep(0.5)
    
    print("="*60)
    print("✅ Simulasi traffic selesai!")
    print("📈 Cek metrics di http://127.0.0.1:8000/metrics")
    print("📊 Cek Prometheus di http://127.0.0.1:9090")
    print("📉 Cek Grafana di http://127.0.0.1:3000")
            
except KeyboardInterrupt:
    print("\n⚠️ Simulasi traffic dihentikan oleh pengguna.")
except FileNotFoundError:
    print(f"❌ File tidak ditemukan: {dataset_path}")
except Exception as e:
    print(f"❌ Terjadi kesalahan: {e}")