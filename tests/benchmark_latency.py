import os
import time
import numpy as np
import requests

def run_latency_benchmark(audio_folder="./test_queries", url = "https://orange-webs-shave.loca.lt/chat/audio"):
    latencies = []
    
    print(f"Starting benchmark on {url}...")
    print("-" * 50)
    
    for filename in os.listdir(audio_folder):
        if filename.endswith((".wav", ".m4a")):
            file_path = os.path.join(audio_folder, filename)
            
            with open(file_path, "rb") as audio_file:
                files = {"audio": (filename, audio_file)}
                headers = {"Bypass-Tunnel-Reminder": "true"}
                start_time = time.time()
                try:
                    response = requests.post(url, files=files)
                    end_time = time.time()
                    
                    latency_ms = (end_time - start_time) * 1000
                    
                    if response.status_code == 200:
                        latencies.append(latency_ms)
                        print(f"✅ Success | {filename}: {latency_ms:.2f} ms")
                    else:
                        print(f"❌ Error | {filename}: Status {response.status_code}")
                except Exception as e:
                    print(f"⚠️ Request Failed | {filename}: {str(e)}")
                    
    return latencies

if __name__ == "__main__":
    latencies = run_latency_benchmark()
    
    if latencies:
        p50 = np.percentile(latencies, 50)
        p70 = np.percentile(latencies, 70)
        p100 = np.percentile(latencies, 100) 
        
        print("\n" + "=" * 50)
        print("🚀 FINAL SUBMISSION ANALYTICS 🚀")
        print("=" * 50)
        print(f"P50 Latency : {p50:.2f} ms")
        print(f"P70 Latency : {p70:.2f} ms")
        print(f"P100 Latency: {p100:.2f} ms")
        print("=" * 50)
    else:
        print("\nNo successful requests to calculate latencies.")