
import time
import requests
import sys
import threading

BASE_URL = "http://127.0.0.1:8080"

def test_endpoints():
    print("Waiting for server to start...")
    time.sleep(3)
    
    try:
        # 1. Test Config
        print("Testing /api/config...")
        res = requests.get(f"{BASE_URL}/api/config")
        if res.status_code == 200:
            data = res.json()
            print(f"✅ /api/config passed. Root Dir: '{data.get('root_dir')}'")
        else:
            print(f"❌ /api/config failed: {res.status_code}")
            sys.exit(1)

        # 2. Test Files
        print("Testing /api/files...")
        res = requests.get(f"{BASE_URL}/api/files")
        if res.status_code == 200:
            print(f"✅ /api/files passed.")
        else:
            print(f"❌ /api/files failed: {res.status_code}")
            sys.exit(1)

        print("✅ All initial tests passed.")
        
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_endpoints()
