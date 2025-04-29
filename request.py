import requests
import json

# Lấy URL mới từ terminal output
public_url = "https://5b10-35-227-150-67.ngrok-free.app/"  # Thay thế bằng URL thật

url = f"{public_url}/recommend"
data = {"user_id": 21015}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=data, headers=headers)

if response.status_code == 200:
    print("✅ Kết quả từ API:", json.dumps(response.json(), indent=2, ensure_ascii=False))
else:
    print(f"🚨 Lỗi: {response.status_code} - {response.text}")