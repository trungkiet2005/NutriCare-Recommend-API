
import requests
import json


url = "https://huynhtrungkiet09032005-food-recommend-api.hf.space/recommend_for_new_user"
headers = {
    "Content-Type": "application/json"
}


payload = {
    "gender": "Nữ",
    "age_group": "Từ 35 đến 44 tuổi",
    "race": "Kinh",
    "household_income": "10 - 15 triệu/tháng",
    "education": "Đại học",
    "symptom": ["mệt mỏi", "đau đầu"],
    "spefical_diet": ["ăn kiêng"],
    "disease": ["huyết áp cao", "tiểu đường"],
}




response = requests.post(url, headers=headers, json=payload)
print(response.json())

recommendations = response.json().get("recommendations", [])
print(f"Số lượng món recommend: {len(recommendations)}")

with open("response.json", "w", encoding="utf-8") as file:
    json.dump(response.json(), file, ensure_ascii=False, indent=4)
