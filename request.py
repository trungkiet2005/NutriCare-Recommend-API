import requests
import json

def call_recommend_for_new_user_api():
    """
    Function to call the recommend_for_new_user API endpoint using requests library
    """
    # API endpoint
    url = "http://localhost:8000/recommend_for_new_user"
    
    # Request data
    payload = {
        "gender": "Nam",
        "age_group": "Từ 25 đến 34 tuổi",
        "race": "Kinh",
        "household_income": "7 - 10 triệu/tháng",
        "education": "Đại học",
        "symptom": ["Đau đầu", "Mệt mỏi"],
        "spefical_diet": ["Ăn chay"],
        "disease": ["Tiểu đường"]
    }
    
    # Set headers
    headers = {
        "Content-Type": "application/json",
    }
    
    print("Gửi yêu cầu đến API...")
    
    # Make the POST request
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            
            print("\n=== KẾT QUẢ API ===")
            print(f"Status: {data['status']}")
            
            # Print recommendations
            if 'recommendations' in data and len(data['recommendations']) > 0:
                print(f"\nTổng số món ăn được đề xuất: {len(data['recommendations'])}")
                print("\nDanh sách món ăn được đề xuất:")
                
                for i, food in enumerate(data['recommendations'][:5], 1):
                    print(f"\n{i}. {food.get('name', 'Unknown')}")
                    print(f"   Nguyên liệu: {food.get('ingredients', 'Không có thông tin')}")
            
            # Print generated tags if available
            if 'generated_tags' in data and data['generated_tags']:
                tag_names = [
                    "low_calorie", "high_calorie", "low_carb", "high_carb", 
                    "low_protein", "high_protein", "low_sugar", "high_sugar", 
                    "low_saturated_fat", "high_saturated_fat", "low_cholesterol", 
                    "high_cholesterol", "low_sodium", "high_sodium"
                ]
                
                print("\nThẻ dinh dưỡng được sinh ra:")
                active_tags = [tag_names[i] for i, val in enumerate(data['generated_tags']) if val == 1]
                print(", ".join(active_tags) if active_tags else "Không có thẻ dinh dưỡng")
            
            # Print full JSON response (uncomment if needed)
            # print("\nJSON Response:")
            # print(json.dumps(data, indent=2, ensure_ascii=False))
            
        else:
            print(f"Lỗi {response.status_code}: {response.text}")
    
    except requests.exceptions.RequestException as e:
        print(f"Lỗi kết nối: {e}")
    except json.JSONDecodeError:
        print("Lỗi khi phân tích JSON response")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")

if __name__ == "__main__":
    call_recommend_for_new_user_api()