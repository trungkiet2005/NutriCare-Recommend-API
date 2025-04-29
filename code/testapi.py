import google.generativeai as genai

# Cấu hình API key
genai.configure(api_key="AIzaSyDHRsA1G42JsicRxJFMMZZ9chcwLxDoVZU")

# Khởi tạo mô hình Gemini Pro
model = genai.GenerativeModel("gemini-pro")

# Gọi hàm generate_content từ object model
response = model.generate_content("Give me 3 healthy dinner suggestions")

print(response.text)
