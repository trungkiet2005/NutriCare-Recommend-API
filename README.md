# 🦀 Food Recommend API

🌐 **URL**: [Hugging Face Space - Food Recommend API](https://huggingface.co/spaces/huynhtrungkiet09032005/food-recommend-api)

🎯 **Mục tiêu**:  
Cung cấp hệ thống gợi ý món ăn thông minh dựa trên **Graph Neural Network (GNN)**, hỗ trợ cá nhân hóa khẩu phần ăn, đặc biệt hữu ích cho bệnh nhân hoặc người ăn kiêng.

---

## 🚀 Mô tả chi tiết

Dự án **Food Recommend API** là một hệ thống đề xuất thông minh được thiết kế để gợi ý các món ăn phù hợp với nhu cầu dinh dưỡng và sở thích cá nhân của người dùng. Hệ thống được xây dựng trên nền tảng của mạng thần kinh đồ thị (GNN) để phân tích và hiểu mối quan hệ phức tạp giữa:

- **Người dùng**: Thông tin sức khỏe, tiền sử bệnh, sở thích ăn uống
- **Món ăn**: Thành phần, giá trị dinh dưỡng, cách chế biến
- **Mối quan hệ**: Tương tác giữa món ăn và tình trạng sức khỏe cụ thể

API này đặc biệt hữu ích cho những đối tượng:
- Người mắc bệnh mãn tính cần chế độ ăn đặc biệt (tiểu đường, tim mạch, huyết áp cao...)
- Người theo đuổi mục tiêu sức khỏe cụ thể (giảm cân, tăng cơ, cải thiện sức khỏe tim mạch...)
- Người có nhu cầu ăn uống đặc biệt (chay, thuần chay, keto, không gluten...)

---

## 🧠 Công nghệ sử dụng

### Backend
- `Python 3.9+` làm ngôn ngữ lập trình chính
- `FastAPI` làm framework phát triển API
- `PyTorch` và `PyTorch Geometric` để xây dựng và huấn luyện GNN
- `pandas` và `numpy` để xử lý dữ liệu
- `scikit-learn` cho các tác vụ machine learning phụ trợ

### Lưu trữ & Xử lý dữ liệu
- `SQLite` cho lưu trữ dữ liệu cơ bản
- `Neo4j` (tùy chọn) cho lưu trữ và truy vấn dữ liệu đồ thị

### Triển khai
- `Docker` để đóng gói ứng dụng
- `Hugging Face Spaces` cho việc triển khai và lưu trữ
- `GitHub Actions` cho CI/CD

---

## 🔧 Cài đặt và Sử dụng

### Yêu cầu hệ thống
- Python 3.9 trở lên
- pip (trình quản lý gói Python)
- Git

### Cài đặt
1. Clone dự án về máy:
   ```bash
   git clone https://github.com/trungkiet2005/NutriCare-Recommend-API.git
   cd NutriCare-Recommend-API
   ```

2. Tạo và kích hoạt môi trường ảo:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. Cài đặt các phụ thuộc:
   ```bash
   pip install -r requirements.txt
   ```

### Chạy ứng dụng
1. Khởi động server:
   ```bash
   python app.py
   ```
   Hoặc với FastAPI:
   ```bash
   uvicorn app.main:app --reload
   ```

2. Truy cập API tại:
   - API: http://localhost:8000
   - Tài liệu API (Swagger): http://localhost:8000/docs

---

## 📦 Các Endpoint API

| Phương thức | Đường dẫn           | Mô tả                                             |
|------------|---------------------|--------------------------------------------------|
| `GET`      | `/`                 | Kiểm tra API hoạt động                           |
| `POST`     | `/recommend`        | Gửi thông tin người dùng, trả về món ăn gợi ý   |
| `GET`      | `/foods`            | Lấy danh sách tất cả các món ăn                 |
| `GET`      | `/foods/{food_id}`  | Lấy thông tin chi tiết của một món ăn           |
| `GET`      | `/users/{user_id}`  | Lấy thông tin người dùng                        |
| `POST`     | `/users`            | Tạo người dùng mới                              |
| `PUT`      | `/users/{user_id}`  | Cập nhật thông tin người dùng                   |

---

## 📥 Ví dụ sử dụng

### 🎯 1. Yêu cầu đề xuất món ăn (`POST /recommend`)

**Body Request:**
```json
{
  "user_id": "user_123",
  "health_condition": "diabetes",
  "preferences": ["low sugar", "vegetarian"],
  "meal_type": "lunch",
  "exclude_ingredients": ["nuts", "dairy"]
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "food_id": "food_001",
      "name": "Salad đậu hũ và rau củ",
      "calories": 320,
      "protein": 15,
      "carbs": 30,
      "fat": 12,
      "suitability_score": 0.92,
      "image_url": "https://example.com/images/tofu_salad.jpg"
    },
    {
      "food_id": "food_045",
      "name": "Cháo yến mạch và rau củ",
      "calories": 280,
      "protein": 10,
      "carbs": 45,
      "fat": 5,
      "suitability_score": 0.87,
      "image_url": "https://example.com/images/veggie_porridge.jpg"
    }
  ]
}
```

### 🎯 2. Lấy thông tin món ăn (`GET /foods/food_001`)

**Response:**
```json
{
  "food_id": "food_001",
  "name": "Salad đậu hũ và rau củ",
  "description": "Món salad giàu protein từ đậu hũ, kết hợp với các loại rau củ tươi như cà rốt, dưa chuột, ớt chuông...",
  "ingredients": [
    {"name": "Đậu hũ", "amount": "150g"},
    {"name": "Cà rốt", "amount": "50g"},
    {"name": "Dưa chuột", "amount": "50g"},
    {"name": "Ớt chuông", "amount": "30g"},
    {"name": "Dầu olive", "amount": "10ml"}
  ],
  "nutrition": {
    "calories": 320,
    "protein": 15,
    "carbs": 30,
    "fat": 12,
    "fiber": 8,
    "sugar": 5
  },
  "suitable_for": ["diabetes", "heart_disease", "weight_loss"],
  "avoid_for": ["nut_allergy"],
  "preparation_time": 15,
  "cooking_time": 10,
  "image_url": "https://example.com/images/tofu_salad.jpg"
}
```

---

## 📂 Cấu trúc dự án

```
NutriCare-Recommend-API/
├── app/
│   ├── __init__.py
│   ├── main.py              # Điểm khởi chạy ứng dụng FastAPI
│   ├── config.py            # Cấu hình ứng dụng
│   ├── api/                 # Các endpoint API
│   │   ├── __init__.py
│   │   ├── recommend.py
│   │   ├── foods.py
│   │   └── users.py
│   ├── models/              # Các model dữ liệu
│   │   ├── __init__.py
│   │   ├── food.py
│   │   ├── user.py
│   │   └── recommendation.py
│   ├── services/            # Các service xử lý logic
│   │   ├── __init__.py
│   │   ├── recommend_service.py
│   │   └── user_service.py
│   ├── database/            # Kết nối và quản lý DB
│   │   ├── __init__.py
│   │   └── connection.py
│   └── ml/                  # Các model ML/GNN
│       ├── __init__.py
│       ├── gnn_model.py
│       └── trainer.py
├── data/                    # Dữ liệu
│   ├── foods.csv
│   ├── users.csv
│   └── interactions.csv
├── notebooks/               # Jupyter notebooks cho phân tích
│   └── model_development.ipynb
├── tests/                   # Unit tests
│   ├── __init__.py
│   ├── test_api.py
│   └── test_model.py
├── .gitignore
├── requirements.txt         # Các phụ thuộc Python
├── Dockerfile               # Cấu hình Docker
├── docker-compose.yml       # Cấu hình Docker Compose
└── README.md                # Tài liệu dự án
```

---

## 📝 Hướng dẫn đóng góp

Chúng tôi rất hoan nghênh mọi đóng góp cho dự án! Nếu bạn muốn tham gia, vui lòng:

1. Fork dự án
2. Tạo nhánh tính năng (`git checkout -b feature/amazing-feature`)
3. Commit các thay đổi (`git commit -m 'Add some amazing feature'`)
4. Push lên nhánh (`git push origin feature/amazing-feature`)
5. Mở Pull Request

### Lĩnh vực cần đóng góp
- Cải thiện mô hình GNN
- Bổ sung dữ liệu món ăn Việt Nam
- Tối ưu hóa API performance
- Phát triển giao diện người dùng (frontend)

---

## 📄 Giấy phép sử dụng

Dự án được phân phối dưới giấy phép Apache 2.0. Xem `LICENSE` để biết thêm chi tiết.

---

## 📞 Liên hệ

- **Email**: example@email.com
- **GitHub**: [https://github.com/trungkiet2005/NutriCare-Recommend-API](https://github.com/trungkiet2005/NutriCare-Recommend-API)
- **Website**: [https://your-website.com](https://your-website.com)
