# 🦀 Food Recommend API

🌐 **URL**: [Hugging Face Space - Food Recommend API](https://huggingface.co/spaces/huynhtrungkiet09032005/food-recommend-api)

🎯 **Mục tiêu**:  
Cung cấp hệ thống gợi ý món ăn thông minh dựa trên **Graph Neural Network (GNN)**, hỗ trợ cá nhân hóa khẩu phần ăn, đặc biệt hữu ích cho bệnh nhân hoặc người ăn kiêng.

---

## 🚀 Mô tả

Đây là một **API RESTful** được triển khai trên Hugging Face Spaces, sử dụng mô hình học sâu để đưa ra các gợi ý món ăn phù hợp với từng người dùng. Hệ thống sử dụng thông tin:

- Hồ sơ người dùng (bệnh lý, sở thích ăn uống)
- Quan hệ giữa món ăn và thành phần dinh dưỡng
- Mạng đồ thị biểu diễn mối quan hệ người - món ăn - dinh dưỡng

---

## 🧠 Công nghệ sử dụng

- `Python` + `FastAPI`
- `Graph Neural Network (GNN)`
- `Docker` (sử dụng Hugging Face SDK)
- Triển khai trực tiếp trên Hugging Face Spaces

---

## 📦 Các Endpoint chính

| Phương thức | Đường dẫn    | Mô tả                                             |
|------------|--------------|--------------------------------------------------|
| `GET`      | `/`          | Kiểm tra API hoạt động                           |
| `POST`     | `/recommend` | Gửi thông tin người dùng, trả về món ăn gợi ý   |

---

## 📥 Ví dụ sử dụng

### 🎯 JSON gửi vào `/recommend`

```json
{
  "user_id": "user_123",
  "health_condition": "diabetes",
  "preferences": ["low sugar", "vegetarian"]
}
