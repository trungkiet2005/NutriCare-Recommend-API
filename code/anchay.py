import pandas as pd

# Load the CSV file
file_path = "food_tagging.csv"
df = pd.read_csv(file_path)

# Display the first few rows and column names to understand the structure
print(df.head(), df.columns)

# Lọc các món ăn chay dựa vào nguyên liệu (loại trừ các nguyên liệu có nguồn gốc từ động vật)
# Danh sách các nguyên liệu KHÔNG phải món chay (có thể mở rộng thêm nếu cần)
non_vegan_keywords = [
    "cá", "tôm", "mực", "bò", "gà", "heo", "lợn", "trứng", "xúc xích", "thịt", "sò", "điệp", "bầu", "cua", "hàu", "sò điệp", "tôm hùm", "cá hồi", "cá ngừ", "cá trích", "vịt lợn", "thịt gà", "thịt bò", "thịt heo", "thịt vịt", "thịt ngan", "thịt cừu", "thịt dê", "trà",
    "thịt ngựa", "thịt trâu", "thịt nai", "thịt cầy", "thịt chuột", "thịt rắn", "thịt nhím", "thịt sóc", "thịt gà tây", "vịt", "trà" 
    "lưỡi", "mỡ", "tai", "mũi", "bò viên", "cua", "ốc", "ba chỉ", "tim", "gan", "má heo", "giò", "xương", "hến", "pancake", "chocolate", 
    "lòng", "chả", "chim", "dê", "ếch", "rắn", "thỏ", "cút", "bạch tuộc", "hải sản", "lòng đỏ", "rọi" "nhộng", "dồi", "lươn", "ghẹ", "nhộng", "sườn", "ba khía", "nghêu", "ngư", "tép", "lù", "rọi"
]

# Chuyển toàn bộ nguyên liệu sang chữ thường
df['Nguyên liệu_lower'] = (df['Nguyên liệu'] + "   " + df["Tiêu đề"]).astype(str).str.lower()

# Hàm kiểm tra món ăn chay (không chứa từ khóa nguyên liệu động vật)
def is_vegan(ingredient_text):
    return not any(keyword in ingredient_text for keyword in non_vegan_keywords)

# Áp dụng hàm lọc
df['is_vegan'] = df['Nguyên liệu_lower'].apply(is_vegan)

# Lấy danh sách các món ăn chay
vegan_dishes = df[df['is_vegan'] & df['Tên món ăn'].notna()]['Tên món ăn'].dropna().unique().tolist()
print(vegan_dishes, len(vegan_dishes))  # Hiển thị 20 món đầu tiên và tổng số 

with open("vegan_dishes.txt", "w", encoding="utf-8") as f:
    for dish in vegan_dishes:
        f.write(f"{dish}\n")

