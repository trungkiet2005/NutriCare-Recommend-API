import pandas as pd
import numpy as np


def filter_unwanted_foods(food_df, exclude_keywords=None):
    """
    Lọc DataFrame món ăn dựa trên từ khóa cần loại trừ
    
    Args:
        food_df: DataFrame chứa thông tin món ăn
        exclude_keywords: Danh sách các từ khóa cần loại trừ
        
    Returns:
        DataFrame đã lọc
    """
    if not exclude_keywords or food_df.empty:
        return food_df
        
    # Chuyển tất cả từ khóa sang chữ thường để so sánh không phân biệt hoa thường
    exclude_keywords_lower = [keyword.lower() for keyword in exclude_keywords]
    
    # Xác định các cột văn bản để kiểm tra
    text_columns = []
    for col in food_df.columns:
        # Xác định các cột văn bản tiềm năng cần kiểm tra
        potential_text_cols = [
            'name', 'ingredients', 'Tiêu đề', 'Nguyên liệu', 'Tên món ăn',
            'Thực hiện', 'Sơ chế', 'Cách dùng', 'Mách nhỏ', 'Thực đơn'
        ]
        if col in potential_text_cols:
            text_columns.append(col)
    
    # Nếu không tìm thấy cột văn bản, trả về DataFrame gốc
    if not text_columns:
        return food_df
    
    # Tạo điều kiện lọc
    filter_condition = True  # Khởi tạo với True để kết hợp AND
    
    for col in text_columns:
        # Đảm bảo cột là kiểu chuỗi
        if col in food_df.columns:
            # Tạo một bản sao cột và chuyển thành chữ thường
            food_df[f'{col}_lower'] = food_df[col].astype(str).str.lower()
            
            # Tạo điều kiện cho mỗi từ khóa
            for keyword in exclude_keywords_lower:
                # Kết hợp điều kiện với toán tử AND
                filter_condition = filter_condition & ~food_df[f'{col}_lower'].str.contains(keyword, na=False)
    
    # Áp dụng điều kiện lọc
    filtered_df = food_df[filter_condition].copy()
    
    # Xóa các cột tạm thời
    for col in text_columns:
        if f'{col}_lower' in filtered_df.columns:
            filtered_df = filtered_df.drop(columns=[f'{col}_lower'])
    
    return filtered_df


food_vn_df = pd.read_csv('recipes_cleaned.csv')
food_nutrition_df = pd.read_csv('food_nutrition_vn.csv')

# In ra số dòng không có dữ liệu null trong mỗi DataFrame
print("\nSố dòng không có dữ liệu null trong food_recipe:")
print(food_vn_df.dropna().shape[0])

print("\nSố dòng không có dữ liệu null trong food_nutrition_vn:")
print(food_nutrition_df.dropna().shape[0])

# Đếm số dòng không null trong cột Sơ chế và Thực hiện
print("\nSố dòng không null trong cột Sơ chế:")
print(food_vn_df['Sơ chế'].count())

print("\nSố dòng không null trong cột Thực hiện:")
print(food_vn_df['Thực hiện'].count())

merge_df = pd.merge(food_vn_df, food_nutrition_df, left_on='Tiêu đề', right_on='Tên món ăn', how='left')
exclude_keywords = ["ngọt", "sữa", "bánh", "đường", "sweet", "milk", "cake", "bread", "sugar", "cheese", "candy", "pastry"]
merge_filter_df = filter_unwanted_foods(merge_df, exclude_keywords)


merge_df.to_csv('food_tagging.csv', index=False)
merge_filter_df.to_csv('food_tagging_filter.csv', index=False)

print(merge_filter_df)
print(merge_df)
print(merge_df.columns)
print(merge_df.describe())
print(merge_df.dropna().shape[0])

