import pandas as pd
import os
import logging
import torch
from typing import List, Dict, Union, Optional, Tuple

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FoodConverter:
    """
    Lớp tiện ích để chuyển đổi giữa food_id và tên món ăn
    """
    
    def __init__(self, food_tagging_path: str = 'food_tagging.csv'):
        """
        Khởi tạo bộ chuyển đổi
        
        Args:
            food_tagging_path: Đường dẫn đến file CSV chứa dữ liệu món ăn
        """
        self.food_tagging_path = food_tagging_path
        self._load_food_data()
        
    def _load_food_data(self) -> None:
        """Tải dữ liệu món ăn từ file CSV"""
        try:
            self.df_food = pd.read_csv(self.food_tagging_path)
            logger.info(f"Đã tải {len(self.df_food)} món ăn từ {self.food_tagging_path}")
            
            # Tạo mapping từ ID sang tên và ngược lại
            self.id_to_name = {i: name for i, name in enumerate(self.df_food['Tên món ăn'])}
            self.name_to_id = {name: i for i, name in enumerate(self.df_food['Tên món ăn'])}
            
        except FileNotFoundError:
            logger.error(f"Không tìm thấy file {self.food_tagging_path}")
            self.df_food = None
            self.id_to_name = {}
            self.name_to_id = {}
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu món ăn: {str(e)}")
            self.df_food = None
            self.id_to_name = {}
            self.name_to_id = {}
    
    def food_id_to_name(self, food_ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """
        Chuyển đổi food ID sang tên món ăn
        
        Args:
            food_ids: ID hoặc danh sách ID của món ăn
            
        Returns:
            Tên món ăn hoặc danh sách tên món ăn
        """
        if isinstance(food_ids, int):
            return self.id_to_name.get(food_ids, f"Unknown food ID: {food_ids}")
        elif isinstance(food_ids, list):
            return [self.id_to_name.get(food_id, f"Unknown food ID: {food_id}") for food_id in food_ids]
        elif isinstance(food_ids, torch.Tensor):
            food_ids = food_ids.cpu().numpy().tolist()
            return [self.id_to_name.get(food_id, f"Unknown food ID: {food_id}") for food_id in food_ids]
        else:
            logger.error(f"Không hỗ trợ kiểu dữ liệu {type(food_ids)}")
            return None
    
    def food_name_to_id(self, food_names: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Chuyển đổi tên món ăn sang food ID
        
        Args:
            food_names: Tên hoặc danh sách tên món ăn
            
        Returns:
            ID hoặc danh sách ID của món ăn
        """
        if isinstance(food_names, str):
            return self.name_to_id.get(food_names, -1)
        elif isinstance(food_names, list):
            return [self.name_to_id.get(name, -1) for name in food_names]
        else:
            logger.error(f"Không hỗ trợ kiểu dữ liệu {type(food_names)}")
            return None
    
    def get_food_info(self, food_id: int) -> Dict:
        """
        Lấy toàn bộ thông tin của một món ăn dựa trên ID
        
        Args:
            food_id: ID của món ăn
            
        Returns:
            Dict chứa thông tin của món ăn
        """
        if self.df_food is None:
            return {}
        
        try:
            food_info = self.df_food.iloc[food_id].to_dict()
            return food_info
        except (IndexError, KeyError):
            logger.error(f"Không tìm thấy thông tin cho món ăn có ID {food_id}")
            return {}
    
    def get_nutrition_info(self, food_id: int) -> Dict:
        """
        Lấy thông tin dinh dưỡng của một món ăn dựa trên ID
        
        Args:
            food_id: ID của món ăn
            
        Returns:
            Dict chứa thông tin dinh dưỡng của món ăn
        """
        food_info = self.get_food_info(food_id)
        if not food_info:
            return {}
        
        nutrition_cols = [
            'Carbohydrate', 'Calories', 'Protein', 'Sugar', 
            'Fiber dietary', 'Vitamin C', 'Vitamin D', 'Vitamin B12',
            'Calcium', 'Iron', 'Cholesterol', 'Phosphorous', 
            'Folic acid', 'Saturated fat', 'Potassium', 'Sodium'
        ]
        
        nutrition_info = {col: food_info.get(col, 0) for col in nutrition_cols if col in food_info}
        return nutrition_info
    
    def search_food_by_name(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Tìm kiếm món ăn theo tên
        
        Args:
            query: Chuỗi tìm kiếm
            limit: Số lượng kết quả tối đa
            
        Returns:
            Danh sách các món ăn phù hợp với từ khóa tìm kiếm
        """
        if self.df_food is None:
            return []
        
        # Tìm kiếm trong cột 'Tên món ăn' và 'Tên món ăn_lower'
        query = query.lower()
        
        # Tìm kiếm chính xác
        exact_matches = self.df_food[self.df_food['Tên món ăn_lower'] == query]
        
        # Tìm kiếm các món có chứa từ khóa
        contains_matches = self.df_food[self.df_food['Tên món ăn_lower'].str.contains(query)]
        
        # Loại bỏ các kết quả trùng lặp
        results = pd.concat([exact_matches, contains_matches]).drop_duplicates()
        
        if len(results) == 0:
            return []
        
        # Trả về thông tin cần thiết
        results = results.head(limit)
        result_list = []
        
        for idx, row in results.iterrows():
            result_list.append({
                'food_id': idx,
                'name': row['Tên món ăn'],
                'nutrition': {
                    col: row[col] for col in ['Carbohydrate', 'Calories', 'Protein'] 
                    if col in row and not pd.isna(row[col])
                }
            })
        
        return result_list

    def get_food_count(self) -> int:
        """
        Lấy tổng số lượng món ăn
        
        Returns:
            Số lượng món ăn trong cơ sở dữ liệu
        """
        return len(self.id_to_name)
    
    def get_all_foods(self) -> List[Dict]:
        """
        Lấy danh sách tất cả các món ăn
        
        Returns:
            Danh sách tất cả các món ăn với ID và tên
        """
        return [{'food_id': food_id, 'name': name} for food_id, name in self.id_to_name.items()]

    def format_recommendations(self, 
                              user_id: int, 
                              recommendations: List[int],
                              include_nutrition: bool = False) -> Dict:
        """
        Định dạng kết quả đề xuất thành dạng dễ đọc
        
        Args:
            user_id: ID của người dùng
            recommendations: Danh sách ID món ăn được đề xuất
            include_nutrition: Có bao gồm thông tin dinh dưỡng không
            
        Returns:
            Dict chứa thông tin về người dùng và các đề xuất món ăn
        """
        food_names = self.food_id_to_name(recommendations)
        
        result = {
            "user_id": user_id,
            "recommendations": []
        }
        
        for i, (food_id, name) in enumerate(zip(recommendations, food_names), 1):
            food_info = {"food_id": food_id, "name": name, "rank": i}
            
            if include_nutrition:
                food_info["nutrition"] = self.get_nutrition_info(food_id)
                
            result["recommendations"].append(food_info)
            
        return result

# Ví dụ sử dụng
if __name__ == "__main__":
    converter = FoodConverter()
    
    # Ví dụ chuyển đổi ID sang tên
    food_id = 0
    print(f"ID {food_id} tương ứng với món: {converter.food_id_to_name(food_id)}")
    
    # Ví dụ chuyển đổi tên sang ID
    food_name = "Cá trứng chiên xốt trứng muối"
    print(f"Món {food_name} có ID: {converter.food_name_to_id(food_name)}")
    
    # Ví dụ tìm kiếm món ăn theo tên
    query = "cá"
    search_results = converter.search_food_by_name(query)
    print(f"Kết quả tìm kiếm cho '{query}':")
    for result in search_results:
        print(f"- ID: {result['food_id']}, Tên: {result['name']}")
    
    # Ví dụ định dạng kết quả đề xuất
    user_id = 21005
    recommendations = [0, 2, 5, 7, 9]
    formatted_results = converter.format_recommendations(user_id, recommendations, include_nutrition=True)
    print(f"Đề xuất cho người dùng {user_id}:")
    for rec in formatted_results["recommendations"]:
        print(f"- {rec['rank']}. {rec['name']}")