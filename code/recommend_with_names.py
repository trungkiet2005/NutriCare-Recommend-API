import torch
import logging
import argparse
import json
from typing import List, Dict, Union, Optional
from RCSYS_models import SGSL
from food_converter import FoodConverter

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_and_graph(model_path: str = 'vn_trained_model.pth', graph_path: str = 'vn_food_graph.pt'):
    """
    Tải mô hình và đồ thị đã huấn luyện
    
    Args:
        model_path: Đường dẫn đến file mô hình
        graph_path: Đường dẫn đến file đồ thị
        
    Returns:
        tuple: (model, graph) - mô hình và đồ thị đã tải
    """
    try:
        # Tải đồ thị
        graph = torch.load(graph_path)
        logger.info(f"Đã tải đồ thị từ {graph_path}")
        
        # Khởi tạo mô hình với cùng tham số như khi huấn luyện
        HIDDEN_DIM = 128
        FEATURE_THRESHOLD = 0.3
        LAYERS = 3
        
        model = SGSL(graph, embedding_dim=HIDDEN_DIM, feature_threshold=FEATURE_THRESHOLD, num_layer=LAYERS)
        
        # Tải trọng số đã huấn luyện
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Đặt model ở chế độ đánh giá (không huấn luyện)
        logger.info(f"Đã tải mô hình từ {model_path}")
        
        return model, graph
    
    except FileNotFoundError as e:
        logger.error(f"Không tìm thấy file: {str(e)}")
        return None, None
    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình hoặc đồ thị: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

def recommend_dishes_with_names(
    user_id: int, 
    top_k: int = 10,
    model_path: str = 'vn_trained_model.pth',
    graph_path: str = 'vn_food_graph.pt',
    food_csv_path: str = 'food_tagging.csv',
    include_nutrition: bool = False
) -> Dict:
    """
    Đề xuất món ăn cho người dùng và trả về tên món ăn thay vì chỉ ID
    
    Args:
        user_id: ID của người dùng
        top_k: Số lượng món ăn đề xuất
        model_path: Đường dẫn đến file mô hình
        graph_path: Đường dẫn đến file đồ thị
        food_csv_path: Đường dẫn đến file CSV chứa thông tin về các món ăn
        include_nutrition: Có bao gồm thông tin dinh dưỡng không
        
    Returns:
        Dict: Kết quả đề xuất bao gồm thông tin về người dùng và các món ăn được đề xuất
    """
    # Tải mô hình và đồ thị
    model, graph = load_model_and_graph(model_path, graph_path)
    
    if model is None or graph is None:
        logger.error("Không thể tải mô hình hoặc đồ thị")
        return {"error": "Không thể tải mô hình hoặc đồ thị"}
    
    # Khởi tạo food converter
    converter = FoodConverter(food_csv_path)
    
    # Chuyển sang device phù hợp (CPU hoặc CUDA nếu có)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Tìm index của user
    user_idx = None
    for i, node_id in enumerate(graph['user'].node_id):
        if node_id.item() == user_id:
            user_idx = i
            break
    
    if user_idx is None:
        logger.error(f"Không tìm thấy người dùng với ID {user_id}")
        return {"error": f"Không tìm thấy người dùng với ID {user_id}"}
    
    # Chuẩn bị dữ liệu
    feature_dict = {'user': graph['user'].x.to(device), 'food': graph['food'].x.to(device)}
    edge_index = graph[('user', 'eats', 'food')].edge_index.to(device)
    edge_label_index = graph[('user', 'eats', 'food')].edge_label_index.to(device)
    
    # Forward pass qua mô hình
    with torch.no_grad():  # Không tính gradient khi đánh giá
        users_emb_final, _, items_emb_final, _ = model.forward(
            feature_dict, edge_index, edge_label_index, edge_label_index
        )
    
    # Tính điểm cho tất cả món ăn
    user_emb = users_emb_final[user_idx].unsqueeze(0)
    scores = torch.mm(user_emb, items_emb_final.t()).squeeze()
    
    # Xác định các món ăn đã tiêu thụ cần loại trừ
    consumed_food_indices = []
    for i in range(edge_index.size(1)):
        if edge_index[0, i].item() == user_idx:
            consumed_food_indices.append(edge_index[1, i].item())
    
    # Đặt điểm của món ăn đã tiêu thụ thành -inf
    scores[consumed_food_indices] = -float('inf')
    
    # Lấy k món ăn có điểm cao nhất
    _, indices = torch.topk(scores, top_k)
    indices = indices.cpu().numpy().tolist()
    
    # Định dạng kết quả và chuyển đổi ID thành tên món ăn
    result = converter.format_recommendations(user_id, indices, include_nutrition)
    
    return result

def export_recommendation_to_json(
    user_id: int, 
    output_file: str, 
    top_k: int = 10, 
    include_nutrition: bool = True
):
    """
    Xuất đề xuất món ăn ra file JSON
    
    Args:
        user_id: ID của người dùng
        output_file: Tên file JSON đầu ra
        top_k: Số lượng món ăn đề xuất
        include_nutrition: Có bao gồm thông tin dinh dưỡng không
    """
    result = recommend_dishes_with_names(user_id, top_k, include_nutrition=include_nutrition)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        logger.info(f"Đã xuất kết quả đề xuất ra file {output_file}")
    except Exception as e:
        logger.error(f"Lỗi khi xuất file JSON: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Đề xuất món ăn cho người dùng')
    parser.add_argument('--user_id', type=int, required=True, help='ID của người dùng')
    parser.add_argument('--top_k', type=int, default=10, help='Số lượng món ăn đề xuất')
    parser.add_argument('--output', type=str, default='', help='Tên file JSON đầu ra (nếu không cung cấp, kết quả sẽ in ra console)')
    parser.add_argument('--include_nutrition', action='store_true', help='Bao gồm thông tin dinh dưỡng')
    
    args = parser.parse_args()
    
    # Đề xuất món ăn
    results = recommend_dishes_with_names(args.user_id, args.top_k, include_nutrition=args.include_nutrition)
    
    if "error" in results:
        logger.error(results["error"])
        return
    
    # Xuất kết quả
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            logger.info(f"Đã xuất kết quả đề xuất ra file {args.output}")
        except Exception as e:
            logger.error(f"Lỗi khi xuất file JSON: {str(e)}")
    else:
        # In kết quả ra console
        print(f"Đề xuất cho người dùng {args.user_id}:")
        for rec in results["recommendations"]:
            print(f"{rec['rank']}. {rec['name']}")
            if args.include_nutrition and "nutrition" in rec:
                print(f"   - Dinh dưỡng: Calories: {rec['nutrition'].get('Calories', 'N/A')}, " 
                      f"Carbohydrate: {rec['nutrition'].get('Carbohydrate', 'N/A')}, "
                      f"Protein: {rec['nutrition'].get('Protein', 'N/A')}")

if __name__ == "__main__":
    main()