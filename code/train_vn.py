import pandas as pd
import torch
import numpy as np
import os
import logging
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch.optim as optim
import torch_geometric
from torch_geometric.utils import structured_negative_sampling
from RCSYS_models import SGSL
from RCSYS_utils import split_data_new, sample_mini_batch, pareto_loss, eval

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Định nghĩa hằng số cho huấn luyện mô hình
SEED = 42
BATCH_SIZE = 2048
LAMBDA = 1e-6
HIDDEN_DIM = 128
LAYERS = 3
LEARNING_RATE = 1e-3
EPOCHS = 60
K = 20  # Số lượng món ăn đề xuất
FEATURE_THRESHOLD = 0.3
ITERS_PER_EVAL = 20
ITERS_PER_LR_DECAY = 200

def create_vietnamese_food_graph():
    """
    Tạo đồ thị mới bằng cách thay thế các node thức ăn trong đồ thị gốc
    với dữ liệu thức ăn Việt Nam.
    
    Returns:
        HeteroData: Đồ thị PyTorch Geometric cho dữ liệu Việt Nam
    """
    # Tải đồ thị US gốc
    logger.info("Đang tải đồ thị US gốc...")
    graph_paths = [
        'processed_data/benchmark_macro.pt',
        '../processed_data/benchmark_macro.pt'
    ]
    
    graph_file = None
    for path in graph_paths:
        if os.path.exists(path):
            graph_file = path
            break
    
    if graph_file is None:
        raise FileNotFoundError("Không tìm thấy file đồ thị US gốc!")
    
    us_graph = torch.load(graph_file)
    logger.info(f"Đã tải đồ thị US với {us_graph['user'].num_nodes} người dùng và {us_graph['food'].num_nodes} thức ăn")
    
    # In kích thước đặc trưng để phân tích
    logger.info(f"Kích thước đặc trưng của user: {us_graph['user'].x.shape}")
    logger.info(f"Kích thước đặc trưng của food: {us_graph['food'].x.shape}")
    logger.info(f"Kích thước tags của user: {us_graph['user'].tags.shape}")
    logger.info(f"Kích thước tags của food: {us_graph['food'].tags.shape}")
    
    # Tải dữ liệu thức ăn Việt Nam
    logger.info("Đang tải dữ liệu thức ăn Việt Nam...")
    df_vn_food = pd.read_csv('food_tagging.csv')
    logger.info(f"Đã tải {len(df_vn_food)} món ăn Việt Nam")
    
    # Xử lý dữ liệu thức ăn Việt Nam
    nutrition_columns = [
        'Carbohydrate', 'Calories', 'Protein', 'Sugar', 
        'Fiber dietary', 'Vitamin C', 'Vitamin D', 'Vitamin B12',
        'Calcium', 'Iron', 'Cholesterol', 'Phosphorous', 
        'Folic acid', 'Saturated fat', 'Potassium', 'Sodium'
    ]
    
    # Lọc và chuẩn hóa dữ liệu dinh dưỡng
    available_nutrition_cols = [col for col in nutrition_columns if col in df_vn_food.columns]
    df_nutrition = df_vn_food[['Tên món ăn'] + available_nutrition_cols].copy()
    df_nutrition.fillna(0, inplace=True)
    
    # Chuẩn hóa giá trị dinh dưỡng
    scaler = StandardScaler()
    nutrition_values = scaler.fit_transform(df_nutrition[available_nutrition_cols].values)
    
    # QUAN TRỌNG: Đảm bảo kích thước đặc trưng của thức ăn Việt Nam 
    # phải giống với kích thước đặc trưng của thức ăn US
    food_feature_dim = us_graph['food'].x.shape[1]
    logger.info(f"Cần điều chỉnh kích thước đặc trưng thức ăn từ {nutrition_values.shape[1]} đến {food_feature_dim}")
    
    # Nếu kích thước không khớp, điều chỉnh bằng cách padding hoặc PCA
    if nutrition_values.shape[1] < food_feature_dim:
        # Padding thêm 0
        padded_values = np.zeros((nutrition_values.shape[0], food_feature_dim))
        padded_values[:, :nutrition_values.shape[1]] = nutrition_values
        nutrition_values = padded_values
        logger.info(f"Đã padding vector dinh dưỡng thành kích thước {nutrition_values.shape}")
    elif nutrition_values.shape[1] > food_feature_dim:
        # Trong trường hợp cần giảm chiều dữ liệu
        from sklearn.decomposition import PCA
        pca = PCA(n_components=food_feature_dim)
        nutrition_values = pca.fit_transform(nutrition_values)
        logger.info(f"Đã giảm chiều vector dinh dưỡng thành kích thước {nutrition_values.shape}")
    
    # Tạo đồ thị mới
    logger.info("Đang tạo đồ thị mới với thức ăn Việt Nam...")
    vn_graph = HeteroData()
    
    # Giữ nguyên các node người dùng và đặc trưng
    vn_graph['user'].x = us_graph['user'].x.clone()
    vn_graph['user'].node_id = us_graph['user'].node_id.clone()
    vn_graph['user'].num_nodes = us_graph['user'].num_nodes
    
    if hasattr(us_graph['user'], 'tags'):
        vn_graph['user'].tags = us_graph['user'].tags.clone()
    
    if hasattr(us_graph['user'], 'prompt'):
        vn_graph['user'].prompt = us_graph['user'].prompt.copy()
    
    if hasattr(us_graph['user'], 'prompt_health'):
        vn_graph['user'].prompt_health = us_graph['user'].prompt_health.copy()
    
    # Tạo đặc trưng cho các node thức ăn Việt Nam
    num_vn_foods = len(df_nutrition)
    vn_graph['food'].x = torch.tensor(nutrition_values, dtype=torch.float)
    vn_graph['food'].node_id = torch.tensor(list(range(num_vn_foods)), dtype=torch.long)
    vn_graph['food'].num_nodes = num_vn_foods
    
    # Tạo tags cho thức ăn Việt Nam - đảm bảo kích thước giống với đồ thị gốc
    food_tag_dim = us_graph['food'].tags.shape[1]
    logger.info(f"Kích thước tags của thức ăn gốc: {food_tag_dim}")
    food_tags = create_vn_food_tags(df_nutrition, tag_dim=food_tag_dim)
    vn_graph['food'].tags = torch.tensor(food_tags, dtype=torch.float)
    
    # Tạo prompt cho thức ăn Việt Nam
    food_prompts = []
    for idx, row in df_nutrition.iterrows():
        food_name = row['Tên món ăn']
        nutrients = ', '.join([f"{col}: {row[col]}" for col in available_nutrition_cols if pd.notna(row[col])])
        prompt = f"Món ăn {food_name}: Món ăn Việt Nam với các thành phần dinh dưỡng: {nutrients}"
        food_prompts.append(prompt)
    
    vn_graph['food'].prompt = food_prompts
    
    # Tạo liên kết giữa người dùng và thức ăn
    logger.info("Đang tạo liên kết giữa người dùng và thức ăn Việt Nam...")
    user_food_edges = create_user_food_edges(us_graph, num_vn_foods)
    vn_graph[('user', 'eats', 'food')].edge_index = user_food_edges['edge_index']
    vn_graph[('user', 'eats', 'food')].edge_label_index = user_food_edges['edge_label_index']
    
    # Lưu đồ thị mới
    torch.save(vn_graph, 'vn_food_graph.pt')
    logger.info("Đã tạo và lưu đồ thị thức ăn Việt Nam thành công!")
    
    return vn_graph

def create_vn_food_tags(df_nutrition, tag_dim=14):
    """
    Tạo các tag nhị phân cho thức ăn dựa trên giá trị dinh dưỡng
    
    Args:
        df_nutrition (DataFrame): DataFrame chứa thông tin dinh dưỡng của thức ăn
        tag_dim (int): Số lượng tag cần tạo
        
    Returns:
        np.ndarray: Ma trận tags nhị phân
    """
    # Xác định ngưỡng cho mỗi thành phần dinh dưỡng
    thresholds = {
        'Carbohydrate': {'low': 40, 'high': 225},
        'Calories': {'low': 40, 'high': 225},
        'Protein': {'low': 10, 'high': 15},
        'Sugar': {'low': 5, 'high': 22.5},
        'Fiber dietary': {'low': 3, 'high': 6},
        'Saturated fat': {'low': 1.5, 'high': 5},
        'Cholesterol': {'low': 20, 'high': 40},
        'Sodium': {'low': 120, 'high': 200},
        'Calcium': {'low': 0, 'high': 150},
        'Phosphorous': {'low': 0, 'high': 105},
        'Potassium': {'low': 0, 'high': 525},
        'Iron': {'low': 0, 'high': 3.3},
        'Folic acid': {'low': 0, 'high': 60},
        'Vitamin C': {'low': 0, 'high': 15},
        'Vitamin D': {'low': 0, 'high': 2.25},
        'Vitamin B12': {'low': 0, 'high': 0.36}
    }
    
    # Khởi tạo ma trận tags với số chiều cố định
    num_foods = len(df_nutrition)
    tags = np.zeros((num_foods, tag_dim), dtype=np.float32)
    
    # Lọc các cột có trong DataFrame
    available_cols = [col for col in thresholds.keys() if col in df_nutrition.columns]
    
    # Tạo tags dựa trên ngưỡng
    tag_idx = 0
    for col in available_cols:
        if tag_idx >= tag_dim - 1:  # Đảm bảo không vượt quá số lượng tag
            break
            
        # Low tag
        tags[:, tag_idx] = df_nutrition[col].apply(
            lambda x: 1 if x <= thresholds[col]['low'] else 0
        ).values
        tag_idx += 1
        
        # High tag (nếu còn chỗ)
        if tag_idx < tag_dim:
            tags[:, tag_idx] = df_nutrition[col].apply(
                lambda x: 1 if x > thresholds[col]['high'] else 0
            ).values
            tag_idx += 1
    
    # Đảm bảo số lượng tag đúng
    logger.info(f"Đã tạo {tag_idx} tags từ dữ liệu dinh dưỡng")
    
    return tags

def create_user_food_edges(us_graph, num_vn_foods):
    """
    Tạo liên kết giữa người dùng và thức ăn Việt Nam dựa trên mẫu của đồ thị gốc
    
    Args:
        us_graph (HeteroData): Đồ thị US gốc
        num_vn_foods (int): Số lượng món ăn Việt Nam
        
    Returns:
        dict: Dict chứa edge_index và edge_label_index cho đồ thị mới
    """
    num_users = us_graph['user'].num_nodes
    
    # Tỷ lệ số lượng cạnh trên tổng số cạnh có thể có
    original_edges = us_graph[('user', 'eats', 'food')].edge_index
    original_edge_ratio = original_edges.shape[1] / (us_graph['user'].num_nodes * us_graph['food'].num_nodes)
    
    # Tỷ lệ cạnh nhãn dương trên tổng số cạnh
    original_label_edges = us_graph[('user', 'eats', 'food')].edge_label_index
    original_label_ratio = original_label_edges.shape[1] / original_edges.shape[1]
    
    # Tạo cạnh mới theo tỷ lệ tương tự
    # Mỗi người dùng sẽ được kết nối với một tập hợp các món ăn Việt Nam
    target_num_edges = int(original_edge_ratio * num_users * num_vn_foods)
    target_num_label_edges = int(original_label_ratio * target_num_edges)
    
    logger.info(f"Mục tiêu: {target_num_edges} cạnh tổng, {target_num_label_edges} cạnh nhãn")
    
    # Tạo cạnh ngẫu nhiên giữa người dùng và thức ăn
    user_indices = []
    food_indices = []
    
    for user_idx in range(num_users):
        # Số lượng món ăn cho mỗi người dùng theo phân phối mũ
        num_foods_per_user = np.random.randint(10, 40)
        # Chọn ngẫu nhiên các món ăn không lặp lại
        selected_foods = np.random.choice(num_vn_foods, min(num_foods_per_user, num_vn_foods), replace=False)
        
        for food_idx in selected_foods:
            user_indices.append(user_idx)
            food_indices.append(food_idx)
    
    # Đảm bảo số lượng cạnh không vượt quá target
    if len(user_indices) > target_num_edges:
        indices = np.random.choice(len(user_indices), target_num_edges, replace=False)
        user_indices = [user_indices[i] for i in indices]
        food_indices = [food_indices[i] for i in indices]
    
    # Tạo cạnh nhãn (giả định một phần nhỏ các cạnh là cạnh nhãn dương)
    label_indices = np.random.choice(len(user_indices), target_num_label_edges, replace=False)
    label_user_indices = [user_indices[i] for i in label_indices]
    label_food_indices = [food_indices[i] for i in label_indices]
    
    # Tạo tensors
    edge_index = torch.tensor([user_indices, food_indices], dtype=torch.long)
    edge_label_index = torch.tensor([label_user_indices, label_food_indices], dtype=torch.long)
    
    logger.info(f"Đã tạo {edge_index.shape[1]} cạnh và {edge_label_index.shape[1]} cạnh nhãn")
    
    return {
        'edge_index': edge_index,
        'edge_label_index': edge_label_index
    }

def train_vietnamese_model():
    """
    Huấn luyện mô hình SGSL trên đồ thị thức ăn Việt Nam
    
    Returns:
        SGSL: Mô hình đã huấn luyện
    """
    # Đặt seed để tái tạo kết quả
    torch_geometric.seed_everything(SEED)
    
    # Tải đồ thị
    logger.info("Đang tải đồ thị thức ăn Việt Nam...")
    if not os.path.exists('vn_food_graph.pt'):
        logger.info("Đồ thị chưa tồn tại, đang tạo mới...")
        graph = create_vietnamese_food_graph()
    else:
        graph = torch.load('vn_food_graph.pt')
    
    # Lấy các thuộc tính của đồ thị
    num_users = graph['user'].num_nodes
    num_foods = graph['food'].num_nodes
    edge_index = graph[('user', 'eats', 'food')].edge_index
    edge_label_index = graph[('user', 'eats', 'food')].edge_label_index
    feature_dict = {'user': graph['user'].x, 'food': graph['food'].x}
    
    logger.info(f"Đồ thị đã tải với {num_users} người dùng và {num_foods} món ăn")
    logger.info(f"Edge index shape: {edge_index.shape}")
    logger.info(f"Edge label index shape: {edge_label_index.shape}")
    
    # In kích thước đặc trưng để kiểm tra tính đồng nhất
    logger.info(f"Kích thước đặc trưng user: {feature_dict['user'].shape}")
    logger.info(f"Kích thước đặc trưng food: {feature_dict['food'].shape}")
    logger.info(f"Kích thước tags user: {graph['user'].tags.shape}")
    logger.info(f"Kích thước tags food: {graph['food'].tags.shape}")
    
    # Chia dữ liệu thành train/val/test
    try:
        train_edge_index, val_edge_index, test_edge_index, \
        pos_train_edge_index, neg_train_edge_index, pos_val_edge_index, neg_val_edge_index, \
        pos_test_edge_index, neg_test_edge_index = split_data_new(edge_index, edge_label_index)
        
        logger.info(f"Đã chia dữ liệu thành train: {train_edge_index.shape[1]}, val: {val_edge_index.shape[1]}, test: {test_edge_index.shape[1]} cạnh")
    except Exception as e:
        logger.error(f"Lỗi khi chia dữ liệu: {str(e)}")
        raise
    
    # Khởi tạo mô hình
    try:
        model = SGSL(graph, embedding_dim=HIDDEN_DIM, feature_threshold=FEATURE_THRESHOLD, num_layer=LAYERS)
        logger.info("Đã khởi tạo mô hình SGSL thành công")
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo mô hình: {str(e)}")
        raise
    
    # Thiết lập device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Sử dụng device: {device}")
    
    # Chuyển dữ liệu sang device
    train_edge_index = train_edge_index.to(device)
    val_edge_index = val_edge_index.to(device)
    test_edge_index = test_edge_index.to(device)
    pos_train_edge_index = pos_train_edge_index.to(device)
    neg_train_edge_index = neg_train_edge_index.to(device)
    pos_val_edge_index = pos_val_edge_index.to(device)
    neg_val_edge_index = neg_val_edge_index.to(device)
    pos_test_edge_index = pos_test_edge_index.to(device)
    neg_test_edge_index = neg_test_edge_index.to(device)
    
    feature_dict = {key: x.to(device) for key, x in feature_dict.items()}
    
    # Trích xuất đặc trưng tag
    user_tags = graph['user'].tags.to(device)
    food_tags = graph['food'].tags.to(device)
    user_features = graph['user'].x.to(device)
    food_features = graph['food'].x.to(device)
    
    # Chuyển mô hình sang device
    model = model.to(device)
    model.train()
    
    # Khởi tạo optimizer và scheduler
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    # Vòng lặp huấn luyện
    train_losses = []
    val_losses = []
    
    logger.info("Bắt đầu huấn luyện...")
    for epoch in range(EPOCHS):
        # Forward propagation
        try:
            users_emb_final, users_emb_0, items_emb_final, items_emb_0 = \
                model.forward(feature_dict, train_edge_index, pos_train_edge_index, neg_train_edge_index)
            
            logger.info(f"Epoch {epoch}: Forward pass thành công") if epoch == 0 else None
            logger.info(f"users_emb_final shape: {users_emb_final.shape}") if epoch == 0 else None
            logger.info(f"items_emb_final shape: {items_emb_final.shape}") if epoch == 0 else None
        except Exception as e:
            logger.error(f"Lỗi khi thực hiện forward pass: {str(e)}")
            continue
        
        # Mini batching
        try:
            user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(BATCH_SIZE, train_edge_index)
            logger.info(f"Mini-batch: User indices shape: {user_indices.shape}") if epoch == 0 else None
            logger.info(f"Mini-batch: Positive item indices shape: {pos_item_indices.shape}") if epoch == 0 else None
            logger.info(f"Mini-batch: Negative item indices shape: {neg_item_indices.shape}") if epoch == 0 else None
        except Exception as e:
            logger.error(f"Lỗi khi lấy mini-batch: {str(e)}")
            continue
            
        users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
        pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
        neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]
        
        user_tags_batch = user_tags[user_indices]
        pos_item_tags_batch = food_tags[pos_item_indices]
        neg_item_tags_batch = food_tags[neg_item_indices]
        
        # Điền user features nếu cần
        user_features_batch = user_features[user_indices]
        pos_item_features_batch = food_features[pos_item_indices]
        neg_item_features_batch = food_features[neg_item_indices]
        
        # Ghi log kích thước để debug
        if epoch == 0:
            logger.info(f"user_features_batch shape: {user_features_batch.shape}")
            logger.info(f"pos_item_features_batch shape: {pos_item_features_batch.shape}")
            logger.info(f"neg_item_features_batch shape: {neg_item_features_batch.shape}")
            logger.info(f"user_tags_batch shape: {user_tags_batch.shape}")
            logger.info(f"pos_item_tags_batch shape: {pos_item_tags_batch.shape}")
            logger.info(f"neg_item_tags_batch shape: {neg_item_tags_batch.shape}")
        
        # Tính toán Pareto Loss
        try:
            train_loss, loss_data, _ = pareto_loss(
                model, users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, 
                neg_items_emb_final, neg_items_emb_0, user_features_batch, pos_item_features_batch, 
                neg_item_features_batch, user_tags_batch, pos_item_tags_batch, neg_item_tags_batch, LAMBDA
            )
            logger.info(f"Epoch {epoch}: Pareto loss thành công, loss = {train_loss.item()}")
        except Exception as e:
            logger.error(f"Lỗi khi tính pareto loss: {str(e)}")
            # Ghi thêm thông tin để debug
            import traceback
            logger.error(traceback.format_exc())
            continue
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        if epoch % ITERS_PER_EVAL == 0 and epoch != 0:
            model.eval()
            # Đánh giá trên tập validation
            try:
                val_loss, recall, precision, ndcg, health_score, avg_health_tags_ratio, percentage_recommended_foods = \
                    eval(model, feature_dict, user_tags, food_tags, val_edge_index, pos_val_edge_index, neg_val_edge_index,
                         [neg_train_edge_index], K, LAMBDA)
                
                logger.info(f"Epoch: {epoch}, "
                      f"train_loss: {round(train_loss.item(), 5)}, "
                      f"val_loss: {round(val_loss, 5)}, "
                      f"val_recall@{K}: {round(recall, 5)}, "
                      f"val_precision@{K}: {round(precision, 5)}, "
                      f"val_ndcg@{K}: {round(ndcg, 5)}, " 
                      f"val_health_score: {round(health_score, 5)}, "
                      f"avg_health_tags_ratio: {round(avg_health_tags_ratio, 5)}, "
                      f"percentage_recommended_foods: {round(percentage_recommended_foods, 5)}")
                
                train_losses.append(train_loss.item())
                val_losses.append(val_loss)
            except Exception as e:
                logger.error(f"Lỗi khi đánh giá mô hình: {str(e)}")
                
            model.train()
        
        if epoch % ITERS_PER_LR_DECAY == 0 and epoch != 0:
            scheduler.step()
    
    # Đánh giá cuối cùng trên tập test
    with torch.no_grad():
        model.eval()
        try:
            _, recall, precision, ndcg, health_score, avg_health_tags_ratio, percentage_recommended_foods = \
                eval(model, feature_dict, user_tags, food_tags, test_edge_index, pos_test_edge_index, neg_test_edge_index,
                     [neg_train_edge_index], K, LAMBDA)
            
            logger.info(f"Kết quả test: "
                  f"test_recall@{K}: {round(recall, 5)}, "
                  f"test_precision@{K}: {round(precision, 5)}, "
                  f"test_ndcg@{K}: {round(ndcg, 5)}, "
                  f"test_health_score: {round(health_score, 5)}, "
                  f"test_avg_health_tags_ratio: {round(avg_health_tags_ratio, 5)}, "
                  f"test_percentage_recommended_foods: {round(percentage_recommended_foods, 5)}")
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá mô hình trên tập test: {str(e)}")
    
    # Lưu mô hình đã huấn luyện
    torch.save(model.state_dict(), 'vn_trained_model.pth')
    logger.info("Mô hình đã được huấn luyện và lưu thành công!")
    
    return model

def fix_user_food_feature_compatibility():
    """
    Hàm này kiểm tra và sửa sự không tương thích về kích thước đặc trưng 
    giữa user và food trong đồ thị Vietnamese food
    
    Returns:
        bool: True nếu sửa thành công, False nếu không cần sửa
    """
    if not os.path.exists('vn_food_graph.pt'):
        logger.error("Không tìm thấy đồ thị Vietnamese food để sửa")
        return False
    
    graph = torch.load('vn_food_graph.pt')
    user_feature_shape = graph['user'].x.shape
    food_feature_shape = graph['food'].x.shape
    
    logger.info(f"Kiểm tra tính tương thích: user feature shape = {user_feature_shape}, food feature shape = {food_feature_shape}")
    
    # Kiểm tra tính tương thích của diversity_loss trong pareto_loss
    # Trong hàm diversity_loss, user_features_batch và pos_item_features_batch phải có cùng chiều
    if user_feature_shape[1] != food_feature_shape[1]:
        logger.info("Phát hiện không tương thích về kích thước đặc trưng. Đang sửa...")
        
        # Trường hợp 1: Chiều user features > chiều food features
        if user_feature_shape[1] > food_feature_shape[1]:
            # Padding thêm 0 cho food features
            new_food_features = torch.zeros((food_feature_shape[0], user_feature_shape[1]), dtype=graph['food'].x.dtype)
            new_food_features[:, :food_feature_shape[1]] = graph['food'].x
            graph['food'].x = new_food_features
            logger.info(f"Đã padding food features từ {food_feature_shape} thành {graph['food'].x.shape}")
        # Trường hợp 2: Chiều food features > chiều user features
        else:
            # Padding thêm 0 cho user features
            new_user_features = torch.zeros((user_feature_shape[0], food_feature_shape[1]), dtype=graph['user'].x.dtype)
            new_user_features[:, :user_feature_shape[1]] = graph['user'].x
            graph['user'].x = new_user_features
            logger.info(f"Đã padding user features từ {user_feature_shape} thành {graph['user'].x.shape}")
        
        # Lưu đồ thị đã sửa
        torch.save(graph, 'vn_food_graph.pt')
        logger.info("Đã lưu đồ thị với kích thước đặc trưng đã được điều chỉnh")
        return True
    else:
        logger.info("Kích thước đặc trưng đã tương thích, không cần sửa")
        return False

def examine_pareto_loss(graph):
    """
    Phân tích chi tiết hàm pareto_loss để tìm ra vấn đề
    
    Args:
        graph (HeteroData): Đồ thị cần kiểm tra
    """
    logger.info("Đang phân tích hàm pareto_loss...")
    
    # Giả lập các đầu vào của hàm pareto_loss để tìm lỗi
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Tạo model mẫu
    model = SGSL(graph, embedding_dim=HIDDEN_DIM, feature_threshold=FEATURE_THRESHOLD, num_layer=LAYERS)
    model = model.to(device)
    
    # Lấy một mẫu nhỏ dữ liệu
    num_samples = 8  # Số lượng mẫu nhỏ để test
    
    users_emb_final = torch.randn(num_samples, HIDDEN_DIM).to(device)
    users_emb_0 = torch.randn(num_samples, HIDDEN_DIM).to(device)
    pos_items_emb_final = torch.randn(num_samples, HIDDEN_DIM).to(device)
    pos_items_emb_0 = torch.randn(num_samples, HIDDEN_DIM).to(device)
    neg_items_emb_final = torch.randn(num_samples, HIDDEN_DIM).to(device)
    neg_items_emb_0 = torch.randn(num_samples, HIDDEN_DIM).to(device)
    
    # Lấy kích thước đặc trưng thực tế
    user_feature_dim = graph['user'].x.shape[1]
    food_feature_dim = graph['food'].x.shape[1]
    
    logger.info(f"Kích thước đặc trưng user: {user_feature_dim}")
    logger.info(f"Kích thước đặc trưng food: {food_feature_dim}")
    
    # Tạo features batch mẫu
    user_features_batch = torch.randn(num_samples, user_feature_dim).to(device)
    pos_item_features_batch = torch.randn(num_samples, food_feature_dim).to(device)
    neg_item_features_batch = torch.randn(num_samples, food_feature_dim).to(device)
    
    # Lấy kích thước tags thực tế
    user_tag_dim = graph['user'].tags.shape[1]
    food_tag_dim = graph['food'].tags.shape[1]
    
    logger.info(f"Kích thước tags user: {user_tag_dim}")
    logger.info(f"Kích thước tags food: {food_tag_dim}")
    
    # Tạo tags batch mẫu
    user_tags_batch = torch.randint(0, 2, (num_samples, user_tag_dim), dtype=torch.float).to(device)
    pos_item_tags_batch = torch.randint(0, 2, (num_samples, food_tag_dim), dtype=torch.float).to(device)
    neg_item_tags_batch = torch.randint(0, 2, (num_samples, food_tag_dim), dtype=torch.float).to(device)
    
    logger.info("Đang thử nghiệm các thành phần của hàm pareto_loss...")
    
    # Kiểm tra từng thành phần của pareto_loss
    from RCSYS_utils import bpr_loss, diversity_loss, health_loss
    
    try:
        # Thử bpr_loss
        bpr_result = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, LAMBDA)
        logger.info(f"bpr_loss thành công, giá trị: {bpr_result.item()}")
    except Exception as e:
        logger.error(f"Lỗi trong bpr_loss: {str(e)}")
    
    try:
        # Thử health_loss
        health_result = health_loss(users_emb_final, pos_items_emb_final, neg_items_emb_final, user_tags_batch, pos_item_tags_batch, neg_item_tags_batch)
        logger.info(f"health_loss thành công, giá trị: {health_result.item()}")
    except Exception as e:
        logger.error(f"Lỗi trong health_loss: {str(e)}")
    
    try:
        # Thử diversity_loss - đây là nơi nhiều khả năng xảy ra lỗi
        # Trước tiên, đảm bảo kích thước của user_features_batch và pos_item_features_batch giống nhau
        if user_features_batch.shape[1] != pos_item_features_batch.shape[1]:
            logger.warning(f"Phát hiện không tương thích kích thước: user_features_batch {user_features_batch.shape}, pos_item_features_batch {pos_item_features_batch.shape}")
            
            # Điều chỉnh kích thước nếu cần
            if user_features_batch.shape[1] < pos_item_features_batch.shape[1]:
                user_features_batch = torch.nn.functional.pad(
                    user_features_batch, (0, pos_item_features_batch.shape[1] - user_features_batch.shape[1])
                )
                logger.info(f"Đã điều chỉnh user_features_batch thành {user_features_batch.shape}")
            else:
                pos_item_features_batch = torch.nn.functional.pad(
                    pos_item_features_batch, (0, user_features_batch.shape[1] - pos_item_features_batch.shape[1])
                )
                neg_item_features_batch = torch.nn.functional.pad(
                    neg_item_features_batch, (0, user_features_batch.shape[1] - neg_item_features_batch.shape[1])
                )
                logger.info(f"Đã điều chỉnh item_features_batch thành {pos_item_features_batch.shape}")
        
        # Thử lại diversity_loss sau khi điều chỉnh
        diversity_result = diversity_loss(users_emb_final, pos_items_emb_final, neg_items_emb_final, user_features_batch, pos_item_features_batch, neg_item_features_batch)
        logger.info(f"diversity_loss thành công, giá trị: {diversity_result.item()}")
    except Exception as e:
        logger.error(f"Lỗi trong diversity_loss: {str(e)}")
        
    logger.info("Phân tích hàm pareto_loss hoàn tất")

def recommend_dishes(user_id, top_k=10):
    """
    Đề xuất món ăn Việt Nam cho người dùng
    
    Args:
        user_id (int): ID của người dùng
        top_k (int): Số lượng món ăn đề xuất
        
    Returns:
        list: Danh sách các món ăn được đề xuất
    """
    # Tải đồ thị và mô hình
    if not os.path.exists('vn_food_graph.pt'):
        logger.error("Không tìm thấy đồ thị Vietnamese food")
        return []
    
    if not os.path.exists('vn_trained_model.pth'):
        logger.error("Không tìm thấy mô hình đã huấn luyện")
        return []
    
    graph = torch.load('vn_food_graph.pt')
    model = SGSL(graph, embedding_dim=HIDDEN_DIM, feature_threshold=FEATURE_THRESHOLD, num_layer=LAYERS)
    model.load_state_dict(torch.load('vn_trained_model.pth'))
    model.eval()
    
    # Chuyển sang device
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
        return []
    
    # Chuẩn bị dữ liệu
    feature_dict = {'user': graph['user'].x.to(device), 'food': graph['food'].x.to(device)}
    edge_index = graph[('user', 'eats', 'food')].edge_index.to(device)
    edge_label_index = graph[('user', 'eats', 'food')].edge_label_index.to(device)
    
    # Forward pass qua mô hình
    users_emb_final, _, items_emb_final, _ = model.forward(feature_dict, edge_index, edge_label_index, edge_label_index)
    
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
    indices = indices.cpu().numpy()
    
    # Chuyển indices thành tên món ăn
    df_food = pd.read_csv('food_tagging.csv')
    food_names = df_food['Tên món ăn'].iloc[indices].tolist()
    
    return food_names

def modify_pareto_loss_for_compatibility():
    """
    Tạo một phiên bản mới của hàm pareto_loss để đảm bảo tính tương thích
    giữa kích thước của user_features và food_features
    
    Lưu vào file RCSYS_utils.py
    """
    logger.info("Tạo phiên bản tương thích của hàm pareto_loss")
    
    # Mã nguồn của hàm diversity_loss sửa đổi
    diversity_loss_code = """
def diversity_loss(users_emb_final, pos_items_emb_final, neg_items_emb_final, user_features_batch, pos_item_features_batch, neg_item_features_batch, k=20):
    def get_top_k_recommendations(user_emb, item_emb, k=10):
        scores = torch.matmul(user_emb, item_emb.T)
        _, top_k_indices = torch.topk(scores, k=k, dim=1)
        return top_k_indices
    
    def get_mean_similarity(user_features_batch, item_features_batch, k):
        # Đảm bảo user_features_batch và item_features_batch có cùng kích thước chiều cuối
        feature_dim = min(user_features_batch.shape[1], item_features_batch.shape[1])
        user_features = user_features_batch[:, :feature_dim]
        item_features = item_features_batch[:, :feature_dim]
        
        # Get the top K item indices for each user
        top_k_indices = get_top_k_recommendations(user_features, item_features, k)
        top_k_item_embs = item_features[top_k_indices]

        # Calculate the cosine similarities for all pairs in the top K items
        similarities = cosine_similarity(
            top_k_item_embs.unsqueeze(2),  # Shape: (num_users, k, 1, embedding_dim)
            top_k_item_embs.unsqueeze(1),  # Shape: (num_users, 1, k, embedding_dim)
            dim=3
        )

        # Select the upper triangular part of the similarity matrix, excluding the diagonal
        upper_triangular_indices = torch.triu_indices(k, k, 1)
        selected_similarities = similarities[:, upper_triangular_indices[0], upper_triangular_indices[1]]

        # Calculate the mean similarity for each user
        return selected_similarities.mean(dim=1)

    # Đảm bảo kích thước tương thích trước khi gọi get_mean_similarity
    feature_dim = min(user_features_batch.shape[1], pos_item_features_batch.shape[1], neg_item_features_batch.shape[1])
    user_features = user_features_batch[:, :feature_dim]
    pos_item_features = pos_item_features_batch[:, :feature_dim]
    neg_item_features = neg_item_features_batch[:, :feature_dim]
    
    pos_similarity = get_mean_similarity(user_features, pos_item_features, k)
    neg_similarity = get_mean_similarity(user_features, neg_item_features, k)

    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1)  # predicted scores of positive samples
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1)  # predicted scores of negative samples

    # Calculate and return the diversity loss
    loss = -torch.mean(torch.log((torch.sigmoid(torch.mul(pos_similarity - neg_similarity, pos_scores - neg_scores)))))
    return loss
"""
    
    # Đường dẫn đến file RCSYS_utils.py
    utils_path = "RCSYS_utils.py"
    
    if not os.path.exists(utils_path):
        logger.error(f"Không tìm thấy file {utils_path} để sửa đổi")
        return False
    
    try:
        # Đọc nội dung file
        with open(utils_path, 'r') as f:
            content = f.read()
        
        # Tìm và thay thế hàm diversity_loss
        import re
        pattern = r"def diversity_loss\([^)]*\):.*?(?=def [a-zA-Z_])"
        replacement = diversity_loss_code
        
        # Sử dụng re.DOTALL để tìm kiếm trên nhiều dòng
        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        # Nếu không tìm thấy, thử mẫu khác (trường hợp diversity_loss là hàm cuối cùng trong file)
        if new_content == content:
            pattern = r"def diversity_loss\([^)]*\):.*"
            new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        # Lưu file mới
        with open(utils_path + '.backup', 'w') as f:
            f.write(content)  # Lưu bản sao lưu
            
        with open(utils_path, 'w') as f:
            f.write(new_content)
            
        logger.info(f"Đã sửa đổi hàm diversity_loss trong {utils_path}")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi sửa đổi file: {str(e)}")
        return False

if __name__ == "__main__":
    # Bước 1: Kiểm tra và sửa đồ thị hiện có nếu cần
    if os.path.exists('vn_food_graph.pt'):
        logger.info("Đang kiểm tra và sửa đồ thị Vietnamese food hiện có...")
        fixed = fix_user_food_feature_compatibility()
        if fixed:
            logger.info("Đã sửa đồ thị thành công")
        else:
            logger.info("Đồ thị không cần sửa hoặc không thể sửa")
            
        # Phân tích chi tiết hàm pareto_loss
        graph = torch.load('vn_food_graph.pt')
        examine_pareto_loss(graph)
        
        # Sửa đổi hàm pareto_loss
        modify_pareto_loss_for_compatibility()
    
    # Bước 2: Tạo đồ thị mới nếu chưa tồn tại hoặc muốn tạo lại
    create_new = True
    if os.path.exists('vn_food_graph.pt') and not create_new:
        logger.info("Đồ thị thức ăn Việt Nam đã tồn tại, bỏ qua bước tạo đồ thị.")
    else:
        logger.info("Đang tạo đồ thị mới...")
        graph = create_vietnamese_food_graph()
    
    # Bước 3: Huấn luyện mô hình
    train_model = True
    if train_model:
        logger.info("Bắt đầu huấn luyện mô hình...")
        model = train_vietnamese_model()
    
    # Bước 4: Kiểm thử đề xuất
    user_ids = [21005, 21015, 21020]  # Chọn một số user ID bất kỳ từ dữ liệu gốc
    
    for user_id in user_ids:
        recommendations = recommend_dishes(user_id, top_k=10)
        logger.info(f"Đề xuất cho người dùng {user_id}:")
        for i, dish in enumerate(recommendations, 1):
            logger.info(f"{i}. {dish}")