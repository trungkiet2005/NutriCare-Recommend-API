from RCSYS_utils import *
import argparse
from RCSYS_models import *
from utils import *
import wandb
import os
os.environ["WANDB_SILENT"] = "true"  # Add this line


def main():
    if args.use_wandb:
        run = wandb.init()
        config = wandb.config
        run.name = f"Run_with_{config.seed}_{config.lr}_{config.hidden_dim}_{config.layers}_{config.batch_size}_{config.feature_threshold}"
        run.save()
        SEED = config.seed
        BATCH_SIZE = config.batch_size
        LAMBDA = config.LAMBDA
        HIDDEN_DIM = config.hidden_dim
        LAYERS = config.layers
        LR = config.lr
        TH = config.feature_threshold

    else:        
        SEED = args.seed
        BATCH_SIZE = args.batch_size
        LAMBDA = args.LAMBDA
        HIDDEN_DIM = args.hidden_dim
        LAYERS = args.layers
        LR = args.lr
        TH = args.feature_threshold

    # set_seed(SEED)
    torch_geometric.seed_everything(SEED)
    # Data loading & Preprocessing
    graph = torch.load('../processed_data/benchmark_macro.pt')
    num_users, num_foods = graph['user'].num_nodes, graph['food'].num_nodes
    edge_index = graph[('user', 'eats', 'food')].edge_index
    edge_label_index = graph[('user', 'eats', 'food')].edge_label_index
    feature_dict = graph.x_dict

    train_edge_index, val_edge_index, test_edge_index, \
    pos_train_edge_index, neg_train_edge_index, pos_val_edge_index, neg_val_edge_index, \
    pos_test_edge_index, neg_test_edge_index = split_data_new(edge_index, edge_label_index)

    model = SGSL(graph, embedding_dim=HIDDEN_DIM, feature_threshold=TH, num_layer=LAYERS)

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    # Extract tag features from the graph
    user_tags = graph['user'].tags.to(device)
    food_tags = graph['food'].tags.to(device)
    user_features = graph['user'].x.to(device)
    food_features = graph['food'].x.to(device) 

    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # training loop
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        # forward propagation
        users_emb_final, users_emb_0, items_emb_final, items_emb_0 = \
            model.forward(feature_dict, train_edge_index, pos_train_edge_index, neg_train_edge_index)
        
        # mini batching
        user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(BATCH_SIZE, train_edge_index)
        users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
        pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
        neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]

        user_tags_batch = user_tags[user_indices]
        pos_item_tags_batch = food_tags[pos_item_indices]
        neg_item_tags_batch = food_tags[neg_item_indices]

        # Pad user features
        user_features_batch = user_features[user_indices]
        user_features_batch = torch.nn.functional.pad(user_features_batch, (0, food_features.size(1) - user_features_batch.size(1)))

        pos_item_features_batch = food_features[pos_item_indices]
        neg_item_features_batch = food_features[neg_item_indices]

        ### Pareto Loss here ### 
        train_loss, loss_data, _ = pareto_loss(model, users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, 
                        user_features_batch, pos_item_features_batch, neg_item_features_batch, 
                        user_tags_batch, pos_item_tags_batch, neg_item_tags_batch, LAMBDA)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # print(f"Epoch: {epoch}, train_loss: {round(train_loss.item(), 5)}, bpr_loss: {round(loss_data['bpr'].item(), 5)}, " 
        #       f"sim_loss: {round(loss_data['sim'].item(), 5)}, health_loss: {round(loss_data['health'].item(), 5)}")
        # if args.use_wandb:
        #         wandb.log({
        #             'train_loss': round(train_loss.item(), 5),
        #             'bpr_loss': round(loss_data['bpr'].item(), 5),
        #             'diversity_loss': round(loss_data['sim'].item(), 5),
        #             'health_loss': round(loss_data['health'].item(), 5)
        #         })

        if epoch % args.iters_per_eval == 0 and epoch != 0:
            model.eval()
            val_loss, recall, precision, ndcg, health_score, avg_health_tags_ratio, percentage_recommended_foods = \
                eval(model, feature_dict, user_tags, food_tags, val_edge_index, pos_val_edge_index, neg_val_edge_index,
                                [neg_train_edge_index], args.K, LAMBDA)
            
            print(f"Epoch: {epoch}, "
                  f"train_loss: {round(train_loss.item(), 5)}, "
                  f"val_loss: {round(val_loss, 5)}, "
                  f"val_recall@{args.K}: {round(recall, 5)}, "
                  f"val_precision@{args.K}: {round(precision, 5)}, "
                  f"val_ndcg@{args.K}: {round(ndcg, 5)}", 
                  f"val_health_score: {round(health_score, 5)}, "
                  f"avg_health_tags_ratio: {round(avg_health_tags_ratio, 5)}, "
                  f"percentage_recommended_foods: {round(percentage_recommended_foods, 5)}")

            if args.use_wandb:
                wandb.log({
                    # 'train_loss': round(train_loss.item(), 5),
                    # 'val_loss': round(val_loss, 5),
                    'val_recall': round(recall, 5),
                    'val_precision': round(precision, 5),
                    'val_ndcg': round(ndcg, 5),
                    'val_health_score': round(health_score, 5),
                    'avg_health_tags_ratio': round(avg_health_tags_ratio, 5),
                    'percentage_recommended_foods': round(percentage_recommended_foods, 5)
                    
                })

            train_losses.append(train_loss.item())
            val_losses.append(val_loss)
            model.train()

        if epoch % args.iters_per_lr_decay == 0 and epoch != 0:
            scheduler.step()

    with torch.no_grad():
        model.eval()
        _, recall, precision, ndcg, health_score, avg_health_tags_ratio, percentage_recommended_foods = \
                eval(model, feature_dict, user_tags, food_tags, test_edge_index, pos_test_edge_index, neg_test_edge_index,
                                [neg_train_edge_index], args.K, LAMBDA)
        
        print(f"test_recall@{args.K}: {round(recall, 5)}, "
              f"test_precision@{args.K}: {round(precision, 5)}, "
              f"test_ndcg@{args.K}: {round(ndcg, 5)}, "
              f"test_health_score: {round(health_score, 5)}, "
              f"avg_health_tags_ratio: {round(avg_health_tags_ratio, 5)}, "
              f"percentage_recommended_foods: {round(percentage_recommended_foods, 5)}")

        if args.use_wandb:
            wandb.log({
                'test_recall': round(recall, 5),
                'test_precision': round(precision, 5),
                'test_ndcg': round(ndcg, 5),
                'test_health_score': round(health_score, 5),
                'test_avg_health_tags_ratio': round(avg_health_tags_ratio, 5),
                'test_percentage_recommended_foods': round(percentage_recommended_foods, 5)
            })

    # Add this line to save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Number of hidden dimension.')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size.')
    parser.add_argument('--K', type=int, default=20,
                        help='Number of ranking list.')
    parser.add_argument('--LAMBDA', type=float, default=1e-6,
                        help='Regularization coefficient.')
    parser.add_argument('--iters_per_eval', type=int, default=500,
                        help='Iterations per evaluation.')
    parser.add_argument('--iters_per_lr_decay', type=int, default=200,
                        help='Iterations per learning rate decay.')
    parser.add_argument('--layers', type=int, default=3,
                        help='Number of layers in the model.')
    parser.add_argument('--use_wandb', type=bool, default=False,
                        help='Whether to use wandb for logging.')
    parser.add_argument('--feature_threshold', type=float, default=0.3,
                        help='Threshold for feature selection.')
    args = parser.parse_args()

    if args.use_wandb:
        wandb.login(key='')  # This will no longer prompt for choices
        sweep_config = {
            'name': 'sweep-try-RCSYS',
            'method': 'grid',
            'parameters': {
                'seed': {'values': [42]},
                'layers': {'values': [3]},
                'hidden_dim': {'values': [128]},
                'lr': {'values': [1e-3]},
                'LAMBDA': {'values': [1e-6]},
                'batch_size': {'values': [2048]},
                'feature_threshold': {'values': [0.3]}
            }
        }

        sweep_id = wandb.sweep(sweep_config, entity='', project='')
        wandb.agent(sweep_id, function=main)
    else:
        main()
