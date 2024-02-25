import argparse
from loader import *
import torch
from torch.optim.lr_scheduler import CyclicLR
from model import DRDM
from sklearn import metrics
import numpy as np
from collections import OrderedDict

sys.path.append(".")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="Run DRDM.")
    parser.add_argument('--dataset', nargs='?', default='Fdataset', help='Choose a dataset. [Fdataset/Cdataset/LRSSL]')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=1024*5, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.01, help='init Learning rate.')
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--disease_TopK', type=int, default=4)
    parser.add_argument('--drug_TopK', type=int, default=4)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument('--exp_coff', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=0.6, help='the coefficient of feature fusion ')
    parser.add_argument('--wr', type=float, default=0.6, help='the coefficient of feature fusion ')
    return parser.parse_args()

def config_model():
    config = OrderedDict()
    config['dataset'] = args.dataset
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['lr'] = args.lr
    config['embedding_size'] = args.embedding_size
    config['n_layers'] = args.n_layers
    config['weight_decay'] = args.weight_decay
    config["disease_TopK"] = args.disease_TopK
    config['drug_TopK'] = args.drug_TopK
    config['seed'] = args.seed
    config['n_splits'] = args.n_splits
    config['device'] = device
    config['exp_coff'] = args.exp_coff
    config['wd'] = args.wd
    config['wr'] = args.wr
    return config

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

args = parse_args()
config = config_model()

if __name__ == "__main__":
    avg_auroc, avg_aupr = [], []
    for i in range(args.num_trials):
        setup_seed(i)
        disease_adj, drug_adj, original_interactions, all_train_mask, all_test_mask, pos_weight = data_preparation(args)
        all_scores, all_labels = [], []
        print(f'+++++++++++++++This is {i + 1}-th 10 fold validation.+++++++++++++++')
        for fold_num in range(len(all_train_mask)):
            print(f'---------------This is {fold_num + 1}-th fold validation.---------------')
            # dataset splitting
            train_manager, test_manager = data_split(config, all_train_mask[fold_num], all_test_mask[fold_num],
                                                     original_interactions)
            train_adj = train_manager.train_adj
            # model loading and initialization
            model = DRDM(config, (train_manager, train_adj, disease_adj, drug_adj, pos_weight)).to(device)
            # training
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            lr_scheduler = CyclicLR(optimizer,
                                    base_lr=0.1 * args.lr,
                                    max_lr=args.lr,
                                    step_size_up=20,
                                    mode="exp_range",
                                    gamma=0.995,
                                    cycle_momentum=False)
            for epoch in range(args.epochs):
                model.train()
                loss_list = []
                for batch in train_manager.iter_batch(shuffle=True):
                    loss, scores = model.forward(batch, True)
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                model.eval()
                scores, labels = [], []
                for batch in test_manager.iter_batch():
                    score, label = model.predict(batch)
                    scores.append(score.cpu().detach().numpy())
                    labels.append(label)
                loss_sum = np.sum(loss_list)
                scores = np.concatenate(scores)
                labels = np.concatenate(labels)
                aupr = metrics.average_precision_score(y_true=labels, y_score=scores)
                auroc = metrics.roc_auc_score(y_true=labels, y_score=scores)
                print(f'Epoch: {epoch + 1}, auroc: {auroc}, aupr: {aupr}')
                if (epoch + 1) == args.epochs:
                    all_scores.append(scores)
                    all_labels.append(labels)
        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        aupr = metrics.average_precision_score(y_true=all_labels, y_score=all_scores)
        auroc = metrics.roc_auc_score(y_true=all_labels, y_score=all_scores)
        avg_auroc.append(auroc)
        avg_aupr.append(aupr)
        print(f'------------------------------------------------------------------------')
        print(f"{i + 1}-th 10 cv auroc：{auroc:.5f}")
        print(f"{i + 1}-th 10 cv auroc：{aupr:.5f}")
    print(f'------------------------------------------------------------------------')
    print(f"auroc：{np.mean(avg_auroc):.5f}, std：{np.std(avg_auroc):.5f}")
    print(f"aupr：{np.mean(avg_aupr):.5f}, std：{np.std(avg_aupr):.5f}")