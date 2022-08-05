# In[38]:


import dgl
import torch
import torch.nn as nn
import json
import _pickle as cPickle
import torch.nn.functional as F
import bz2
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.nn import GraphConv
from sklearn.metrics import classification_report
import random
import os
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Main Code for Graph Classification')
parser.add_argument('--train_model', help="To train the model", action='store_true')

parser.add_argument('--gen_plot', help="To generate plot", action='store_true')
parser.add_argument('--fold', type=int ,help="pass the fold number in range [1,10]")

args = parser.parse_args()

# In[]
if args.gen_plot:
    fold=args.fold
    with open("kfold_500/all_losses.json","r") as f:
        all_losses=json.load(f)
    
    loss=all_losses[fold-1]
    plt.plot([i+1 for i in range(len(loss))],loss)
    plt.title(f"Loss Plot for fold {fold}")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig(f"kfold/fold_{fold}_plot.png")
    plt.show()

# In[]
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, out_feats)

    def forward(self, g, in_feat):
        # print("In GCN")
        h = F.relu(self.conv1(g, in_feat))
        h = F.relu(self.conv2(g, h))
        with g.local_scope():
            g.ndata["h"] = h
            return dgl.mean_nodes(g, "h")

class MLP(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_feats, h_feats)
        self.fc2 = nn.Linear(h_feats, num_classes)

    def forward(self, x):
        # print("In MLP")
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class Model(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, num_class):
        super(Model, self).__init__()
        self.conv = GCN(in_feats, h_feats, out_feats)
        self.fc = MLP(out_feats, 10, num_class)

    def forward(self, g, feats):
        # print("In Model")
        c1 = self.conv(g, feats)
        c2 = self.fc(c1)
        return c2

if args.train_model:
    data_dir="kfold_real"
    test_dir="kfold_real"
    g= bz2.BZ2File(f"{data_dir}/graph_dataset/graphs_data.pbz2", 'rb')
    dataset=cPickle.load(g)

    # g1 = bz2.BZ2File(f"{data_dir}/graph_dataset/graphs_data_1000.pbz2", "rb")
    # g2 = bz2.BZ2File(f"{data_dir}/graph_dataset/graphs_data_2000.pbz2", "rb")
    # g4 = bz2.BZ2File(f"{data_dir}/graph_dataset/graphs_data_4000.pbz2", "rb")
    # g5 = bz2.BZ2File(f"{data_dir}/graph_dataset/graphs_data_5000.pbz2", "rb")
    # g3 = bz2.BZ2File(f"{data_dir}/graph_dataset/graphs_data_3000.pbz2", "rb")
    # g6 = bz2.BZ2File(f"{data_dir}/graph_dataset/graphs_data_6000.pbz2", "rb")
    # dataset1 = cPickle.load(g1)
    # dataset2 = cPickle.load(g2)
    # dataset3 = cPickle.load(g3)
    # dataset4 = cPickle.load(g4)
    # dataset5 = cPickle.load(g5)
    # dataset6 = cPickle.load(g6)
    # dataset = dataset1 + dataset2 + dataset3 + dataset4 + dataset5 + dataset6
    random.shuffle(dataset)

    g_test = bz2.BZ2File(f"{test_dir}/graph_dataset/graphs_data.pbz2", "rb")
    dataset_test = cPickle.load(g_test)

    num_test = len(dataset_test)
    print(num_test)

    test_sampler = SubsetRandomSampler(torch.arange(num_test))
    test_dataloader = GraphDataLoader(
        dataset_test, sampler=test_sampler, batch_size=5, drop_last=False
    )

    # Create the model with given dimensions
    model = Model(5, 20, 10, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if not os.path.exists(f"{data_dir}/trained_model/"):
        os.makedirs(f"{data_dir}/trained_model/")

    all_losses=[]

    folds = 10
    num_examples = len(dataset)
    fsize = num_examples // 10
    val_reports = []
    for i in range(folds):
        trll = 0
        trlr = i * fsize
        vall = trlr
        valr = i * fsize + fsize
        trrl = valr
        trrr = num_examples

        train_left_indices = torch.arange(trll, trlr)
        train_right_indices = torch.arange(trrl, trrr)

        train_indices = torch.cat([train_left_indices, train_right_indices])
        val_indices = torch.arange(vall, valr)

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_dataloader = GraphDataLoader(
            dataset, sampler=train_sampler, batch_size=5, drop_last=False
        )
        val_dataloader = GraphDataLoader(
            dataset, sampler=val_sampler, batch_size=5, drop_last=False
        )
        loss_fold=[]
        for epoch in range(100):
            l = 0.0
            cnt = 0
            for batched_graph, labels in train_dataloader:
                pred = model(batched_graph, batched_graph.ndata["h"].float())
                loss = F.cross_entropy(pred, labels)
                l += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cnt += 1
            loss_fold.append(loss)
            if epoch % 10 == 0:
                print(f"epoch : {epoch+1} loss : {l/cnt}")
        all_losses.append(loss_fold)
        with torch.no_grad():

            num_correct = 0
            num_tests = 0
            pred_labels = []
            correct_labels = []

            for batched_graph, labels in val_dataloader:
                pred = model(batched_graph, batched_graph.ndata["h"].float())
                num_correct += (pred.argmax(1) == labels).sum().item()
                pred_labels += list(pred.argmax(1))
                correct_labels += list(labels)

                num_tests += len(labels)
            print("Val accuracy:", num_correct / num_tests)
            val_reports.append(
                classification_report(
                    correct_labels, pred_labels, target_names=["low", "medium", "high"]
                )
            )
        torch.save(model.state_dict, f"{data_dir}/trained_model/classification_{i}.model")

    with open(f"{data_dir}/all_losses.json","w") as f:
        json.dump(all_losses,f)

    with open(f"{data_dir}/result_val.json", "w") as f:
        json.dump(val_reports, f)

    num_correct = 0
    num_tests = 0
    pred_labels = []
    correct_labels = []

    for batched_graph, labels in test_dataloader:
        pred = model(batched_graph, batched_graph.ndata["h"].float())
        num_correct += (pred.argmax(1) == labels).sum().item()
        pred_labels += list(pred.argmax(1))
        correct_labels += list(labels)

        num_tests += len(labels)
    print("Test accuracy:", num_correct / num_tests)

    test_report = classification_report(
        correct_labels, pred_labels, target_names=["low", "medium", "high"]
    )

    with open(f"{data_dir}/result_test.json", "w") as f:
        json.dump(test_report, f)

    num_examples = len(dataset)
    num_train = int(num_examples)

    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    train_dataloader = GraphDataLoader(
        dataset, sampler=train_sampler, batch_size=5, drop_last=False
    )

    num_correct = 0
    num_tests = 0
    pred_labels = []
    correct_labels = []
    with torch.no_grad():
        for batched_graph, labels in train_dataloader:
            pred = model(batched_graph, batched_graph.ndata["h"].float())
            num_correct += (pred.argmax(1) == labels).sum().item()
            pred_labels += list(pred.argmax(1))
            correct_labels += list(labels)
            num_tests += len(labels)
        print("Train accuracy:", num_correct / num_tests)

    train_report = classification_report(
        correct_labels, pred_labels, target_names=["low", "medium", "high"]
    )

    with open(f"{data_dir}/result_train.json", "w") as f:
        json.dump(train_report, f)

