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
import gc
from sklearn.model_selection import StratifiedKFold
import numpy as np
# In[]

parser = argparse.ArgumentParser(description='Main Code for Graph Classification')
parser.add_argument('--train_model', help="To train the model", action='store_true')

parser.add_argument('--gen_plot', help="To generate plot", action='store_true')
parser.add_argument('--fold', type=int ,help="pass the fold number in range [1,10]")

args = parser.parse_args()

data_dir="kfold_500"
test_dir="kfold_real"
# In[]
if args.gen_plot:
    fold=args.fold
    with open(f"{data_dir}/all_losses.json","r") as f:
        all_losses=json.load(f)
    
    loss=all_losses[fold-1]
    plt.plot([i+1 for i in range(len(loss))],loss)
    plt.title(f"Loss Plot for fold {fold}")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig(f"{data_dir}/fold_{fold}_plot.png")
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
    # g= bz2.BZ2File(f"{data_dir}/graph_dataset/graphs_data.pbz2", 'rb')
    # dataset=cPickle.load(g)
    dataset=[]

    g1 = bz2.BZ2File(f"{data_dir}/graph_dataset/graphs_data_1000.pbz2", "rb")
    
    temp = cPickle.load(g1)
    dataset+=temp
    del temp
    gc.collect()

    g2 = bz2.BZ2File(f"{data_dir}/graph_dataset/graphs_data_2000.pbz2", "rb")

    temp = cPickle.load(g2)
    dataset+=temp
    del temp
    gc.collect()

    g3 = bz2.BZ2File(f"{data_dir}/graph_dataset/graphs_data_3000.pbz2", "rb")
    
    temp = cPickle.load(g3)
    dataset+=temp
    del temp
    gc.collect()

    g4 = bz2.BZ2File(f"{data_dir}/graph_dataset/graphs_data_4000.pbz2", "rb")

    temp = cPickle.load(g4)
    dataset+=temp
    del temp
    gc.collect()

    g5 = bz2.BZ2File(f"{data_dir}/graph_dataset/graphs_data_5000.pbz2", "rb")
    
    temp = cPickle.load(g5)
    dataset+=temp
    del temp
    gc.collect()

    g6 = bz2.BZ2File(f"{data_dir}/graph_dataset/graphs_data_6000.pbz2", "rb")

    temp = cPickle.load(g6)
    dataset+=temp
    del temp
    gc.collect()
    
    # dataset2 = cPickle.load(g2)
    # dataset3 = cPickle.load(g3)
    # dataset4 = cPickle.load(g4)
    # dataset5 = cPickle.load(g5)
    # dataset6 = cPickle.load(g6)
    # dataset = dataset1 + dataset2 + dataset3 + dataset4 + dataset5 + dataset6
    # random.shuffle(dataset)

    g_test = bz2.BZ2File(f"{test_dir}/graph_dataset/graphs_data.pbz2", "rb")
    dataset_test = cPickle.load(g_test)

    num_test = len(dataset_test)
    print(num_test)

    test_sampler = SubsetRandomSampler(torch.arange(num_test))
    test_dataloader = GraphDataLoader(
        dataset_test, sampler=test_sampler, batch_size=30, drop_last=False
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

    graphs=np.array([i for i in range(6000)])
    labels=np.array([dataset[i][1] for i in range(6000)])

    skf=StratifiedKFold(n_splits=10)

    train_fold=[]
    test_fold=[]

    for train_index, test_index in skf.split(graphs, labels):
        X_train, X_test = graphs[train_index], graphs[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        train_fold.append([(x,dataset[x][1]) for x in X_train])
        test_fold.append([dataset[x] for x in X_test])

    for i in range(folds):
        train_data=train_fold[i]
        val_data=test_fold[i]

        train_indices1 = torch.arange(0, 1800)
        train_indices2 = torch.arange(1800, 3600)
        train_indices3 = torch.arange(3600, 5400)

        val_indices = torch.arange(0, len(val_data))

        train_sampler1 = SubsetRandomSampler(train_indices1)
        train_sampler2 = SubsetRandomSampler(train_indices2)
        train_sampler3 = SubsetRandomSampler(train_indices3)

        val_sampler = SubsetRandomSampler(val_indices)

        train_dataloader1 = torch.utils.data.DataLoader(
            train_data, sampler=train_sampler1, batch_size=10, drop_last=False
        )
        train_dataloader2 = torch.utils.data.DataLoader(
            train_data, sampler=train_sampler2, batch_size=10, drop_last=False
        )
        train_dataloader3 = torch.utils.data.DataLoader(
            train_data, sampler=train_sampler3, batch_size=10, drop_last=False
        )

        val_dataloader = GraphDataLoader(
            val_data, sampler=val_sampler, batch_size=30, drop_last=False
        )

        loss_fold=[]
        for epoch in range(100):
            l = 0.0
            cnt = 0
            for l1,l2,l3 in zip(train_dataloader1,train_dataloader2,train_dataloader3):
                graph_index=torch.cat([l1[0],l2[0],l3[0]])
                labels=torch.cat([l1[1],l2[1],l3[1]])
                
                temp=list(zip(graph_index,labels))
                random.shuffle(temp)
                graph_index,labels=zip(*temp)

                batched_graph=dgl.batch([dataset[x][0] for x in graph_index])

                pred = model(batched_graph, batched_graph.ndata["h"].float())
                loss = F.cross_entropy(pred, labels)
                l += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cnt += 1
            loss_fold.append(float(loss))
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

