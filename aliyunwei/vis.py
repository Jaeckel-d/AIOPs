import collections
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from aiop_dataset import AIOPSTest, AIOPSVal, AIOPS
import torch
from sklearn.cluster import KMeans
import pandas as pd

if __name__ == '__main__':

    # load data and preprocess
    test_set = AIOPSTest(root_path="./tmp_data", split="test", base=[0, 1, 2], novel=[3])
    data = test_set.data
    labels = test_set.label
    msg, mask1, venus, mask2, server_model, dump = [], [], [], [], [], []
    for q, w, e, r, t, y in data:
        msg.append(q)
        mask1.append(w)
        venus.append(e)
        mask2.append(r)
        server_model.append(t)
        dump.append(y)
    msg = torch.stack(msg)
    msg = torch.flatten(msg, start_dim=1, end_dim=-1)
    mask1 = torch.stack(mask1)
    venus = torch.stack(venus)
    venus = torch.squeeze(venus)
    mask2 = torch.stack(mask2)
    server_model = torch.stack(server_model)
    server_model = torch.unsqueeze(server_model, dim=1)
    dump = torch.stack(dump)

    stack_tensor = torch.cat([msg, mask1, venus, mask2, server_model, dump], dim=1)

    # only T-SNE

    # X_tsne = TSNE(n_components=2, random_state=33, perplexity=50).fit_transform(stack_tensor)
    # plt.figure(figsize=(10, 5))
    # # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, label="t-SNE")
    # scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, label="t-SNE")
    # plt.legend(handles=scatter.legend_elements()[0], labels=["0", "1", "2", "3"], title="classes")
    # plt.show()

    # K-means and T-SNE
    model = KMeans(n_clusters=4, init="k-means++", n_init=10, max_iter=300, random_state=42)
    model.fit(stack_tensor)
    X_out = pd.DataFrame(stack_tensor, index=model.labels_)
    print(collections.Counter(model.labels_))
    X_out_center = pd.DataFrame(model.cluster_centers_)  # 聚类中心

    X_out_with_center = X_out.append(X_out_center)
    tsne = TSNE()
    tsne.fit_transform(X_out_with_center)
    X_tsne = pd.DataFrame(tsne.embedding_, index=X_out_with_center.index)

    d = X_tsne[X_tsne.index == 0]
    plt.scatter(d[0], d[1], c='lightgreen',
                marker='o')
    d = X_tsne[X_tsne.index == 1]
    plt.scatter(d[0], d[1], c='orange',
                marker='o')
    d = X_tsne[X_tsne.index == 2]
    plt.scatter(d[0], d[1], c='lightblue',
                marker='o')
    d = X_tsne[X_tsne.index == 3]
    plt.scatter(d[0], d[1], c='yellow',
                marker='o')
    d = X_tsne.tail(4)
    plt.scatter(d[0], d[1], c='red', s=150,
                marker='*')
    plt.show()
