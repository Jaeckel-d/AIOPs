import collections
import os.path

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from aiop_dataset import AIOPSTest, AIOPSVal, AIOPS
import torch
from sklearn.cluster import KMeans
import pandas as pd
from spectralcluster import AutoTune, AutoTuneProxy


def load_test_set():
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
    return labels, stack_tensor


def load_csv_features(file_name, balance=True):
    df = pd.read_csv(file_name)

    if balance is not None:
        # rng = np.random.default_rng(seed=42)
        counts = df["label"].value_counts()
        min_count = counts.min()
        result = pd.DataFrame()
        for name, group in df.groupby('label'):
            result = pd.concat([result, df[df["label"] == name].sample(n=min_count, random_state=42)])
        df = result

    labels = df["label"].values
    labels = torch.tensor([float(label[1:-1]) for label in labels])
    feature_less = df["feature_less"].values.tolist()
    feature_less = [torch.tensor(list(map(float, feature[2:-2].split()))) for feature in feature_less]
    feature_less = torch.stack(feature_less)
    feature_more = df["feature_more"].values.tolist()
    feature_more = [torch.tensor(list(map(float, feature[2:-2].split()))) for feature in feature_more]
    feature_more = torch.stack(feature_more)

    return labels, feature_less, feature_more


def feature_cluster_test(base_path='./src', feature_type='gate', save_path="./src"):
    labels, feature_less, feature_more = load_csv_features(os.path.join(base_path, f'{feature_type}_feature.csv'))

    x_tsne_less = TSNE(n_components=2, random_state=33, perplexity=50).fit_transform(feature_less)
    plt.figure(figsize=(10, 5))
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, label="t-SNE")
    scatter = plt.scatter(x_tsne_less[:, 0], x_tsne_less[:, 1], c=labels, label="t-SNE")
    plt.legend(handles=scatter.legend_elements()[0], labels=["0", "1", "2", "3"], title="classes")
    plt.savefig(os.path.join(save_path, f"{feature_type}_less_only_tsne.png"))
    plt.show()

    x_tsne_more = TSNE(n_components=2, random_state=33, perplexity=50).fit_transform(feature_more)
    plt.figure(figsize=(10, 5))
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, label="t-SNE")
    scatter = plt.scatter(x_tsne_more[:, 0], x_tsne_more[:, 1], c=labels, label="t-SNE")
    plt.legend(handles=scatter.legend_elements()[0], labels=["0", "1", "2", "3"], title="classes")
    plt.savefig(os.path.join(save_path, f"{feature_type}_more_only_tsne.png"))
    plt.show()

    k_means = KMeans(n_clusters=4, init="k-means++", n_init=10, max_iter=300, random_state=42)
    k_means.fit(feature_less)
    plt.figure(figsize=(10, 5))
    x_out = pd.DataFrame(feature_less, index=k_means.labels_)
    print(collections.Counter(k_means.labels_))
    x_out_center = pd.DataFrame(k_means.cluster_centers_)  # 聚类中心
    x_out_with_center = x_out.append(x_out_center)

    tsne = TSNE()
    tsne.fit_transform(x_out_with_center)
    x_kmeans_tsne = pd.DataFrame(tsne.embedding_, index=x_out_with_center.index)

    d = x_kmeans_tsne[x_kmeans_tsne.index == 0]
    plt.scatter(d[0], d[1], c='lightgreen',
                marker='o')
    d = x_kmeans_tsne[x_kmeans_tsne.index == 1]
    plt.scatter(d[0], d[1], c='orange',
                marker='o')
    d = x_kmeans_tsne[x_kmeans_tsne.index == 2]
    plt.scatter(d[0], d[1], c='lightblue',
                marker='o')
    d = x_kmeans_tsne[x_kmeans_tsne.index == 3]
    plt.scatter(d[0], d[1], c='yellow',
                marker='o')
    d = x_kmeans_tsne.tail(4)
    plt.scatter(d[0], d[1], c='red', s=150,
                marker='*')
    plt.savefig(os.path.join(save_path, f"{feature_type}_less_kmeans_tsne.png"))
    plt.show()

    k_means = KMeans(n_clusters=4, init="k-means++", n_init=10, max_iter=300, random_state=42)
    k_means.fit(feature_more)
    plt.figure(figsize=(10, 5))
    x_out = pd.DataFrame(feature_more, index=k_means.labels_)
    print(collections.Counter(k_means.labels_))
    x_out_center = pd.DataFrame(k_means.cluster_centers_)  # 聚类中心
    x_out_with_center = x_out.append(x_out_center)

    tsne = TSNE()
    tsne.fit_transform(x_out_with_center)
    x_kmeans_tsne = pd.DataFrame(tsne.embedding_, index=x_out_with_center.index)

    d = x_kmeans_tsne[x_kmeans_tsne.index == 0]
    plt.scatter(d[0], d[1], c='lightgreen',
                marker='o')
    d = x_kmeans_tsne[x_kmeans_tsne.index == 1]
    plt.scatter(d[0], d[1], c='orange',
                marker='o')
    d = x_kmeans_tsne[x_kmeans_tsne.index == 2]
    plt.scatter(d[0], d[1], c='lightblue',
                marker='o')
    d = x_kmeans_tsne[x_kmeans_tsne.index == 3]
    plt.scatter(d[0], d[1], c='yellow',
                marker='o')
    d = x_kmeans_tsne.tail(4)
    plt.scatter(d[0], d[1], c='red', s=150,
                marker='*')
    plt.savefig(os.path.join(save_path, f"{feature_type}_more_kmeans_tsne.png"))
    plt.show()


if __name__ == '__main__':
    # tensors = load_test_set()
    feature_cluster_test()
    feature_cluster_test(feature_type="pool")
    feature_cluster_test(feature_type="lstm")
    # autotune = AutoTune(
    #     p_percentile_min=0.60,
    #     p_percentile_max=0.95,
    #     init_search_step=0.01,
    #     search_level=3,
    #     proxy=AutoTuneProxy.PercentileSqrtOverNME)
    # print(autotune.tune())

    #
    # # only T-SNE
    #
    # X_tsne = TSNE(n_components=2, random_state=33, perplexity=50).fit_transform(feature_less)
    # plt.figure(figsize=(10, 5))
    # # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, label="t-SNE")
    # scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, label="t-SNE")
    # plt.legend(handles=scatter.legend_elements()[0], labels=["0", "1", "2", "3"], title="classes")
    # plt.show()

    # only T-SNE

    # X_tsne = TSNE(n_components=2, random_state=33, perplexity=50).fit_transform(stack_tensor)
    # plt.figure(figsize=(10, 5))
    # # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, label="t-SNE")
    # scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, label="t-SNE")
    # plt.legend(handles=scatter.legend_elements()[0], labels=["0", "1", "2", "3"], title="classes")
    # plt.show()

    # K-means and T-SNE
    # model = KMeans(n_clusters=4, init="k-means++", n_init=10, max_iter=300, random_state=42)
    # model.fit(feature_less)
    # plt.figure(figsize=(10, 5))
    # X_out = pd.DataFrame(feature_less, index=model.labels_)
    # print(collections.Counter(model.labels_))
    # X_out_center = pd.DataFrame(model.cluster_centers_)  # 聚类中心
    #
    # X_out_with_center = X_out.append(X_out_center)
    # tsne = TSNE()
    # tsne.fit_transform(X_out_with_center)
    # X_tsne = pd.DataFrame(tsne.embedding_, index=X_out_with_center.index)
    #
    # d = X_tsne[X_tsne.index == 0]
    # plt.scatter(d[0], d[1], c='lightgreen',
    #             marker='o')
    # d = X_tsne[X_tsne.index == 1]
    # plt.scatter(d[0], d[1], c='orange',
    #             marker='o')
    # d = X_tsne[X_tsne.index == 2]
    # plt.scatter(d[0], d[1], c='lightblue',
    #             marker='o')
    # d = X_tsne[X_tsne.index == 3]
    # plt.scatter(d[0], d[1], c='yellow',
    #             marker='o')
    # d = X_tsne.tail(4)
    # plt.scatter(d[0], d[1], c='red', s=150,
    #             marker='*')
    # plt.show()
