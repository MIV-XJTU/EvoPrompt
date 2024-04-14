import torch


def KMeans(x, device=torch.device, K=10, Niters=10, verbose=False):
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means loop:
    # - x  is the point cloud,
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    # start = time.time()
    x = x.to(device)
    c = x[:K, :].clone()  # Simplistic random initialization
    # x_i = LazyTensor(x[:, None, :])  # (Npoints, 1, D)
    x_i = x[:, None, :]  # (Npoints, 1, D)

    for i in range(Niters):
        c_j = c[None, :, :]  # (1, Nclusters, D)
        # c_j = LazyTensor(c[None, :, :])  # (1, Nclusters, D)
        D_ij = ((x_i - c_j) ** 2).sum(
            -1
        )  # (Npoints, Nclusters) symbolic matrix of squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # Ncl = torch.bincount(cl).type(torchtype[dtype])  # Class weights
        # for d in range(D):  # Compute the cluster centroids with torch.bincount:
        #     c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl
        # print(c)
        Ncl = cl.view(cl.size(0), 1).expand(-1, D).to(device)
        unique_labels, labels_count = Ncl.unique(dim=0, return_counts=True)
        # As some clusters don't contain any samples, manually assign count as 1
        labels_count_all = torch.ones([K]).long().to(device)
        labels_count_all[unique_labels[:, 0]] = labels_count
        c = torch.zeros([K, D], dtype=torch.float).to(device).scatter_add_(0, Ncl, x)
        c = c / labels_count_all.float().unsqueeze(1)

    return cl, c
