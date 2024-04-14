from loguru import logger
from tqdm import tqdm
from typing import List

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F

from inclearn.lib import distance as distance_lib
from inclearn.lib import utils

from .postprocessors import FactorScalar


class Classifier(nn.Module):
    classifier_type = "fc"

    def __init__(
        self,
        features_dim,
        device,
        *,
        use_bias=False,
        normalize=False,
        init="kaiming",
        train_negative_weights=False,
        **kwargs,
    ):
        super().__init__()

        self.features_dim = features_dim
        self.use_bias = use_bias
        self.init_method = init
        self.device = device
        self.normalize = normalize
        self._weights = nn.ParameterList([])
        self._bias = nn.ParameterList([]) if self.use_bias else None

        self.train_negative_weights = train_negative_weights
        self._negative_weights = None
        self.use_neg_weights = True
        self.eval_negative_weights = False

        self.proxy_per_class = 1

        self.n_classes = 0

    def on_task_end(self):
        pass

    def on_epoch_end(self):
        pass

    @property
    def weights(self):
        return torch.cat([w for w in self._weights])

    @property
    def new_weights(self):
        return self._weights[-1]

    @property
    def old_weights(self):
        if len(self._weights) > 1:
            return self._weights[:-1]
        return None

    @property
    def bias(self):
        if self._bias is not None:
            return torch.cat([b for b in self._bias])
        return None

    @property
    def new_bias(self):
        return self._bias[-1]

    @property
    def old_bias(self):
        if len(self._bias) > 1:
            return self._bias[:-1]
        return None

    def forward(self, features):
        if len(self._weights) == 0:
            raise Exception("Add some classes before training.")

        weights = self.weights
        if (
            self._negative_weights is not None
            and (self.training is True or self.eval_negative_weights)
            and self.use_neg_weights
        ):
            weights = torch.cat((weights, self._negative_weights), 0)

        if self.normalize:
            features = F.normalize(features, dim=1, p=2)

        logits = F.linear(features, weights, bias=self.bias)
        return {"logits": logits}

    def add_classes(self, n_classes):
        self._weights.append(nn.Parameter(torch.randn(n_classes, self.features_dim)))
        self._init(self.init_method, self.new_weights)

        if self.use_bias:
            self._bias.append(nn.Parameter(torch.randn(n_classes)))
            self._init(0.0, self.new_bias)

        self.to(self.device)

    def reset_weights(self):
        self._init(self.init_method, self.weights)

    @staticmethod
    def _init(init_method, parameters):
        if isinstance(init_method, float) or isinstance(init_method, int):
            nn.init.constant_(parameters, init_method)
        elif init_method == "kaiming":
            nn.init.kaiming_normal_(parameters, nonlinearity="linear")
        else:
            raise NotImplementedError(
                "Unknown initialization method: {}.".format(init_method)
            )

    def align_weights(self):
        """Align new weights based on old weights norm.

        # Reference:
            * Maintaining Discrimination and Fairness in Class Incremental Learning
              Zhao et al. 2019
        """
        with torch.no_grad():
            old_weights = torch.cat([w for w in self.old_weights])

            old_norm = torch.mean(old_weights.norm(dim=1))
            new_norm = torch.mean(self.new_weights.norm(dim=1))

            self._weights[-1] = nn.Parameter((old_norm / new_norm) * self._weights[-1])

    def align_features(self, features):
        avg_weights_norm = self.weights.data.norm(dim=1).mean()
        avg_features_norm = features.data.norm(dim=1).mean()

        features.data = features.data * (avg_weights_norm / avg_features_norm)
        return features

    def add_custom_weights(self, weights, ponderate=None, **kwargs):
        if isinstance(ponderate, str):
            if ponderate == "weights_imprinting":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                weights = weights * avg_weights_norm
            elif ponderate == "align_weights":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                avg_new_weights_norm = weights.data.norm(dim=1).mean()

                ratio = avg_weights_norm / avg_new_weights_norm
                weights = weights * ratio
            else:
                raise NotImplementedError(f"Unknown ponderation type {ponderate}.")

        self._weights.append(nn.Parameter(weights))
        self.to(self.device)

    def set_negative_weights(self, negative_weights, ponderate=False):
        """Add weights that are used like the usual weights, but aren't actually
        parameters.

        :param negative_weights: Tensor of shape (n_classes * nb_proxy, features_dim)
        :param ponderate: Reponderate the negative weights by the existing weights norm, as done by
                          "Weights Imprinting".
        """
        logger.info("Add negative weights.")
        if isinstance(ponderate, str):
            if ponderate == "weights_imprinting":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                negative_weights = negative_weights * avg_weights_norm
            elif ponderate == "align_weights":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                avg_negative_weights_norm = negative_weights.data.norm(dim=1).mean()

                ratio = avg_weights_norm / avg_negative_weights_norm
                negative_weights = negative_weights * ratio
            elif ponderate == "inv_align_weights":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                avg_negative_weights_norm = negative_weights.data.norm(dim=1).mean()

                ratio = avg_negative_weights_norm / avg_weights_norm
                negative_weights = negative_weights * ratio
            else:
                raise NotImplementedError(f"Unknown ponderation type {ponderate}.")

        if self.train_negative_weights:
            self._negative_weights = nn.Parameter(negative_weights)
        else:
            self._negative_weights = negative_weights


class CosineClassifier(nn.Module):
    classifier_type = "cosine"

    def __init__(
        self,
        features_dim,
        device,
        *,
        proxy_per_class=1,
        distance="cosine",
        merging="softmax",
        scaling=1,
        gamma=1.0,
        use_bias=False,
        type=None,
        pre_fc=None,
        negative_weights_bias=None,
        train_negative_weights=False,
        eval_negative_weights=False,
    ):
        super().__init__()

        self.n_classes = 0
        self._weights = nn.ParameterList([])
        self.bias = None
        self.features_dim = features_dim
        self.proxy_per_class = proxy_per_class
        self.device = device
        self.distance = distance
        self.merging = merging
        self.gamma = gamma

        self.negative_weights_bias = negative_weights_bias
        self.train_negative_weights = train_negative_weights
        self.eval_negative_weights = eval_negative_weights

        self._negative_weights = None
        self.use_neg_weights = True

        if isinstance(scaling, int) or isinstance(scaling, float):
            self.scaling = scaling
        else:
            logger.warning("Using inner learned scaling")
            self.scaling = FactorScalar(1.0)

        if proxy_per_class > 1:
            logger.info("Using {} proxies per class.".format(proxy_per_class))

        if pre_fc is not None:
            self.pre_fc = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(self.features_dim),
                nn.Linear(self.features_dim, pre_fc),
            )
            self.features_dim = pre_fc
        else:
            self.pre_fc = None

        self._task_idx = 0

    def on_task_end(self):
        self._task_idx += 1
        if isinstance(self.scaling, nn.Module):
            self.scaling.on_task_end()

    def on_epoch_end(self):
        if isinstance(self.scaling, nn.Module):
            self.scaling.on_epoch_end()

    def forward(self, features):
        if hasattr(self, "pre_fc") and self.pre_fc is not None:
            features = self.pre_fc(features)

        weights = self.weights
        if (
            self._negative_weights is not None
            and (self.training is True or self.eval_negative_weights)
            and self.use_neg_weights
        ):
            weights = torch.cat((weights, self._negative_weights), 0)

        if self.distance == "cosine":
            raw_similarities = distance_lib.cosine_similarity(features, weights)
        elif self.distance == "stable_cosine_distance":
            features = self.scaling * F.normalize(features, p=2, dim=-1)
            weights = self.scaling * F.normalize(weights, p=2, dim=-1)

            raw_similarities = distance_lib.stable_cosine_distance(features, weights)
        elif self.distance == "neg_stable_cosine_distance":
            features = self.scaling * F.normalize(features, p=2, dim=-1)
            weights = self.scaling * F.normalize(weights, p=2, dim=-1)

            raw_similarities = -distance_lib.stable_cosine_distance(features, weights)
        elif self.distance == "prelu_stable_cosine_distance":
            features = self.scaling * F.normalize(F.relu(features), p=2, dim=-1)
            weights = self.scaling * F.normalize(weights, p=2, dim=-1)

            raw_similarities = distance_lib.stable_cosine_distance(features, weights)
        elif self.distance == "prelu_neg_stable_cosine_distance":
            features = self.scaling * F.normalize(F.relu(features), p=2, dim=-1)
            weights = self.scaling * F.normalize(weights, p=2, dim=-1)

            raw_similarities = -distance_lib.stable_cosine_distance(features, weights)
        else:
            raise NotImplementedError(
                "Unknown distance function {}.".format(self.distance)
            )

        if self.proxy_per_class > 1:
            similarities = self._reduce_proxies(raw_similarities)
        else:
            similarities = raw_similarities

            if (
                self._negative_weights is not None
                and self.negative_weights_bias is not None
                and self.training is True
            ):
                qt = self._negative_weights.shape[0]
                if isinstance(self.negative_weights_bias, float):
                    similarities[..., -qt:] = torch.clamp(
                        similarities[..., -qt:] - self.negative_weights_bias, min=0
                    )
                elif (
                    isinstance(self.negative_weights_bias, str)
                    and self.negative_weights_bias == "min"
                ):
                    min_simi = similarities[..., :-qt].min(dim=1, keepdim=True)[0]
                    similarities = torch.min(
                        similarities,
                        torch.cat(
                            (similarities[..., :-qt], min_simi.repeat(1, qt)), dim=1
                        ),
                    )
                elif (
                    isinstance(self.negative_weights_bias, str)
                    and self.negative_weights_bias == "max"
                ):
                    max_simi = (
                        similarities[..., :-qt].max(dim=1, keepdim=True)[0] - 1e-6
                    )
                    similarities = torch.min(
                        similarities,
                        torch.cat(
                            (similarities[..., :-qt], max_simi.repeat(1, qt)), dim=1
                        ),
                    )
                elif isinstance(
                    self.negative_weights_bias, str
                ) and self.negative_weights_bias.startswith("top_"):
                    topk = int(self.negative_weights_bias.replace("top_", ""))
                    botk = min(qt - topk, qt)

                    indexes = (-similarities[..., -qt:]).topk(botk, dim=1)[1]
                    similarities[..., -qt:].scatter_(1, indexes, 0.0)
                else:
                    raise NotImplementedError(f"Unknown {self.negative_weights_bias}.")

        return {"logits": similarities, "raw_logits": raw_similarities}

    def _reduce_proxies(self, similarities):
        # shape (batch_size, n_classes * proxy_per_class)
        n_classes = similarities.shape[1] / self.proxy_per_class
        assert n_classes.is_integer(), (similarities.shape[1], self.proxy_per_class)
        n_classes = int(n_classes)
        bs = similarities.shape[0]

        if self.merging == "mean":
            return similarities.view(bs, n_classes, self.proxy_per_class).mean(-1)
        elif self.merging == "softmax":
            simi_per_class = similarities.view(bs, n_classes, self.proxy_per_class)
            attentions = F.softmax(
                self.gamma * simi_per_class, dim=-1
            )  # shouldn't be -gamma?
            return (simi_per_class * attentions).sum(-1)
        elif self.merging == "max":
            return similarities.view(bs, n_classes, self.proxy_per_class).max(-1)[0]
        elif self.merging == "min":
            return similarities.view(bs, n_classes, self.proxy_per_class).min(-1)[0]
        else:
            raise ValueError(
                "Unknown merging for multiple centers: {}.".format(self.merging)
            )

    # ------------------
    # Weights management
    # ------------------

    def align_features(self, features):
        avg_weights_norm = self.weights.data.norm(dim=1).mean()
        avg_features_norm = features.data.norm(dim=1).mean()

        features.data = features.data * (avg_weights_norm / avg_features_norm)
        return features

    def add_custom_weights(self, weights, ponderate=None, **kwargs):
        if isinstance(ponderate, str):
            if ponderate == "weights_imprinting":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                weights = weights * avg_weights_norm
            elif ponderate == "align_weights":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                avg_new_weights_norm = weights.data.norm(dim=1).mean()

                ratio = avg_weights_norm / avg_new_weights_norm
                weights = weights * ratio
            else:
                raise NotImplementedError(f"Unknown ponderation type {ponderate}.")

        self._weights.append(nn.Parameter(weights))
        self.to(self.device)

    def align_weights(self):
        """Align new weights based on old weights norm.

        # Reference:
            * Maintaining Discrimination and Fairness in Class Incremental Learning
              Zhao et al. 2019
        """
        if len(self._weights) == 1:
            return

        with torch.no_grad():
            old_weights = torch.cat([w for w in self.old_weights])

            old_norm = torch.mean(old_weights.norm(dim=1))
            new_norm = torch.mean(self.new_weights.norm(dim=1))

            self._weights[-1] = nn.Parameter((old_norm / new_norm) * self._weights[-1])

    def align_weights_i_to_j(self, indexes_i, indexes_j):
        with torch.no_grad():
            base_weights = self.weights[indexes_i]

            old_norm = torch.mean(base_weights.norm(dim=1))
            new_norm = torch.mean(self.weights[indexes_j].norm(dim=1))

            self.weights[indexes_j] = nn.Parameter(
                (old_norm / new_norm) * self.weights[indexes_j]
            )

    def align_inv_weights(self):
        """Align new weights based on old weights norm.

        # Reference:
            * Maintaining Discrimination and Fairness in Class Incremental Learning
              Zhao et al. 2019
        """
        with torch.no_grad():
            old_weights = torch.cat([w for w in self.old_weights])

            old_norm = torch.mean(old_weights.norm(dim=1))
            new_norm = torch.mean(self.new_weights.norm(dim=1))

            self._weights[-1] = nn.Parameter((new_norm / old_norm) * self._weights[-1])

    @property
    def weights(self):
        return torch.cat([clf for clf in self._weights])

    @property
    def new_weights(self):
        return self._weights[-1]

    @property
    def old_weights(self):
        if len(self._weights) > 1:
            return self._weights[:-1]
        return None

    def add_classes(self, n_classes):
        new_weights = nn.Parameter(
            torch.zeros(self.proxy_per_class * n_classes, self.features_dim)
        )
        nn.init.kaiming_normal_(new_weights, nonlinearity="linear")

        self._weights.append(new_weights)

        self.to(self.device)
        self.n_classes += n_classes
        return self

    def add_imprinted_classes(
        self, class_indexes, inc_dataset, network, multi_class_diff="normal", type=None
    ):
        if self.proxy_per_class > 1:
            logger.info("Multi class diff {}.".format(multi_class_diff))

        weights_norm = self.weights.data.norm(dim=1, keepdim=True)
        avg_weights_norm = torch.mean(weights_norm, dim=0).cpu()

        new_weights = []
        for class_index in class_indexes:
            _, loader = inc_dataset.get_custom_loader([class_index])
            features, _ = utils.extract_features(network, loader)

            features_normalized = F.normalize(torch.from_numpy(features), p=2, dim=1)
            class_embeddings = torch.mean(features_normalized, dim=0)
            class_embeddings = F.normalize(class_embeddings, dim=0, p=2)

            if self.proxy_per_class == 1:
                new_weights.append(class_embeddings * avg_weights_norm)
            else:
                if multi_class_diff == "normal":
                    std = torch.std(features_normalized, dim=0)
                    for _ in range(self.proxy_per_class):
                        new_weights.append(
                            torch.normal(class_embeddings, std) * avg_weights_norm
                        )
                elif multi_class_diff == "kmeans":
                    clusterizer = KMeans(n_clusters=self.proxy_per_class)
                    clusterizer.fit(features_normalized.numpy())

                    for center in clusterizer.cluster_centers_:
                        new_weights.append(torch.tensor(center) * avg_weights_norm)
                else:
                    raise ValueError(
                        "Unknown multi class differentiation for imprinted weights: {}.".format(
                            multi_class_diff
                        )
                    )

        new_weights = torch.stack(new_weights)
        self._weights.append(nn.Parameter(new_weights))

        self.to(self.device)
        self.n_classes += len(class_indexes)

        return self

    def set_negative_weights(self, negative_weights, ponderate=False):
        """Add weights that are used like the usual weights, but aren't actually
        parameters.

        :param negative_weights: Tensor of shape (n_classes * nb_proxy, features_dim)
        :param ponderate: Reponderate the negative weights by the existing weights norm, as done by
                          "Weights Imprinting".
        """
        logger.info("Add negative weights.")
        if isinstance(ponderate, str):
            if ponderate == "weights_imprinting":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                negative_weights = negative_weights * avg_weights_norm
            elif ponderate == "align_weights":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                avg_negative_weights_norm = negative_weights.data.norm(dim=1).mean()

                ratio = avg_weights_norm / avg_negative_weights_norm
                negative_weights = negative_weights * ratio
            elif ponderate == "inv_align_weights":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                avg_negative_weights_norm = negative_weights.data.norm(dim=1).mean()

                ratio = avg_negative_weights_norm / avg_weights_norm
                negative_weights = negative_weights * ratio
            else:
                raise NotImplementedError(f"Unknown ponderation type {ponderate}.")

        if self.train_negative_weights:
            self._negative_weights = nn.Parameter(negative_weights)
        else:
            self._negative_weights = negative_weights


class MCCosineClassifier(CosineClassifier):
    """CosineClassifier with MC-Dropout."""

    def __init__(self, *args, dropout=0.2, nb_samples=10, **kwargs):
        super().__init__(*args, **kwargs)

        self._dropout = dropout
        self.nb_samples = nb_samples

    def forward(self, x):
        if self.training:
            return super().forward(F.dropout(x, p=self._dropout))

        sampled_similarities = (
            torch.zeros(x.shape[0], self.nb_samples, self.n_classes)
            .to(x.device)
            .float()
        )
        for i in range(self.nb_samples):
            similarities = super().forward(F.dropout(x, p=self._dropout))["logits"]
            sampled_similarities[:, i] = similarities

        return {
            "logits": sampled_similarities.mean(dim=1),
            "var_ratio": self.var_ratio(sampled_similarities),
        }

    def var_ratio(self, sampled_similarities):
        predicted_class = sampled_similarities.max(dim=2)[1].cpu().numpy()

        hist = np.array(
            [
                np.histogram(predicted_class[i, :], range=(0, 10))[0]
                for i in range(predicted_class.shape[0])
            ]
        )

        return 1.0 - hist.max(axis=1) / self.nb_samples


class CosineM2KDClassifier(CosineClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._auxilliary_weights = nn.ParameterList([])
        self.auxilliary_features_dim = (
            64 * 8 * 8
        )  # Hard coded penultimate residual block
        # Only work on ResNet34-rebuffi with nf=16

    def add_imprinted_classes(self, class_indexes, *args, **kwargs):
        super().add_imprinted_classes(class_indexes, *args, **kwargs)
        self.add_classes_to_auxilliary(len(class_indexes))

    def add_classes_to_auxilliary(self, n_classes):
        new_weights = nn.Parameter(torch.zeros(n_classes, self.auxilliary_features_dim))
        nn.init.kaiming_normal_(new_weights, nonlinearity="linear")

        self._auxilliary_weights.append(new_weights)

        self.to(self.device)
        return self

    @property
    def auxilliary_weights(self):
        return torch.cat([clf for clf in self._weights])

    @property
    def new_weights(self):
        return torch.cat([self._weights[-1], self._auxilliary_weights[-1]])

    @property
    def old_weights(self):
        if len(self._weights) > 1:
            return torch.cat([self._weights[:-1], self._auxilliary_weights[:-1]])
        return None


class DomainClassifier(nn.Module):
    def __init__(self, features_dim, device=None):
        super().__init__()

        self.features_dim = features_dim
        self.device = device

        self.gradreverse = GradReverse.apply
        self.linear = nn.Linear(features_dim, 1)

        self.to(device)

    def forward(self, x):
        return self.linear(self.gradreverse(x))


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


class BinaryCosineClassifier(nn.Module):
    def __init__(self, features_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, features_dim))
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")

    def forward(self, x):
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(self.weight, dim=1, p=2)

        return {"logits": torch.mm(x, w.T)}


class CosineSimilarityClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        num_proxies_per_cls: int = 1,
        scale_cls=1.0,
        reduction: str = "softmax",
        imprinting_weights: bool = False,
        multi_class_diff: str = "normal",
        gamma=1.0,
        normalize_imprinted_weight: bool = True,
        use_avg_weights_norm: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_proxies_per_cls = num_proxies_per_cls
        self.in_features = in_features
        self.reduction = reduction
        self.imprinting_weights = imprinting_weights
        self.multi_class_diff = multi_class_diff
        self.gamma = gamma
        self.normalize_imprinted_weight = normalize_imprinted_weight
        self.use_avg_weights_norm = use_avg_weights_norm

        # weight per task
        self.scale_cls = nn.Parameter(
            torch.FloatTensor(1).fill_(scale_cls), requires_grad=True
        )
        logger.info(
            "\nClassifier config:\n"
            f"num classes: {self.num_classes}\n"
            f"num proxy per cls: {self.num_proxies_per_cls}\n"
            f"reduction: {self.reduction}\n"
            f"imprinting_weights: {self.imprinting_weights}\n"
            f"multi_class_diff: {self.multi_class_diff}\n"
            f"normalize_imprinted_weight: {self.normalize_imprinted_weight}\n"
            f"use_avg_weights_norm: {self.use_avg_weights_norm}"
        )

        self._nparam = 0
        self.init_weights(self.num_classes)

    def init_weights(self, num_classes):
        weights = nn.Parameter(
            torch.zeros(self.num_proxies_per_cls * num_classes, self.in_features)
        )
        nn.init.kaiming_normal_(weights, nonlinearity="linear")
        self.register_parameter(f"_weights{self._nparam}", weights)

    def forward(self, x: torch.Tensor):
        """forward pass to cosine classifier

        Args:
            x (torch.Tensor): input features with shape (batch size, num classes, embedding dim)

        Returns:
            torch.Tensor: output logits with shape (batch size, num classes)
        """
        # n_cls*n_proxies, n_dim -> 1, n_cls, n_proxies, n_dim
        weights = self.weight.view(
            1, self.num_classes, self.num_proxies_per_cls, self.in_features
        )
        if x.ndim == 3:
            # x: bsz, n_cls, n_dim
            feats = x.unsqueeze(2)  # bsz, n_cls, 1, n_dim
        elif x.ndim == 2:
            # x: bsz, n_dim
            bsz = x.size(0)
            ndim = x.size(1)
            feats = x.view(bsz, 1, 1, ndim)

        feats = self.scale_cls * F.normalize(feats, p=2, dim=-1)
        weights = self.scale_cls * F.normalize(weights, p=2, dim=-1)
        sim_mat = (feats * weights).sum(-1)

        logits = sim_mat
        if self.num_proxies_per_cls > 1:
            logits = self.reduce_proxies(sim_mat)
        else:
            logits = logits.squeeze(2)

        return logits, sim_mat

    def reduce_proxies(self, sim_mat):
        # shape (batch_size, n_classes, proxy_per_class)

        if self.reduction == "mean":
            return sim_mat.mean(-1)
        elif self.reduction == "softmax":
            attentions = F.softmax(self.gamma * sim_mat, dim=-1)
            return (sim_mat * attentions).sum(-1)
        elif self.reduction == "max":
            return sim_mat.max(-1)[0]
        elif self.reduction == "min":
            return sim_mat.min(-1)[0]
        else:
            raise ValueError(
                "Unknown merging for multiple centers: {}.".format(self.merging)
            )

    def update(
        self,
        num_new_classes: int,
        model=None,
        dataloader=None,
        new_classes: List[int] = None,
        device: torch.device = None,
    ):

        if self.imprinting_weights:
            assert model is not None, "Model should provided for imprinting new weights"
            assert (
                dataloader is not None
            ), "Dataloader should provided for imprinting new weights"

            with torch.no_grad():
                feats = list()
                labels = list()
                model.to(device)
                for batch_data in tqdm(dataloader, desc="extract feats"):
                    imgs, label = batch_data["inputs"], batch_data["targets"]
                    imgs = imgs.type_as(next(model.parameters()))
                    embedding = model(imgs)["embedding"]
                    feats.append(embedding)
                    labels.append(label)

            feats = torch.cat(feats, dim=0)
            labels = torch.cat(labels)
            prev_weights_norm = self.weight.data.norm(dim=1, keepdim=True)
            avg_weights_norm = torch.mean(prev_weights_norm, dim=0)

            new_weight_per_cls = list()
            for cls in new_classes:
                indices = torch.zeros_like(labels)
                indices += labels == cls
                cur_cls_index = indices.nonzero().squeeze(1)

                cls_embedding_norm = (
                    F.normalize(feats[cur_cls_index], p=2, dim=1)
                    if self.normalize_imprinted_weight
                    else feats[cur_cls_index]
                )
                cls_embedding_mean = torch.mean(cls_embedding_norm, dim=0)
                cls_embedding = (
                    F.normalize(cls_embedding_mean, p=2, dim=0)
                    if self.normalize_imprinted_weight
                    else cls_embedding_mean
                )

                if self.num_proxies_per_cls == 1:
                    if self.use_avg_weights_norm:
                        cls_embedding = cls_embedding * avg_weights_norm
                    new_weight_per_cls.append(cls_embedding.type_as(avg_weights_norm))
                else:
                    if self.multi_class_diff == "normal":
                        std = torch.std(cls_embedding_norm, dim=0)
                        for _ in range(self.num_proxies_per_cls):
                            new_w = torch.normal(cls_embedding, std)
                            if self.use_avg_weights_norm:
                                new_w = new_w * avg_weights_norm
                            new_weight_per_cls.append(new_w.type_as(avg_weights_norm))
                    elif self.multi_class_diff == "kmeans":
                        clusterizer = KMeans(n_clusters=self.num_proxies_per_cls)
                        clusterizer.fit(cls_embedding_norm.cpu().numpy())

                        for center in clusterizer.cluster_centers_:
                            new_w = torch.tensor(center).type_as(avg_weights_norm)
                            if self.use_avg_weights_norm:
                                new_w = new_w * avg_weights_norm
                            new_weight_per_cls.append(new_w)
                    else:
                        raise ValueError(
                            f"Unknown multi class differentiation for imprinted weights: {self.multi_class_diff}."
                        )

            new_weights = torch.stack(new_weight_per_cls)
            self._nparam += 1
            self.register_parameter(
                f"_weights{self._nparam}", nn.Parameter(new_weights)
            )
        else:
            self._nparam += 1
            self.init_weights(num_new_classes)

        self.num_classes += num_new_classes

    @property
    def weight(self):
        return torch.cat(
            [getattr(self, f"_weights{i}") for i in range(self._nparam + 1)]
        )

    @property
    def weights_per_cls(self) -> List[torch.Tensor]:
        weights = self.weight
        proxy_list = list()

        for cls in range(self.num_classes):
            start = cls * self.num_proxies_per_cls
            end = start + self.num_proxies_per_cls
            cls_weight = weights[start:end]
            proxy_list.append(cls_weight)

        return proxy_list
