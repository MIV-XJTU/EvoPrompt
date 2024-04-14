"""
Implementation of of optimal transport alignment is adopted 
and modified from https://github.com/sidak/otfusion
"""

import numpy as np
from typing import List
from loguru import logger
from easydict import EasyDict as edict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import ot
import captum


def isnan(x):
    return x != x


class GroundMetric:
    """
    Ground Metric object for Wasserstein computations:

    """

    def __init__(
        self,
        ground_metric: str = "euclidean",
        ground_metric_normalize: str = "none",
        clip_max: int = 5,
        clip_min: int = 0,
        reg: float = 1e-2,
        dist_normalize: bool = False,
        ground_metric_eff: bool = True,
        geom_ensemble_type: str = "wts",
        normalize_wts: bool = True,
        not_squared=False,
        activation_histograms: bool = False,
        act_num_samples: int = 200,
        clip_gm: bool = False,
    ):
        self.ground_metric_type = ground_metric
        self.ground_metric_normalize = ground_metric_normalize
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.reg = reg
        self.dist_normalize = dist_normalize
        self.squared = not not_squared
        self.mem_eff = ground_metric_eff
        self.geom_ensemble_type = geom_ensemble_type
        self.normalize_wts = normalize_wts
        self.activation_histograms = activation_histograms
        self.act_num_samples = act_num_samples
        self.clip_gm = clip_gm

    def _clip(self, ground_metric_matrix):
        percent_clipped = (
            float((ground_metric_matrix >= self.reg * self.clip_max).long().sum().data)
            / ground_metric_matrix.numel()
        ) * 100
        print("percent_clipped is (assumes clip_min = 0) ", percent_clipped)
        # will keep the M' = M/reg in range clip_min and clip_max
        ground_metric_matrix.clamp_(
            min=self.reg * self.clip_min, max=self.reg * self.clip_max
        )

        return ground_metric_matrix

    def _normalize(self, ground_metric_matrix):
        if self.ground_metric_normalize == "log":
            ground_metric_matrix = torch.log1p(ground_metric_matrix)
        elif self.ground_metric_normalize == "max":
            print(
                "Normalizing by max of ground metric and which is ",
                ground_metric_matrix.max(),
            )
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.max()
        elif self.ground_metric_normalize == "median":
            print(
                "Normalizing by median of ground metric and which is ",
                ground_metric_matrix.median(),
            )
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.median()
        elif self.ground_metric_normalize == "mean":
            print(
                "Normalizing by mean of ground metric and which is ",
                ground_metric_matrix.mean(),
            )
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.mean()
        elif self.ground_metric_normalize == "none":
            return ground_metric_matrix
        else:
            raise NotImplementedError

        return ground_metric_matrix

    def _sanity_check(self, ground_metric_matrix):
        assert not (ground_metric_matrix < 0).any()
        assert not (isnan(ground_metric_matrix).any())

    def _cost_matrix_xy(
        self, x: torch.Tensor, y: torch.Tensor, p: int = 2, squared: bool = True
    ):
        # TODO: Use this to guarantee reproducibility of previous results and then move onto better way
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
        if not squared:
            print("dont leave off the squaring of the ground metric")
            c = c ** (1 / 2)
        if self.dist_normalize:
            assert NotImplementedError
        return c

    def _pairwise_distances(
        self, x: torch.Tensor, y: torch.Tensor = None, squared=True
    ):
        """
        Source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        """
        x_norm = (x**2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y**2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        dist = torch.clamp(dist, min=0.0)

        if self.activation_histograms and self.dist_normalize:
            dist = dist / self.act_num_samples
            print("Divide squared distances by the num samples")

        if not squared:
            print("dont leave off the squaring of the ground metric")
            dist = dist ** (1 / 2)

        return dist

    def _get_euclidean(self, coordinates, other_coordinates=None):
        # TODO: Replace by torch.pdist (which is said to be much more memory efficient)

        if other_coordinates is None:
            matrix = torch.norm(
                coordinates.view(coordinates.shape[0], 1, coordinates.shape[1])
                - coordinates,
                p=2,
                dim=2,
            )
        else:
            if self.mem_eff:
                matrix = self._pairwise_distances(
                    coordinates, other_coordinates, squared=self.squared
                )
            else:
                matrix = self._cost_matrix_xy(
                    coordinates, other_coordinates, squared=self.squared
                )

        return matrix

    def _normed_vecs(self, vecs, eps=1e-9):
        norms = torch.norm(vecs, dim=-1, keepdim=True)
        print(
            "stats of vecs are: mean {}, min {}, max {}, std {}".format(
                norms.mean(), norms.min(), norms.max(), norms.std()
            )
        )
        return vecs / (norms + eps)

    def _get_cosine(self, coordinates, other_coordinates=None):
        if other_coordinates is None:
            matrix = coordinates / torch.norm(coordinates, dim=1, keepdim=True)
            matrix = 1 - matrix @ matrix.t()
        else:
            matrix = 1 - torch.div(
                coordinates @ other_coordinates.t(),
                torch.norm(coordinates, dim=1).view(-1, 1)
                @ torch.norm(other_coordinates, dim=1).view(1, -1),
            )
        return matrix.clamp_(min=0)

    def _get_angular(self, coordinates, other_coordinates=None):
        pass

    def get_metric(self, coordinates, other_coordinates=None):
        get_metric_map = {
            "euclidean": self._get_euclidean,
            "cosine": self._get_cosine,
            "angular": self._get_angular,
        }
        return get_metric_map[self.ground_metric_type](coordinates, other_coordinates)

    def process(self, coordinates, other_coordinates=None):
        print("Processing the coordinates to form ground_metric")
        if self.geom_ensemble_type == "wts" and self.normalize_wts:
            print("In weight mode: normalizing weights to unit norm")
            coordinates = self._normed_vecs(coordinates)
            if other_coordinates is not None:
                other_coordinates = self._normed_vecs(other_coordinates)

        ground_metric_matrix = self.get_metric(coordinates, other_coordinates)
        self._sanity_check(ground_metric_matrix)
        ground_metric_matrix = self._normalize(ground_metric_matrix)
        self._sanity_check(ground_metric_matrix)

        if self.clip_gm:
            ground_metric_matrix = self._clip(ground_metric_matrix)

        self._sanity_check(ground_metric_matrix)

        return ground_metric_matrix


def get_histogram(
    idx,
    cardinality,
    layer_name,
    activations=None,
    return_numpy=True,
    float64=False,
    unbalanced: bool = False,
    temperature: float = 1.0,
):
    if activations is None:
        # returns a uniform measure
        if not unbalanced:
            print("returns a uniform measure of cardinality: ", cardinality)
            return np.ones(cardinality) / cardinality
        else:
            return np.ones(cardinality)
    else:
        # return softmax over the activations raised to a temperature
        # layer_name is like 'fc1.weight', while activations only contains 'fc1'
        print(activations[idx].keys())
        unnormalized_weights = activations[idx][layer_name.split(".")[0]]
        print(
            "For layer {},  shape of unnormalized weights is ".format(layer_name),
            unnormalized_weights.shape,
        )
        unnormalized_weights = unnormalized_weights.squeeze()
        assert unnormalized_weights.shape[0] == cardinality

        if return_numpy:
            if float64:
                return (
                    torch.softmax(unnormalized_weights / temperature, dim=0)
                    .data.cpu()
                    .numpy()
                    .astype(np.float64)
                )
            else:
                return (
                    torch.softmax(unnormalized_weights / temperature, dim=0)
                    .data.cpu()
                    .numpy()
                )
        else:
            return torch.softmax(unnormalized_weights / temperature, dim=0)


def _get_neuron_importance_histogram(
    layer_weight: torch.Tensor, importance: str, eps=1e-9, unbalanced: bool = False
):

    layer = layer_weight.cpu().numpy()
    if importance == "l1":
        importance_hist = np.linalg.norm(layer, ord=1, axis=-1).astype(np.float64) + eps
    elif importance == "l2":
        importance_hist = np.linalg.norm(layer, ord=2, axis=-1).astype(np.float64) + eps
    else:
        raise NotImplementedError

    if not unbalanced:
        importance_hist = importance_hist / importance_hist.sum()

    return importance_hist


def get_activation(
    base_model,
    models: List[nn.Module],
    dataloader: torch.utils.data.DataLoader,
) -> List[List[torch.Tensor]]:
    def get_vectorized_params(module: nn.Module) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for param in list(module.parameters()):
            params.append(param.view(-1))
        return torch.cat(params)

    activations_results = list()

    # loop over new and old model (new model first)
    for k, model in enumerate(models):
        params_vec = get_vectorized_params(model)
        base_model.set_memory_params(params_vec)

        # define forward function for captum attribution
        def forward_func(inputs: torch.Tensor):
            with torch.no_grad():
                query_embed = base_model._network.forward_query(
                    inputs,
                )

            logits = base_model._network(inputs, query_embed=query_embed)["logits"]
            base_model.logits_ot = logits

            return logits

        # enlist target modules for hook
        modules = base_model._network.get_memory_modules()
        module_list = list()
        for m in modules:
            module_list.append(m.block[1])
            module_list.append(m.block[2])

        # doing inference
        attr_method = captum.attr.LayerActivation(
            forward_func=forward_func, layer=module_list
        )
        activation = list()
        for data in tqdm(dataloader, f"compute activation model {k}"):
            inputs = data["inputs"].to(base_model._device)
            targets = data["targets"].to(base_model._device)
            with torch.no_grad():
                act_out = attr_method.attribute(inputs)
            act_out = [act.detach().cpu() for act in act_out]
            preds = F.softmax(base_model.logits_ot, dim=-1).max(dim=-1).indices
            act_out = [F.relu(attr[preds == targets]) for attr in act_out]
            activation.append(act_out)

        # concat for each module along batch dimension
        act_model = list()
        num_modules = len(activation[0])
        for module_idx in range(num_modules):
            act_module = list()
            num_batch = len(activation)
            for batch_idx in range(num_batch):
                act_module.append(activation[batch_idx][module_idx])
            act_module = torch.cat(act_module, dim=0)  # num_samples, ndim
            act_module = act_module.mean(dim=0)
            act_model.append(act_module)

        # append to final results
        activations_results.append(act_model)

    # reset params back to original
    new_model = models[0]
    params_vec = get_vectorized_params(new_model)
    base_model.set_memory_params(params_vec)

    return activations_results


def get_wassersteinized_layers_modularized(
    models: List[nn.Module] = None,
    dataloader: torch.utils.data.DataLoader = None,
    base_model=None,
    metric_config: dict = None,
    eps=1e-7,
    correction: bool = True,
    proper_marginals: bool = True,
    ensemble_step: float = 0.5,
    exact: bool = True,
    reg: float = 1e-2,
    past_correction: bool = True,
    importance: str = "l1",
    activation_based: bool = False,
) -> torch.Tensor:
    """
    Two neural models that have to be averaged in geometric manner (i.e. layerwise).
    The 1st network is aligned with respect to the other via wasserstein distance.
    Also this assumes that all the layers are either fully connected or convolutional *(with no bias)*

    :param models: list of models
    :param activations: If not None, use it to build the activation histograms.
    Otherwise assumes uniform distribution over neurons in a layer.
    :return: list of layer weights 'wassersteinized'
    """

    if activation_based:
        activations = get_activation(base_model, models, dataloader)

    avg_aligned_layers = []
    T_vars = []
    ground_metric_object = GroundMetric()

    for idx, (
        (layer0_name, params0),
        (layer1_name, params1),
    ) in enumerate(zip(models[0].named_parameters(), models[1].named_parameters())):
        if idx % 2 == 0:
            T_var = None
            previous_layer_shape = None

        assert params0.shape == params1.shape
        previous_layer_shape = params1.shape
        params0_data = params0.data
        params1_data = params1.data

        if idx % 2 == 0:
            aligned_wt = params0_data
            M = ground_metric_object.process(params0_data, params1_data)
        else:
            aligned_wt = torch.matmul(params0.data, T_var)
            M = ground_metric_object.process(aligned_wt, params1)

        if not activation_based:
            mu = _get_neuron_importance_histogram(
                importance=importance, layer_weight=params0_data
            )
            nu = _get_neuron_importance_histogram(
                importance=importance, layer_weight=params1_data
            )
        else:
            mu = activations[0][idx]  # new memory
            nu = activations[1][idx]  # old memory

            # normalize
            cat_attr = torch.cat([mu, nu])
            max_attr, min_attr = cat_attr.max(), cat_attr.min()
            mu = ((mu - min_attr) / (max_attr - min_attr)).numpy
            nu = ((nu - min_attr) / (max_attr - min_attr)).numpy()

        cpuM = M.data.cpu().numpy()
        if exact:
            T_var = ot.emd(mu, nu, cpuM)
        else:
            T_var = ot.bregman.sinkhorn(mu, nu, cpuM, reg=reg)
        T_var = torch.from_numpy(T_var).type_as(params0_data)

        if correction:
            if not proper_marginals:
                # think of it as m x 1, scaling weights for m linear combinations of points in X
                marginals = torch.ones(T_var.shape[0]) / T_var.shape[0]
                marginals = torch.diag(1.0 / (marginals + eps)).type_as(
                    T_var
                )  # take inverse
                T_var = torch.matmul(T_var, marginals)
            else:
                marginals_beta = T_var.t() @ torch.ones(
                    T_var.shape[0], dtype=T_var.dtype
                ).type_as(T_var)

                marginals = 1 / (marginals_beta + eps)

                T_var = T_var * marginals
                # i.e., how a neuron of 2nd model is constituted by the neurons of 1st model
                # this should all be ones, and number equal to number of neurons in 2nd model
                print(f"Tvar sum: {T_var.sum(dim=0)}")

        print(
            "Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var)
        )
        print(
            "Here, trace is {} and matrix sum is {} ".format(
                torch.trace(T_var), torch.sum(T_var)
            )
        )

        T_vars.append(T_var)
        if past_correction:
            print("this is past correction for weight mode")
            t_fc0_model = torch.matmul(
                T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
            )
        else:
            t_fc0_model = torch.matmul(
                T_var.t(),
                params0_data.view(params0_data.shape[0], -1),
            )

        avg_aligned_layers.append(t_fc0_model.view(-1))
    return torch.cat(avg_aligned_layers), T_vars
