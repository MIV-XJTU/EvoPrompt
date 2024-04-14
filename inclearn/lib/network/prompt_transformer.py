from typing import List
import copy
from easydict import EasyDict as edict
from copy import deepcopy

import torch.nn as nn
import torch.nn.functional as F
import torch
import timm
from scipy.spatial.distance import cdist

from .classifiers import DomainClassifier
from inclearn.lib.network import CosineSimilarityClassifier
from inclearn.lib.utils import freeze_module, unfreeze_module

from loguru import logger


QUERY_FUNCTION = ["whole", "patch_embed", "first"]
CLASSIFIER_POOL = ["image", "cls_token", "prompt", "global"]


def forward_block(self, x: torch.Tensor, **kwargs):
    sa_out, attn = self.attn(self.norm1(x), **kwargs)
    x = x + self.drop_path1(self.ls1(sa_out))
    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

    return x, attn


def forward_attn(self, x: torch.Tensor):
    B, N, C = x.shape

    qkv = (
        self.qkv(x)
        .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv.unbind(
        0
    )  # make torchscript happy (cannot use tensor as tuple) # bsz, num_heads, num_token, ndim_subspace

    attn = (q @ k.transpose(-2, -1)) * self.scale  # bsz, num_heads, q_len, k_len

    # vanilla attn
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x, attn


class PromptMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 8,
        bias: bool = False,
        dropout: float = 0.0,
        length: int = 4,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.len = length

        non_linearity = nn.ReLU(inplace=True)
        if activation == "sigmoid":
            non_linearity = nn.Sigmoid()
        elif activation == "attention":
            non_linearity = nn.Softmax(dim=-1)

        self.block = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features, bias=bias),
            non_linearity,
            nn.Linear(self.hidden_features, self.out_features * self.len, bias=bias),
        )
        if dropout > 0.0:
            self.block[1].register_forward_hook(
                lambda m, inp, out: F.dropout(out, p=dropout, training=m.training)
            )

    def forward(self, x: torch.Tensor):
        bsz = x.size(0)
        out = self.block(x)
        out = out.reshape(bsz, self.len, self.out_features)

        return out


class ContinuousPrompt(nn.Module):
    def __init__(
        self,
        num_layers: int = 12,
        hidden_dim: int = 48,
        use_bias: bool = False,
        length: int = 4,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.length = length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        logger.info(f" Prompt memory dropout: {self.dropout}")

        self.init()

    def init(self):
        self.memory_list = nn.ModuleList()
        out_dim = self.embed_dim
        in_dim = self.embed_dim

        for _ in range(self.num_layers):
            module = PromptMLP(
                in_dim,
                out_dim,
                self.hidden_dim,
                self.use_bias,
                self.dropout,
                self.length,
                self.activation,
            )
            self.memory_list.append(module)

    def forward(self, context: torch.Tensor):
        outputs = list()

        for i, module in enumerate(self.memory_list):
            x = context  # bsz, ndim
            outputs.append(module(x).unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)  # bsz, num_layers, length, ndim
        outputs = outputs.permute(1, 0, 2, 3)  # num_layers, bsz, length, ndim

        return outputs


class ViTPrompt(nn.Module):
    def __init__(
        self,
        convnet_kwargs={},
        classifier_name: str = "linear",
        classifier_kwargs={},
        device=None,
        classifier_pool: str = "cls_token",
        freeze_old_classifier: bool = False,
        using_dynamic_classifier: bool = False,
        num_layers: int = 12,
        query_function: str = "whole",
        query_layer: int = 11,
        # inference
        topk_inference: int = 1,
        # prompt
        # instance aware prompting params,
        continuous_prompt_config: dict = None,
        num_tasks: int = -1,
        *args,
        **kwargs,
    ):
        super(ViTPrompt, self).__init__()

        self.device = device
        self.return_logits_only = False
        self.num_classes_sofar = 0

        # vit backbone
        self.query_layer = query_layer
        backbone = timm.create_model(**convnet_kwargs).to(device)
        freeze_module(backbone)
        self.patch_embed = backbone.patch_embed
        self._pos_embed = backbone._pos_embed
        self.blocks = backbone.blocks
        if len(self.blocks) != num_layers:
            self.blocks = nn.Sequential(*[self.blocks[i] for i in range(num_layers)])
        self.norm = backbone.norm
        self.fc_norm = backbone.fc_norm
        self.num_prefix_tokens = backbone.num_prefix_tokens
        self.embed_dim = backbone.num_features
        self.classifier_name = classifier_name
        self.query_function = query_function
        self.use_prompt = False
        self.embed_len = self.patch_embed.num_patches + self.num_prefix_tokens

        # test-time inference strategy
        self.topk_inference = topk_inference
        assert (
            query_function in QUERY_FUNCTION
        ), f"Not support {query_function}, supported patch embed layer: {QUERY_FUNCTION}"

        self.out_cls_key_map = edict(linear="out_features", cosine="num_classes")

        # ----------- PROMPT -----------
        self.configure_blocks()
        # ----------- Continuous Prompt -----------
        self.continuous_prompt_config = continuous_prompt_config
        self.prompt_memory = None
        self.init_prompt_memory()
        self.prompting_mode = "prompt_tuning"

        # ----------- Classifier Head -----------
        self.freeze_old_classifier = freeze_old_classifier
        self.using_dynamic_classifier = using_dynamic_classifier
        self.classifier_pool = classifier_pool
        classifier_kwargs.in_features = backbone.num_features
        self.classifier_kwargs = classifier_kwargs
        assert (
            self.classifier_pool in CLASSIFIER_POOL
        ), f"Not supported pool {self.classifier_pool}, supported pool: {CLASSIFIER_POOL}"
        self.init_classifier()

    def init_classifier(self):
        if self.classifier_name == "linear":
            classifier_module = nn.Linear
        elif self.classifier_name == "cosine":
            classifier_module = CosineSimilarityClassifier
        else:
            raise RuntimeError(f"Classifier {self.classifier_name} is not supported.")
        self.classifier_module = classifier_module

        kwargs = deepcopy(self.classifier_kwargs)
        self.classifier = self.classifier_module(**kwargs)

        self.classifier.to(self.device)
        self.post_processor = None
        self.domain_classifier = None
        self.to(self.device)

    def init_prompt_memory(self):
        if self.continuous_prompt_config:
            self.use_prompt = True
            self.prompt_memory = ContinuousPrompt(**self.continuous_prompt_config)
            self.prompt_layer_idx = list(range(0, 12))

    def configure_blocks(self):
        for block in self.blocks:
            setattr(block, "forward", forward_block.__get__(block, block.__class__))
            setattr(
                block.attn,
                "forward",
                forward_attn.__get__(block.attn, block.attn.__class__),
            )
            block.attn.embed_len = (
                self.embed_len
            )  # this is original image token length + cls_token

    def on_task_end(self):
        pass

    def get_memory_modules(self) -> List[PromptMLP]:
        return self.prompt_memory.memory_list

    def on_task_start(
        self, task_id: int, num_classes: int, num_prev_classes: int = None, **kwargs
    ):
        self.current_task_id = task_id
        self.num_classes_sofar += num_classes

    def on_epoch_end(self):
        pass

    def get_prompt(
        self,
        layer_idx: int,
        batched_prompts: torch.Tensor = None,
        tuning: str = "prompt_tuning",
    ):
        # attach prompt
        prompt = batched_prompts[layer_idx]  # bsz, n_token, embed_dim
        if tuning in ["prompt_tuning"]:
            self.len_prompt += prompt.size(1)

        return prompt

    def forward_query(self, x: torch.Tensor, query_embed_replay: torch.Tensor = None):
        # get the query embedding
        with torch.no_grad():
            x = self.patch_embed(x)
            x = self._pos_embed(x)
            if self.query_function == "whole":
                for blk in self.blocks:
                    x, _ = blk(x)
                query_embed = x[:, 0]
            elif self.query_function == "first":
                query_embed, _ = self.blocks[0](x)[:, 0]
            elif self.query_function == "patch_embed":
                query_embed = x[:, 0]
            else:
                raise RuntimeError(f"Not supported patch embed {self.query_function}.")

        return query_embed

    def forward_prompted_features(
        self,
        x: torch.Tensor,
        query_embed: torch.Tensor = None,
    ):
        out = edict()

        batched_prompt = None
        batched_continuous_prompt = self.prompt_memory(query_embed)

        batched_prompt = (
            batched_continuous_prompt
            if batched_prompt is None
            else torch.cat([batched_prompt, batched_continuous_prompt], dim=2)
        )
        out["batched_prompt"] = batched_prompt

        if not self.training:
            out["prompt"] = (
                batched_prompt
                if batched_prompt is None
                else batched_prompt.detach().cpu().permute(1, 0, 2, 3)
            )

        # pass forward over layers
        task_prompt_counter = 0
        attn = list()
        for lyr, block in enumerate(self.blocks):
            kwargs = edict()
            if lyr in self.prompt_layer_idx:
                task_prompt = self.get_prompt(
                    task_prompt_counter, batched_prompt, self.prompting_mode
                )
                task_prompt_counter += 1
                x = torch.cat([task_prompt, x], dim=1)

            x, att = block(x, **kwargs)
            attn.append(att)
        out["attn"] = attn

        return x, out

    def forward_features(
        self,
        x: torch.Tensor,
        original_inference: bool = False,
        get_deep_features: bool = False,
        **kwargs,
    ):
        out = edict()
        x = self.patch_embed(x)
        self.len_img = x.size(1)
        self.len_prompt = 0
        x = self._pos_embed(x)
        if self.use_prompt and not original_inference:
            x, out_prompt = self.forward_prompted_features(
                x, **kwargs
            )  # bsz, n_token, embed_dim
            out.update(out_prompt)
        else:
            if get_deep_features:
                deep_feats = list()
            attn = list()
            for block in self.blocks:
                x, att = block(x)  # bsz, n_token, embed_dim
                attn.append(att)
                if get_deep_features:
                    deep_feats.append(
                        x[:, self.len_prompt : self.len_prompt + 1].unsqueeze(0)
                    )  # cls_token
            out["attn"] = attn
            if get_deep_features:
                out.deep_feats = torch.cat(
                    deep_feats, dim=0
                )  # num_layers, bsz, 1, embed_dim
        x = self.norm(x)  # bsz, n_token, embed_dim

        # pooling the token features output
        if self.classifier_pool == "image":
            x = x[:, self.len_prompt + 1 :].mean(dim=1)
        elif self.classifier_pool == "cls_token":
            x = x[:, self.len_prompt]
        elif self.classifier_pool == "prompt":
            x = x[:, 0 : self.len_prompt].mean(dim=1)
        elif self.classifier_pool == "global":
            x = x.mean(dim=1)
        else:
            raise RuntimeError(f"Pooling {self.classifier_pool} is not supported.")

        return x, out

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False):
        x = self.fc_norm(x)
        if pre_logits:
            return x
        logits = self.classifier(x)
        if self.classifier_name == "cosine":
            logits = logits[0]

        return logits

    def forward(self, x: torch.Tensor, **kwargs):
        out = edict()
        backbone_out, out_forward_feats = self.forward_features(x, **kwargs)
        out.feats = backbone_out
        out.update(out_forward_feats)
        clf_outputs = self.forward_head(backbone_out)
        out.logits = clf_outputs

        if self.return_logits_only:
            return out.logits

        return out

    def nme(self, features: torch.Tensor, prototypes: torch.Tensor):
        num_prototypes = prototypes.size(0) // self.num_classes_sofar
        bsz = features.size(0)

        features = features.cpu().numpy()
        prototypes = prototypes.cpu().numpy()
        # Compute score for iCaRL
        sqd = cdist(prototypes, features, "sqeuclidean")
        score_icarl = torch.from_numpy((-sqd).T)  # bsz, num_classes
        score_icarl = F.softmax(score_icarl, dim=-1)
        score_icarl = score_icarl.reshape(
            bsz, self.num_classes_sofar, num_prototypes
        ).sum(dim=2)

        return score_icarl

    def post_process(self, x):
        if self.post_processor is None:
            return x
        return self.post_processor(x)

    @property
    def features_dim(self):
        return self.embed_dim

    def add_classes(self, n_classes, **kwargs):
        if self.using_dynamic_classifier:
            if self.freeze_old_classifier:
                freeze_module(self.classifier)
                logger.info("Freeze old classifier")

            args = deepcopy(self.classifier_kwargs)
            classes_arg = self.out_cls_key_map[self.classifier_name]
            args[classes_arg] = n_classes

            new_classifier = self.classifier_module(**args)
            new_classifier.to(self.device)
            self.classifier.append(new_classifier)
            logger.info(f"Add new {n_classes} classes to classifier head")
        else:
            if self.freeze_old_classifier:
                freeze_module(self.classifier[self.current_task_id - 1])

    def extract(self, x: torch.Tensor, **kwargs):
        embedding, out = self.forward_features(x, **kwargs)
        out.embedding = embedding
        return out

    def freeze(self, model="all"):
        unfreeze_module(self)
        if model == "all":
            freeze_module(self)
        elif model == "convnet":
            freeze_module(self.patch_embed)
            freeze_module(self._pos_embed)
            freeze_module(self.blocks)
            freeze_module(self.norm)
            freeze_module(self.fc_norm)
        elif model == "classifier":
            freeze_module(self.classifier)
        else:
            assert False, model

        return self

    def get_group_parameters(self):
        groups = {"convnet": self.convnet.parameters()}

        if hasattr(self.classifier, "new_weights"):
            groups["new_weights"] = self.classifier.new_weights
        if hasattr(self.classifier, "old_weights"):
            groups["old_weights"] = self.classifier.old_weights
        if self.rotations_predictor:
            groups["rotnet"] = self.rotations_predictor.parameters()
        if hasattr(self.convnet, "last_block"):
            groups["last_block"] = self.convnet.last_block.parameters()
        if hasattr(self.classifier, "_negative_weights") and isinstance(
            self.classifier._negative_weights, nn.Parameter
        ):
            groups["neg_weights"] = self.classifier._negative_weights
        if self.domain_classifier is not None:
            groups["domain_clf"] = self.domain_classifier.parameters()

        return groups

    def copy(self):
        return copy.deepcopy(self)

    @property
    def n_classes(self):
        return self.classifier.num_classes

    def create_domain_classifier(self):
        self.domain_classifier = DomainClassifier(
            self.convnet.out_dim, device=self.device
        )
        return self.domain_classifier

    def del_domain_classifier(self):
        self.domain_classifier = None
