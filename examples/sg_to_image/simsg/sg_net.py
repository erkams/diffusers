#!/usr/bin/python
#
# Copyright 2018 Google LLC
# Modification copyright 2020 Helisa Dhamo, Azade Farshad
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except impliance with the License.
# You may obtain a copy of the License
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from simsg.graph import GraphTripleConv, GraphTripleConvNet

from transformers import CLIPTextModel, CLIPTokenizer

CLIP_MODEL_PATH = 'stabilityai/stable-diffusion-2'


def _init_weights(module):
    if hasattr(module, 'weight'):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)


class SGNet(nn.Module):
    def __init__(self,
                 vocab,
                 embed_dim=1024,
                 hidden_dim=1024,  # width
                 layers=5,
                 use_box=False,
                 use_depth=False,
                 use_clip=False,
                 tokenizer=None,
                 text_encoder=None,
                 **kwargs):
        super(SGNet, self).__init__()

        if len(kwargs) > 0:
            print('WARNING: Model got unexpected kwargs ', kwargs)

        self.vocab = vocab
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.use_box = use_box
        self.use_depth = use_depth

        self.num_objs = len(vocab['object_idx_to_name'])
        self.num_preds = len(vocab['pred_idx_to_name'])
        self.max_obj = 8

        if use_box:
            obj_embed_dim = embed_dim + 4
        else:
            obj_embed_dim = embed_dim

        if use_clip:
            assert embed_dim == 1024  # CLIP uses 1024-dimensional embeddings

            if tokenizer is None:
                self.tokenizer = CLIPTokenizer.from_pretrained(
                    CLIP_MODEL_PATH, subfolder="tokenizer"
                )
            else:
                self.tokenizer = tokenizer

            if text_encoder is None:
                self.text_encoder = CLIPTextModel.from_pretrained(
                    CLIP_MODEL_PATH, subfolder="text_encoder"
                )
                self.text_encoder.requires_grad_(False)
            else:
                self.text_encoder = text_encoder
        else:
            self.obj_embeddings = nn.Embedding(self.num_objs + 1, embed_dim)
            self.pred_embeddings = nn.Embedding(self.num_preds, embed_dim)

        self.gconv = GraphTripleConv(obj_embed_dim, self.embed_dim, output_dim=embed_dim, hidden_dim=hidden_dim,
                                          pooling='avg',
                                          mlp_normalization='none')
        self.gconv_net = GraphTripleConvNet(obj_embed_dim, self.embed_dim, num_layers=layers, hidden_dim=hidden_dim,
                                            pooling='avg',
                                            mlp_normalization='none')

        self.graph_projection = nn.Linear(embed_dim, embed_dim)
        self.graph_projection.apply(_init_weights)

        self.graph_projection2 = nn.Parameter(torch.randn(1, self.max_obj))
        nn.init.normal_(self.graph_projection2, std=self.max_obj ** -0.5)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # box_net_dim = 4
        # box_net_layers = [obj_embed_dim, hidden_dim, box_net_dim]
        # self.box_net = build_mlp(box_net_layers, batch_norm='none')

    # def encode_image(self, image):
    #
    #     return None

    def encode_sg(self, triplets, objects, boxes=None):
        if self.use_box:
            assert boxes is not None
        assert objects.device == triplets.device

        s, p, o = triplets.chunk(3, dim=1)  # All have shape (num_triples, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (num_triples,)
        edges = torch.stack([s, o], dim=1)  # Shape is (num_triples, 2)

        num_objs = len(objects)

        obj_vecs_ = self.obj_embeddings(objects)

        pred_vecs = self.pred_embeddings(p)
        # print(obj_vecs.shape)
        # print(pred_vecs.shape)

        # GCN pass

        if isinstance(self.gconv, nn.Linear):
            obj_vecs = self.gconv(obj_vecs_)
        else:
            obj_vecs, pred_vecs = self.gconv(obj_vecs_, pred_vecs, edges)
        if self.gconv_net is not None:
            obj_vecs, pred_vecs = self.gconv_net(obj_vecs_, pred_vecs, edges)

        # if self.identity:
        #     obj_vecs = obj_vecs + obj_vecs_

        obj_vecs = F.pad(obj_vecs, pad=(0, 0, self.max_obj - obj_vecs.size(0), 0))

        return obj_vecs

    def forward(self, triplets, objects, image=None, latent=None, boxes=None, depth=None):
        if self.use_depth:
            assert depth is not None
        if self.use_box:
            assert boxes is not None

        image_features = []
        graph_features = []
        for i in range(len(latent)):
            i, g = self._forward(triplets[i], objects[i], image, latent[i], boxes, depth)
            image_features.append(i)
            graph_features.append(g)
        image_features = torch.cat(image_features)
        graph_features = torch.cat(graph_features)

        assert graph_features.shape == (len(latent), self.embed_dim)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        graph_features = graph_features / graph_features.norm(dim=1, keepdim=True)
        print(image_features.shape)
        print(graph_features.shape)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ graph_features.t()
        logits_per_graph = logits_per_image.t()

        # if self.use_depth:
        #     depth_features = self.encode_depth(depth)
        #     depth_features = depth_features / depth_features.norm(dim=1, keepdim=True)
        #     logits_per_depth = logit_scale * depth_features @ graph_features.t()
        #     logits_per_graph = logits_per_graph + logits_per_depth.t()
        #
        #     return logits_per_image, logits_per_text, logits_per_depth

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_graph

    def _forward(self, triplets, objects, image=None, latent=None, boxes=None, depth=None):
        if latent is not None:
            image_features = torch.flatten(latent, start_dim=-2)
            assert image_features.shape[-1] == self.embed_dim
        else:
            raise NotImplementedError

        # image_features = self.encode_image(image)

        graph_embed = self.encode_sg(triplets, objects, boxes)

        graph_features = self.graph_projection(graph_embed)

        graph_features = self.graph_projection2 @ graph_features

        return image_features, graph_features
