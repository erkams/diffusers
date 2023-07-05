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

from simsg.graph import GraphTripleConv, GraphTripleConvNet

from transformers import CLIPTextModel, CLIPTokenizer

CLIP_MODEL_PATH = 'stabilityai/stable-diffusion-2'

class SGModel(nn.Module):
    """
    SIMSG network. Given a source image and a scene graph, the model generates
    a manipulated image that satisfies the scene graph constellations
    """
    def __init__(self, vocab, embedding_dim=1024,
                 gconv_dim=1024, gconv_hidden_dim=512,
                 gconv_pooling='avg', gconv_num_layers=5, mlp_normalization='none',
                 feat_dims=128, is_baseline=False, is_supervised=False,
                 feats_in_gcn=False, feats_out_gcn=True, 
                 text_encoder=None, tokenizer=None, identity=False, **kwargs):

        super(SGModel, self).__init__()

        if len(kwargs) > 0:
            print('WARNING: Model got unexpected kwargs ', kwargs)

        self.vocab = vocab

        self.feats_in_gcn = feats_in_gcn
        self.feats_out_gcn = feats_out_gcn
        self.is_baseline = is_baseline
        self.is_supervised = is_supervised
        self.identity = identity
        # num_objs = len(vocab['object_idx_to_name'])
        # num_preds = len(vocab['pred_idx_to_name'])
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

        # self.obj_embeddings = nn.Embedding(num_objs + 1, embedding_dim)
        # self.pred_embeddings = nn.Embedding(num_preds, embedding_dim)

        if self.is_baseline or self.is_supervised:
            gconv_input_dims = embedding_dim
        else:
            if self.feats_in_gcn:
                gconv_input_dims = embedding_dim + 4 + feat_dims
            else:
                gconv_input_dims = embedding_dim + 4

        if gconv_num_layers == 0:
            self.gconv = nn.Linear(gconv_input_dims, gconv_dim)
        elif gconv_num_layers > 0:
            gconv_kwargs = {
                'input_dim_obj': gconv_input_dims,
                'input_dim_pred': embedding_dim,
                'output_dim': gconv_dim,
                'hidden_dim': gconv_hidden_dim,
                'pooling': gconv_pooling,
                'mlp_normalization': mlp_normalization,
            }
            self.gconv = GraphTripleConv(**gconv_kwargs)

        self.gconv_net = None
        if gconv_num_layers > 1:
            gconv_kwargs = {
                'input_dim_obj': gconv_dim,
                'input_dim_pred': gconv_dim,
                'hidden_dim': gconv_hidden_dim,
                'pooling': gconv_pooling,
                'num_layers': gconv_num_layers - 1,
                'mlp_normalization': mlp_normalization,
            }
            self.gconv_net = GraphTripleConvNet(**gconv_kwargs)

        if not (self.is_baseline or self.is_supervised):
            if self.feats_in_gcn:
                self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim + 4 + feat_dims)
                if self.feats_out_gcn:
                    self.layer_norm2 = nn.LayerNorm(normalized_shape=gconv_dim + feat_dims)
            else:
                self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim + 4)
                self.layer_norm2 = nn.LayerNorm(normalized_shape=gconv_dim + feat_dims)

        self.p = 0.25
        self.p_box = 0.35

    def enable_embedding(self, text_encoder=None, tokenizer=None, train=False):
        if text_encoder is not None:
            del self.text_encoder
            self.text_encoder = text_encoder
        if tokenizer is not None:
            del self.tokenizer
            self.tokenizer = tokenizer
        if train:
            self.train()
        else:
            self.eval()

    def forward(self, triples, objects=None, boxes_gt=None, max_length=(12, 66), batch_size=1, return_preds=False):
        """
        Encode a scene graph into a vector representation.

        Inputs:
        - triples: LongTensor of shape (num_triples, 3) where triples[t] = [s, p, o]
          means that there is a triple (objs[s], p, objs[o])

        Returns:
        - sg_vec: FloatTensor of shape (sum(max_length), 256) giving a vector representation of the scene graph
        """
        # Compute the text embedding for each object and predicate

        s, p, o = triples.chunk(3, dim=1)  # All have shape (num_triples, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (num_triples,)
        edges = torch.stack([s, o], dim=1)  # Shape is (num_triples, 2)

        # convert objects to torch tensor
        objs = torch.tensor(objects, dtype=torch.long, device=triples.device)
        # print(f'objs shape: {objs.shape}')
        # print(f'pred shape: {p.shape}')
        obj_names = [self.vocab['object_idx_to_name'][i] for i in objs.cpu().numpy()]
        p_names = [self.vocab['pred_idx_to_name'][i] for i in p.cpu().numpy()]
        # print(f'obj names: {len(obj_names)}')
        # print(f'p_names shape: {len(p_names)}')
        
        # s_names = [self.vocab['object_idx_to_name'][i] for i in s.cpu().numpy()]
        # o_names = [self.vocab['object_idx_to_name'][i] for i in o.cpu().numpy()]
        
        num_objs = len(objs)
        obj_to_img = torch.zeros(num_objs, dtype=objs.dtype, device=objs.device)

        # prepare keep indices
        keep_box_idx = torch.ones_like(objs.unsqueeze(1), dtype=torch.float)
        keep_feat_idx = torch.ones_like(objs.unsqueeze(1), dtype=torch.float)
        # keep_image_idx = torch.ones_like(objs.unsqueeze(1), dtype=torch.float)

        def embed_text(text):
            # output shape: (num_objs, 10)
            tokens = self.tokenizer(
                text, max_length=10, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to('cuda:0')

            # output shape: (num_objs, 10, 1024)
            vecs = self.text_encoder(tokens)[0]

            # take average of the tokens to get a overall 1-dim embedding for each objects (num_objs, 1024)
            vecs = vecs.mean(axis=-2)

            return vecs
        
        obj_vecs_ = embed_text(obj_names)
        # print(f'obj_vecs shape: {obj_vecs.shape}')
        if obj_to_img is None:
            obj_to_img = torch.zeros(num_objs, dtype=objs.dtype, device=objs.device)

        if not (self.is_baseline or self.is_supervised):

            box_ones = torch.ones([num_objs, 1], dtype=boxes_gt.dtype, device=boxes_gt.device)
            box_keep, _ = self.prepare_keep_idx(True, box_ones, batch_size, obj_to_img,
                                                         keep_box_idx, keep_feat_idx)

            boxes_prior = boxes_gt * box_keep

            # if random_feats:
            #     # fill with noise the high level visual features, if the feature is masked/dropped
            #     normal_dist = tdist.Normal(loc=get_mean(self.spade_blocks), scale=get_std(self.spade_blocks))
            #     highlevel_noise = normal_dist.sample([high_feats.shape[0]])
            #     feats_prior += highlevel_noise.cuda() * (1 - feats_keep)

            # # when a query image is used to generate an object of the same category
            # if query_feats is not None:
            #     feats_prior[query_idx] = query_feats

            obj_vecs_ = torch.cat([obj_vecs_, boxes_prior], dim=1)
            obj_vecs_ = self.layer_norm(obj_vecs_)

        pred_vecs = embed_text(p_names)
        # print(obj_vecs.shape)
        # print(pred_vecs.shape)
        
        # GCN pass
        
        if isinstance(self.gconv, nn.Linear):
            obj_vecs = self.gconv(obj_vecs_)
        else:
            obj_vecs, pred_vecs = self.gconv(obj_vecs_, pred_vecs, edges)
        if self.gconv_net is not None:
            obj_vecs, pred_vecs = self.gconv_net(obj_vecs_, pred_vecs, edges)

        if self.identity:
            obj_vecs = obj_vecs + obj_vecs_
        
        obj_vecs = F.pad(obj_vecs, pad=(0, 0, max_length[0] - obj_vecs.size(0), 0))

        # if return_preds:
        #     pred_vecs = F.pad(pred_vecs, pad=(0, 0, max_length[1] - pred_vecs.size(0), 0))

        #     # concat vectors
        #     sg_embed = torch.cat([obj_vecs, pred_vecs], dim=0)
        
        #     return sg_embed
        # # resize vector with interpolation
        # sg_embed = F.interpolate(sg_embed.unsqueeze(0), size=(out_shape[-2], out_shape[-1]), mode='bilinear', align_corners=False)
        # sg_embed = sg_embed.squeeze(0)
        # sg_embed = torch.cat([sg_embed] * out_shape[1], dim=0)


        
        return obj_vecs 

    def prepare_keep_idx(self, evaluating, box_ones, num_images, obj_to_img, keep_box_idx,
                         keep_feat_idx, with_feats=True):
        # random drop of boxes and visual feats on training time
        # use objs idx passed as argument on eval time
        imgbox_idx = torch.zeros(num_images, dtype=torch.int64)
        for i in range(num_images):
            imgbox_idx[i] = (obj_to_img == i).nonzero()[-1]

        if evaluating:
            if keep_box_idx is not None:
                box_keep = keep_box_idx
            else:
                box_keep = box_ones

            if with_feats:
                if keep_feat_idx is not None:
                    feats_keep = keep_feat_idx
                else:
                    feats_keep = box_ones
        else:
            # drop random box(es) and feature(s)
            box_keep = F.dropout(box_ones, self.p_box, True, False) * (1 - self.p_box)
            if with_feats:
                feats_keep = F.dropout(box_ones, self.p, True, False) * (1 - self.p)

        # image obj cannot be dropped
        box_keep[imgbox_idx, :] = 1

        if with_feats:
            # image obj feats should not be dropped
            feats_keep[imgbox_idx, :] = 1
            return box_keep, feats_keep

        else:
            return box_keep

