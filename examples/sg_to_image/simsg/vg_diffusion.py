import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import random
import h5py
import json


class VGDiffDatabase(Dataset):
    def __init__(self, vocab, h5_path, image_dir, image_size=256, max_objects=10, max_samples=None,
                 include_relationships=True, use_orphaned_objects=True, use_depth=False):

        with open(vocab, 'r') as f:
            vocab = json.load(f)
        self.image_dir = image_dir
        self.image_size = (image_size, image_size)
        self.vocab = vocab
        self.num_objects = len(vocab['object_idx_to_name'])
        self.use_orphaned_objects = use_orphaned_objects
        self.max_objects = max_objects
        self.max_samples = max_samples
        self.include_relationships = include_relationships
        self.use_depth = use_depth

        self.is_train = True if 'train' in h5_path else False
        if self.is_train:
            transform = [Resize(self.image_size), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]  # augmentation
        else:
            transform = [Resize(self.image_size), transforms.ToTensor()]
        self.transform = transforms.Compose(transform)

        self.data = {}
        with h5py.File(h5_path, 'r', locking=False) as f:
            for k, v in f.items():
                if k == 'image_paths':
                    self.image_paths = list(v)
                else:
                    self.data[k] = torch.IntTensor(np.asarray(v))

    def __len__(self):
        num = self.data['object_names'].size(0)
        if self.max_samples is not None:
            return min(self.max_samples, num)
        return num

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, str(self.image_paths[index], encoding="utf-8"))
        with open(img_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                WW, HH = image.size
                image = self.transform(image.convert('RGB'))

        if self.use_depth:
            depth_path = f'/mnt/workfiles/MegaDepth/vg/{os.path.splitext(self.image_paths[index])[0].decode("utf-8")}.png '
            with open(depth_path, 'rb') as f:
                with PIL.Image.open(f) as depth:
                    depth = self.transform(depth.convert('L'))
        # image = image * 2 - 1

        obj_idxs_with_rels = set()
        obj_idxs_without_rels = set(range(self.data['objects_per_image'][index].item()))
        for r_idx in range(self.data['relationships_per_image'][index]):
            s = self.data['relationship_subjects'][index, r_idx].item()
            o = self.data['relationship_objects'][index, r_idx].item()
            obj_idxs_with_rels.add(s)
            obj_idxs_with_rels.add(o)
            obj_idxs_without_rels.discard(s)
            obj_idxs_without_rels.discard(o)

        obj_idxs = list(obj_idxs_with_rels)
        obj_idxs_without_rels = list(obj_idxs_without_rels)
        if len(obj_idxs) > self.max_objects - 1:
            obj_idxs = random.sample(obj_idxs, self.max_objects)
        if len(obj_idxs) < self.max_objects - 1 and self.use_orphaned_objects:
            num_to_add = self.max_objects - 1 - len(obj_idxs)
            num_to_add = min(num_to_add, len(obj_idxs_without_rels))
            obj_idxs += random.sample(obj_idxs_without_rels, num_to_add)
        O = len(obj_idxs) + 1

        objs = torch.LongTensor(O).fill_(-1)

        boxes = torch.FloatTensor([[0, 0, 1, 1]]).repeat(O, 1)
        obj_idx_mapping = {}
        for i, obj_idx in enumerate(obj_idxs):
            objs[i] = self.data['object_names'][index, obj_idx].item()
            x, y, w, h = self.data['object_boxes'][index, obj_idx].tolist()
            x0 = float(x) / WW
            y0 = float(y) / HH
            x1 = float(x + w) / WW
            y1 = float(y + h) / HH
            boxes[i] = torch.FloatTensor([x0, y0, x1, y1])
            obj_idx_mapping[obj_idx] = i

        objs[O - 1] = self.vocab['object_name_to_idx']['__image__']

        triples = []
        for r_idx in range(self.data['relationships_per_image'][index].item()):
            if not self.include_relationships:
                break
            s = self.data['relationship_subjects'][index, r_idx].item()
            p = self.data['relationship_predicates'][index, r_idx].item()
            o = self.data['relationship_objects'][index, r_idx].item()
            s = obj_idx_mapping.get(s, None)
            o = obj_idx_mapping.get(o, None)
            if s is not None and o is not None:
                triples.append([s, p, o])

        in_image = self.vocab['pred_name_to_idx']['__in_image__']
        for i in range(O - 1):
            triples.append([i, in_image, O - 1])

        triples = torch.LongTensor(triples)
        objects_str = ', '.join([self.vocab['object_idx_to_name'][obj.item()] for obj in objs[:-1]])

        out = {'image': image,
               'objects': objs,
               'boxes': boxes,
               'triplets': triples,
               'objects_str': objects_str}

        if self.use_depth:
            out['depth'] = depth
        return out


class VGTrainDiff(VGDiffDatabase):
    def __init__(self, vocab, h5_path, image_dir, **kwargs):
        super().__init__(vocab=vocab, h5_path=h5_path, image_dir=image_dir, **kwargs)


class VGValidationDiff(VGDiffDatabase):
    def __init__(self, vocab, h5_path, image_dir, **kwargs):
        super().__init__(vocab=vocab, h5_path=h5_path, image_dir=image_dir, **kwargs)


def get_collate_fn(prepare_sg_embeds, tokenize_captions):
    def vg_collate_fn_diff(batch):
        all_imgs, all_objs, all_boxes, all_triples, all_object_str = [], [], [], [], []
        all_depths = []
        # all_obj_to_img, all_triple_to_img = [], []
        # obj_offset = 0
        for sample in batch:
            img, objs, boxes, triples, objects_str = sample['image'], sample['objects'], sample[
                'boxes'], sample['triplets'], sample['objects_str']
            if 'depth' in sample:
                depth = sample['depth']
                all_depths.append(depth)
            all_imgs.append(img)
            all_objs.append(objs)
            all_boxes.append(boxes)
            all_triples.append(triples)
            all_object_str.append(objects_str)

        all_sg_embeds = prepare_sg_embeds({'objects': all_objs, 'boxes': all_boxes, 'triplets': all_triples})
        all_input_ids = tokenize_captions({'objects_str': all_object_str})

        assert all_sg_embeds.ndim == 3 and all_input_ids.ndim == 2

        all_imgs = torch.stack(all_imgs)
        # all_objs = torch.stack(all_objs)
        # all_boxes = torch.stack(all_boxes)
        # all_triples = torch.stack(all_triples)
        # all_obj_to_img = torch.cat(all_obj_to_img)
        # all_triple_to_img = torch.cat(all_triple_to_img)

        # all_sg_embeds = torch.stack(all_sg_embeds)
        # all_input_ids = torch.stack(all_input_ids)

        out = {
            'image': all_imgs,
            'objects': all_objs,
            'boxes': all_boxes,
            'triplets': all_triples,
            'objects_str': all_object_str,
            'sg_embeds': all_sg_embeds,
            'input_ids': all_input_ids
        }

        if len(all_depths) > 0:
            out['depth'] = torch.stack(all_depths)

        return out

    return vg_collate_fn_diff


class Resize(object):
    def __init__(self, size, interp=PIL.Image.BILINEAR):
        if isinstance(size, tuple):
            H, W = size
            self.size = (W, H)
        else:
            self.size = (size, size)
        self.interp = interp

    def __call__(self, img):
        return img.resize(self.size, self.interp)
