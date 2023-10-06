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
                 include_relationships=True, use_orphaned_objects=True, use_depth=False, enrich_sg=False):

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
        self.enrich_sg = enrich_sg

        self.is_train = True if 'train' in h5_path else False
        if self.is_train:
            transform = [Resize(self.image_size), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        else:
            transform = [Resize(self.image_size), transforms.ToTensor()]

        depth_transform = [Resize(self.image_size), transforms.ToTensor()]

        self.transform = transforms.Compose(transform)
        self.depth_transform = transforms.Compose(depth_transform)
        self.normalize = transforms.Normalize(0.5, 0.5)
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

        if self.use_depth or self.enrich_sg:
            depth_path = f'/mnt/workfiles/MegaDepth/vg/{os.path.splitext(self.image_paths[index])[0].decode("utf-8")}.png'
            with open(depth_path, 'rb') as f:
                with PIL.Image.open(f) as depth:
                    depth = self.depth_transform(depth.convert('L'))
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
                # check if the list [s,p,o] is a duplicate in the triples list
                if [s, p, o] not in triples:
                    triples.append([s, p, o])

        if self.enrich_sg:
            extra_triples = enrich_scene_graph(objs, depth, boxes, triples, self.vocab)
            triples += extra_triples

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
            out['depth'] = self.normalize(depth)
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


people = ['man', 'woman', 'girl', 'lady', 'boy', 'child', 'person']

clothes = ['shirt', 'pant', 'jean', 'jacket', 'helmet', 'short', 'coat', 'tie', 'sunglass', 'glasses', 'umbrella',
           'headlight', 'hat', 'shoe', 'sock', 'boot', 'cap', 'glove']

body_parts = ['head', 'hair', 'face', 'leg', 'hand', 'arm', 'ear', 'mouth', 'neck', 'foot', 'nose', 'finger', 'eye']

valid_objects = ['sunglass', 'ball', 'headlight', 'eye', 'sock', 'stone', 'flag', 'boot', 'apple', 'mouth', 'glasses',
                 'orange', 'pot', 'vehicle', 'glove', 'tie', 'finger', 'engine', 'post', 'brick', 'towel', 'cap', 'ski',
                 'basket', 'wood', 'container', 'letter', 'book', 'edge', 'sink', 'kite', 'surfboard', 'plane', 'child',
                 'sheep', 'lamp', 'skateboard', 'photo', 'wing', 'windshield', 'banana', 'nose', 'lady', 'pizza',
                 'seat', 'ceiling', 'cup', 'bottle', 'shelf',
                 'ocean', 'paper', 'frame', 'animal', 'curtain', 'truck', 'motorcycle', 'bird', 'foot', 'helmet',
                 'sand', 'board', 'bear', 'cow', 'beach', 'clock', 'cabinet', 'counter', 'trunk', 'house', 'bed',
                 'bike', 'box', 'bowl', 'cat', 'flower', 'neck', 'coat', 'mountain', 'hill', 'hat', 'wave', 'boat',
                 'zebra', 'food', 'umbrella', 'plant', 'roof', 'dog', 'horse', 'elephant', 'rock', 'shoe', 'pillow',
                 'picture', 'bench', 'glass', 'giraffe', 'jean', 'bus', 'train', 'track', 'bag', 'light', 'pole',
                 'bush', 'people', 'girl', 'short', 'street', 'face', 'snow', 'sidewalk', 'ear', 'arm', 'background',
                 'boy', 'plate', 'field', 'road', 'jacket', 'door', 'floor', 'chair', 'hand', 'fence', 'pant', 'water',
                 'car', 'sign', 'cloud', 'leg', 'table', 'hair', 'woman', 'person', 'grass', 'head', 'ground', 'window',
                 'sky', 'building', 'wall', 'shirt', 'tree', 'man']


def compute_area(box):
    """Compute the area of a bounding box."""
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def compute_mask(box, width, height):
    """Compute a binary mask for a bounding box."""
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)

    mask = torch.zeros(height, width)
    mask[y1:y2, x1:x2] = 1
    return mask


def subtract_overlaps(boxes, width, height):
    """Subtract overlapping regions from the bigger bounding boxes and maintain original indices."""

    # Associate each box with its index and sort by area in descending order
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: compute_area(x[1]), reverse=True)

    # Create masks for each bounding box
    masks = [compute_mask(box, width, height) for box in boxes]

    for i in range(len(indexed_boxes)):
        for j in range(i + 1, len(indexed_boxes)):
            idx_i, box_i = indexed_boxes[i]
            idx_j, box_j = indexed_boxes[j]
            masks[idx_i] -= masks[idx_j] * masks[idx_i]  # Subtract overlapping regions
            masks[idx_i] = torch.clamp(masks[idx_i], 0, 1)  # Ensure the mask values are between 0 and 1

    return masks


def enrich_scene_graph(objects, depth_map, boxes, org_triplets, vocab):
    avg_depths = []

    depth_map = torch.clone(depth_map).squeeze()
    h, w = depth_map.shape

    masks = subtract_overlaps(boxes, w, h)

    for mask in masks:
        average_depth = (depth_map * mask).sum() / mask.sum()
        avg_depths.append(average_depth)

    # Associate each box with its index and sort by area in descending order
    indexed_boxes = list(enumerate(boxes[:-1]))
    indexed_boxes.sort(key=lambda x: compute_area(x[1]), reverse=True)
    triplets = []

    for i in range(len(indexed_boxes)):
        idx_i, box_i = indexed_boxes[i]
        if vocab['object_idx_to_name'][objects[idx_i]] not in valid_objects:
            continue

        cx_i = (box_i[0] + box_i[2]) / 2
        cy_i = (box_i[1] + box_i[3]) / 2

        is_i_cloth = vocab['object_idx_to_name'][objects[idx_i]] in clothes
        is_i_organ = vocab['object_idx_to_name'][objects[idx_i]] in body_parts
        if is_i_cloth or is_i_organ:
            continue
        for j in range(i + 1, len(indexed_boxes)):
            idx_j, box_j = indexed_boxes[j]

            if vocab['object_idx_to_name'][objects[idx_j]] not in valid_objects:
                continue

            cx_j = (box_j[0] + box_j[2]) / 2
            cy_j = (box_j[1] + box_j[3]) / 2

            if compute_area(box_j) < 1 / 100:
                continue
            mask_i, mask_j = compute_mask(box_i, w, h), compute_mask(box_j, w, h)
            depth_i, depth_j = avg_depths[idx_i], avg_depths[idx_j]
            is_j_cloth = vocab['object_idx_to_name'][objects[idx_j]] in clothes
            is_j_organ = vocab['object_idx_to_name'][objects[idx_j]] in body_parts
            is_i_human = vocab['object_idx_to_name'][objects[idx_i]] in people
            if (is_j_cloth or is_j_organ) and not is_i_human:
                continue

            clothes_on_people = is_i_human and is_j_cloth
            organs = is_i_human and is_j_organ

            if clothes_on_people and depth_i - depth_j < 0.2 and (mask_i * mask_j).sum() > 0.95 * mask_j.sum():
                triplets.append([idx_j, 1, idx_i])
            if organs and depth_i - depth_j < 0.2 and (mask_i * mask_j).sum() > 0.95 * mask_j.sum():
                triplets.append([idx_i, 3, idx_j])
            if clothes_on_people or organs:
                continue

            if depth_j - depth_i < 0.02 and (mask_i * mask_j).sum() == mask_j.sum():
                if vocab['object_idx_to_name'][objects[idx_i]] != 'sky' and vocab['object_idx_to_name'][objects[idx_j]] != 'sky':
                    if vocab['object_idx_to_name'][objects[idx_j]] in people:
                        triplets.append([idx_j, 2, idx_i])
                    else:
                        if vocab['object_idx_to_name'][objects[idx_j]] != vocab['object_idx_to_name'][objects[idx_i]]:
                            triplets.append([idx_j, 1, idx_i])
                            for triplet in triplets:
                                if triplet[2] == idx_j:
                                    triplets.remove(triplet)
                    continue
            if abs(depth_i - depth_j) < 0.06 and (mask_i * mask_j).sum() > 0.15 * mask_j.sum():
                if abs(cy_j - cy_i) < 0.2 and abs(cx_j - cx_i) > 0.95 * (box_j[2] - box_j[0]) / 2:
                    if vocab['object_idx_to_name'][objects[idx_i]] != 'sky' and vocab['object_idx_to_name'][objects[idx_j]] != 'sky':
                        triplets.append([idx_j, 2, idx_i])
                        continue
                if cx_j > cx_i and abs(cy_j - cy_i) < 0.2:
                    if vocab['object_idx_to_name'][objects[idx_i]] != 'sky' and vocab['object_idx_to_name'][objects[idx_j]] != 'sky':
                        triplets.append([idx_j, 39, idx_i])
                        continue
                elif cx_i > cx_j and abs(cy_j - cy_i) < 0.2:
                    if vocab['object_idx_to_name'][objects[idx_i]] != 'sky' and vocab['object_idx_to_name'][objects[idx_j]] != 'sky':
                        triplets.append([idx_j, 18, idx_i])
                        continue

            if (mask_i * mask_j).sum() > 0.2 * mask_j.sum() and depth_i < depth_j:
                triplets.append([idx_i, 19, idx_j])
            if (mask_i * mask_j).sum() > 0.2 * mask_j.sum() and depth_i > depth_j:
                for triplet in triplets:
                    if triplet[2] == idx_i and triplet[1] == 25:
                        if [triplet[0], 25, idx_j] in triplets:
                            triplets.remove([triplet[0], 25, idx_j])
                if vocab['object_idx_to_name'][objects[idx_i]] != 'sky' and vocab['object_idx_to_name'][objects[idx_j]] != 'sky':
                    triplets.append([idx_i, 25, idx_j])
    for triplet in triplets[:]:
        if triplet in org_triplets:
            triplets.remove(triplet)

    return triplets
