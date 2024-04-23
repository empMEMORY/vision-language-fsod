# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import copy
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet_path', default='datasets/imagenet/annotations/imagenet-21k_image_info.json')
    parser.add_argument('--lvis_path', default='datasets/lvis/lvis_v1_train.json')
    parser.add_argument('--save_categories', default='')
    parser.add_argument('--not_save_imagenet', action='store_true')
    parser.add_argument('--not_save_lvis', action='store_true')
    parser.add_argument('--mark', default='lvis-21k')
    args = parser.parse_args()

    print('Loading', args.imagenet_path)
    in_data = json.load(open(args.imagenet_path, 'r'))
    print('Loading', args.lvis_path)
    lvis_data = json.load(open(args.lvis_path, 'r'))

    categories = copy.deepcopy(lvis_data['categories'])
    cat_count = max(x['id'] for x in categories)
    synset2id = {x['synset']: x['id'] for x in categories}
    name2id = {x['name']: x['id'] for x in categories}
    in_id_map = {}
    for x in in_data['categories']:
        if x['synset'] in synset2id:
            in_id_map[x['id']] = synset2id[x['synset']]
        elif x['name'] in name2id:
            in_id_map[x['id']] = name2id[x['name']]
            x['id'] = name2id[x['name']]
        else:
            cat_count = cat_count + 1
            name2id[x['name']] = cat_count
            in_id_map[x['id']] = cat_count
            x['id'] = cat_count
            categories.append(x)
    
    print('lvis cats', len(lvis_data['categories']))
    print('imagenet cats', len(in_data['categories']))
    print('merge cats', len(categories))

    filtered_images = []
    for x in in_data['images']: