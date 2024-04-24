# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='datasets/coco/annotations/instances_val2017_unseen_2.json')
    parser.add_argument('--cat_path', default='datasets/coco/annotations/instances_val2017.json')
    args = parser.parse_args()
    print('Loading', args.cat_path)
    cat = json.load(open(args.cat_path, 'r'))['categories']

    print('Loading', args.data_path)
    data = json.load(open(args.data_path, 'r'))
    data['categories'] = cat
    out