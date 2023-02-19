# Part of the code is from https://github.com/xingyizhou/UniDet/blob/master/projects/UniDet/unidet/data/datasets/oid.py
# Copyright (c) Facebook, Inc. and its affiliates.
from .register_oid import register_oid_instances
import os

categories = [
    {'id': 1, 'name': 'Infant bed', 'freebase_id': '/m/061hd_'},
    {'id': 2, 'name': 'Rose', 'freebase_id': '/m/06m11'},
    {'id': 3, 'name': 'Flag', 'freebase_id': '/m/03120'},
    {'id': 4, 'name': 'Flashlight', 'freebase_id': '/m/01kb5b'},
    {'id': 5, 'name': 'Sea turtle', 'freebase_id': '/m/0120dh'},
    {'id': 6, 'name': 'Camera', 'freebase_id': '/m/0dv5r'},
    {'id': 7, 'name': 'Animal', 'freebase_id': '/m/0jbk'},
    {'id': 8, 'name': 'Glove', 'freebase_id': '/m/0174n1'},
    {'id': 9, 'name': 'Crocodile', 'freebase_id': '/m/09f_2'},
    {'id': 10, 'name': 'Cattle', 'freebase_id': '/m/01xq0k1'},
    {'id': 11, 'name': 'House', 'freebase_id': '/m/03jm5'},
    {'id': 12, 'name': 'Guacamole', 'freebase_id': '/m/02g30s'},
    {'id': 13, 'name': 'Penguin', 'freebase_id': '/m/05z6w'},
    {'id': 14, 'name': 'Vehicle registration plate', 'freebase_id': '/m/01jfm_'},
    {'id': 15, 'name': 'Bench', 'freebase_id': '/m/076lb9'},
    {'id': 16, 'name': 'Ladybug', 'freebase_id': '/m/0gj37'},
    {'id': 17, 'name': 'Human nose', 'freebase_id': '/m/0k0pj'},
    {'id': 18, 'name': 'Watermelon', 'freebase_id': '/m/0kpqd'},
    {'id': 19, 'name': 'Flute', 'fre