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
    {'id': 10, 'name': 'Cattle', 'freebase_