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
    {'id': 19, 'name': 'Flute', 'freebase_id': '/m/0l14j_'},
    {'id': 20, 'name': 'Butterfly', 'freebase_id': '/m/0cyf8'},
    {'id': 21, 'name': 'Washing machine', 'freebase_id': '/m/0174k2'},
    {'id': 22, 'name': 'Raccoon', 'freebase_id': '/m/0dq75'},
    {'id': 23, 'name': 'Segway', 'freebase_id': '/m/076bq'},
    {'id': 24, 'name': 'Taco', 'freebase_id': '/m/07crc'},
    {'id': 25, 'name': 'Jellyfish', 'freebase_id': '/m/0d8zb'},
    {'id': 26, 'name': 'Cake', 'freebase_id': '/m/0fszt'},
    {'id': 27, 'name': 'Pen', 'freebase_id': '/m/0k1tl'},
    {'id': 28, 'name': 'Cannon', 'freebase_id': '/m/020kz'},
    {'id': 29, 'name': 'Bread', 'freebase_id': '/m/09728'},
    {'id': 30, 'name': 'Tree', 'freebase_id': '/m/07j7r'},
    {'id': 31, 'name': 'Shellfish', 'freebase_id': '/m/0fbdv'},
    {'id': 32, 'name': 'Bed', 'freebase_id': '/m/03ssj5'},
    {'id': 33, 'name': 'Hamster', 'freebase_id': '/m/03qrc'},
    {'id': 34, 'name': 'Hat', 'freebase_id': '/m/02dl1y'},
    {'id': 35, 'name': 'Toaster', 'freebase_id': '/m/01k6s3'},
    {'id': 36, 'name': 'Sombrero', 'freebase_id': '/m/02jfl0'},
    {'id': 37, 'name': 'Tiara', 'freebase_id': '/m/01krhy'},
    {'id': 38, 'name': 'Bowl', 'freebase_id': '/m/04kkgm'},
    {'id': 39, 'name': 'Dragonfly', 'freebase_id': '/m/0ft9s'},
    {'id': 40, 'name': 'Moths and butterflies', 'freebase_id': '/m/0d_2m'},
    {'id': 41, 'name': 'Antelope', 'freebase_id': '/m/0czz2'},
    {'id': 42, 'name': 'Vegetable', 'freebase_id': '/m/0f4s2w'},
    {'id': 43, 'name': 'Torch', 'freebase_id': '/m/07dd4'},
    {'id': 44, 'name': 'Building', 'freebase_id': '/m/0cgh4'},
    {'id': 45, 'name': 'Power plugs and sockets', 'freebase_id': '/m/03bbps'},
    {'id': 46, 'name': 'Blender', 'freebase_id': '/m/02pjr4'},
    {'id': 47, 'name': 'Billiard table', 'freebase_id': '/m/04p0qw'},
    {'id': 48, 'name': 'Cutting board', 'freebase_id': '/m/02pdsw'},
    {'id': 49, 'name': 'Bronze sculpture', 'freebase_id': '/m/01yx86'},
    {'id': 50, 'name': 'Turtle', 'freebase_id': '/m/09dzg'},
    {'id': 51, 'name': 'Broccoli', 'freebase_id': '/m/0hkxq'},
    {'id': 52, 'name': 'Tiger', 'freebase_id': '/m/07dm6'},
    {'id': 53, 'name': 'Mirror', 'freebase_id': '/m/054_l'},
    {'id': 54, 'name': 'Bear', 'freebase_id': '/m/01dws'},
    {'id': 55, 'name': 'Zucchini', 'freebase_id': '/m/027pcv'},
    {'id': 56, 'name': 'Dress', 'freebase_id': '/m/01d40f'},
    {'id': 57, 'name': 'Volleyball', 'freebase_id': '/m/02rgn06'},
    {'id': 58, 'name': 'Guitar', 'freebase_id': '/m/0342h'},
    {'id': 59, 'name': 'Reptile', 'freebase_id': '/m/06bt6'},
    {'id': 60, 'name': 'Golf cart', 'freebase_id': '/m/0323sq'},
    {'id': 61, 'name': 'Tart', 'freebase_id': '/m/02zvsm'},
    {'id': 62, 'name': 'Fedora', 'freebase_id': '/m/02fq_6'},
    {'id': 63, 'name': 'Carnivore', 'freebase_id': '/m/01lrl'},
    {'id': 64, 'name': 'Car', 'freebase_id': '/m/0k4j'},
    {'id': 65, 'name': 'Lighthouse', 'freebase_id': '/m/04h7h'},
    {'id': 66, 'name': 'Coffeemaker', 'freebase_id': '/m/07xyvk'},
    {'id': 67, 'name': 'Food processor', 'freebase_id': '/m/03y6mg'},
    {'id': 68, 'name': 'Truck', 'freebase_id': '/m/07r04'},
    {'id': 69, 'name': 'Bookcase', 'freebase_id': '/m/03__z0'},
    {'id': 70, 'name': 'Surfboard', 'freebase_id': '/m/019w40'},
    {'id': 71, 'name': 'Footwear', 'freebase_id': '/m/09j5n'},
    {'id': 72, 'name': 'Bench', 'freebase_id': '/m/0cvnqh'},
    {'id': 73, 'name': 'Necklace', 'freebase_id': '/m/01llwg'},
    {'id': 74, 'name': 'Flower', 'freebase_id': '/m/0c9ph5'},
    {'id': 75, 'name': 'Radish', 'freebase_id': '/m/015x5n'},
    {'id': 76, 'name': 'Marine mammal', 'freebase_id': '/m/0gd2v'},
    {'id': 77, 'name': 'Frying pan', 'freebase_id': '/m/04v6l4'},
    {'id': 78, 'name': 'Tap', 'freebase_id': '/m/02jz0l'},
    {'id': 79, 'name': 'Peach', 'freebase_id': '/m/0dj6p'},
    {'id': 80, 'name': 'Knife', 'freebase_id': '/m/04ctx'},
    {'id': 81, 'name': 'Handbag', 'freebase_id': '/m/080hkjn'},
    {'id': 82, 'name': 'Laptop', 'freebase_id': '/m/01c648'},
    {'id': 83, 'name': 'Tent', 'freebase_id': '/m/01j61q'},
    {'id': 84, 'name': 'Ambulance', 'freebase_id': '/m/012n7d'},
    {'id': 85, 'name': 'Christmas tree', 'freebase_id': '/m/025nd'},
    {'id': 86, 'name': 'Eagle', 'freebase_id': '/m/09csl'},
    {'id': 87, 'name': 'Limousine', 'freebase_id': '/m/01lcw4'},
    {'id': 88, 'name': 'Kitchen & dining room table', 'freebase_id': '/m/0h8n5zk'},
    {'id': 89, 'name': 'Polar bear', 'freebase_id': '/m/0633h'},
    {'id': 90, 'name': 'Tower', 'freebase_id': '/m/01fdzj'},
    {'id': 91, 'name': 'Football', 'freebase_id': '/m/01226z'},
    {'id': 92, 'name': 'Willow', 'freebase_id': '/m/0mw_6'},
    {'id': 93, 'name': 'Human head', 'freebase_id': '/m/04hgtk'},
    {'id': 94, 'name': 'Stop sign', 'freebase_id': '/m/02pv19'},
    {'id': 95, 'name': 'Banana', 'freebase_id': '/m/09qck'},
    {'id': 96, 'name': 'Mixer', 'freebase_id': '/m/063rgb'},
    {'id': 97, 'name': 'Binoculars', 'freebase_id': '/m/0lt4_'},
    {'id': 98, 'name': 'Dessert', 'freebase_id': '/m/0270h'},
    {'id': 99, 'name': 'Bee', 'freebase_id': '/m/01h3n'},
    {'id': 100, 'name': 'Chair', 'freebase_id': '/m/01mzpv'},
    {'id': 101, 'name': 'Wood-burning stove', 'freebase_id': '/m/04169hn'},
    {'id': 102, 'name': 'Flowerpot', 'freebase_id': '/m/0fm3zh'},
    {'id': 103, 'name': 'Beaker', 'freebase_id': '/m/0d20w4'},
    {'id': 104, 'name': 'Oyster', 'freebase_id': '/m/0_cp5'},
    {'id': 105, 'name': 'Woodpecker', 'freebase_id': '/m/01dy8n'},
    {'id': 106, 'name': 'Harp', 'freebase_id': '/m/03m5k'},
    {'id': 107, 'name': 'Bathtub', 'freebase_id': '/m/03dnzn'},
    {'id': 108, 'name': 'Wall clock', 'freebase_id': '/m/0h8mzrc'},
    {'id': 109, 'name': 'Sports uniform', 'freebase_id': '/m/0h8mhzd'},
    {'id': 110, 'name': 'Rhinoceros', 'freebase_id': '/m/03d443'},
    {'id': 111, 'name': 'Beehive', 'freebase_id': '/m/01gllr'},
    {'id': 112, 'name': 'Cupboard', 'freebase_id': '/m/0642b4'},
    {'id': 113, 'name': 'Chicken', 'freebase_id': '/m/09b5t'},
    {'id': 114, 'name': 'Man', 'freebase_id': '/m/04yx4'},
    {'id': 115, 'name': 'Blue jay', 'freebase_id': '/m/01f8m5'},
    {'id': 116, 'name': 'Cucumber', 'freebase_id': '/m/015x4r'},
    {'id': 117, 'name': 'Balloon', 'freebase_id': '/m/01j51'},
    {'id': 118, 'name': 'Kite', 'freebase_id': '/m/02zt3'},
    {'id': 119, 'name': 'Fireplace', 'freebase_id': '/m/03tw93'},
    {'id': 120, 'name': 'Lantern', 'freebase_id': '/m/01jfsr'},
    {'id': 121, 'name': 'Missile', 'freebase_id': '/m/04ylt'},
    {'id': 122, 'name': 'Book', 'freebase_id': '/m/0bt_c3'},
    {'id': 123, 'name': 'Spoon', 'freebase_id': '/m/0cmx8'},
    {'id': 124, 'name': 'Grapefruit', 'freebase_id': '/m/0hqkz'},
    {'id': 125, 'name': 'Squirrel', 'freebase_id': '/m/071qp'},
    {'id': 126, 'name': 'Orange', 'freebase_id': '/m/0cyhj_'},
    {'id': 127, 'name': 'Coat', 'freebase_id': '/m/01xygc'},
    {'id': 128, 'name': 'Punching bag', 'freebase_id': '/m/0420v5'},
    {'id': 129, 'name': 'Zebra', 'freebase_id': '/m/0898b'},
    {'id': 130, 'name': 'Billboard', 'freebase_id': '/m/01knjb'},
    {'id': 131, 'name': 'Bicycle', 'freebase_id': '/m/0199g'},
    {'id': 132, 'name': 'Door handle', 'freebase_id': '/m/03c7gz'},
    {'id': 133, 'name': 'Mechanical fan', 'freebase_id': '/m/02x984l'},
    {'id': 134, 'name': 'Ring binder', 'freebase_id': '/m/04zwwv'},
    {'id': 135, 'name': 'Table', 'freebase_id': '/m/04bcr3'},
    {'id': 136, 'name': 'Parrot', 'freebase_id': '/m/0gv1x'},
    {'id': 137, 'name': 'Sock', 'freebase_id': '/m/01nq26'},
    {'id': 138, 'name': 'Vase', 'freebase_id': '/m/02s195'},
    {'id': 139, 'name': 'Weapon', 'freebase_id': '/m/083kb'},
    {'id': 140, 'name': 'Shotgun', 'freebase_id': '/m/06nrc'},
    {'id': 141, 'name': 'Glasses', 'freebase_id': '/m/0jyfg'},
    {'id': 142, 'name': 'Seahorse', 'freebase_id': '/m/0nybt'},
    {'id': 143, 'name': 'Belt', 'freebase_id': '/m/0176mf'},
    {'id': 144, 'name': 'Watercraft', 'freebase_id': '/m/01rzcn'},
    {'id': 145, 'name': 'Window', 'freebase_id': '/m/0d4v4'},
    {'id': 146, 'name': 'Giraffe', 'freebase_id': '/m/03bk1'},
    {'id': 147, 'name': 'Lion', 'freebase_id': '/m/096mb'},
    {'id': 148, 'name': 'Tire', 'freebase_id': '/m/0h9mv'},
    {'id': 149, 'name': 'Vehicle', 'freebase_id': '/m/07yv9'},
    {'id': 150, 'name': 'Canoe', 'freebase_id': '/m/0ph39'},
    {'id': 151, 'name': 'Tie', 'freebase_id': '/m/01rkbr'},
    {'id': 152, 'name': 'Shelf', 'freebase_id': '/m/0gjbg72'},
    {'id': 153, 'name': 'Picture frame', 'freebase_id': '/m/06z37_'},
    {'id': 154, 'name': 'Printer', 'freebase_id': '/m/01m4t'},
    {'id': 155, 'name': 'Human leg', 'freebase_id': '/m/035r7c'},
    {'id': 156, 'name': 'Boat', 'freebase_id': '/m/019jd'},
    {'id': 157, 'name': 'Slow cooker', 'freebase_id': '/m/02tsc9'},
    {'id': 158, 'name': 'Croissant', 'freebase_id': '/m/015wgc'},
    {'id': 159, 'name': 'Candle', 'freebase_id': '/m/0c06p'},
    {'id': 160, 'name': 'Pancake', 'freebase_id': '/m/01dwwc'},
    {'id': 161, 'name': 'Pillow', 'freebase_id': '/m/034c16'},
    {'id': 162, 'name': 'Coin', 'freebase_id': '/m/0242l'},
    {'id': 163, 'name': 'Stretcher', 'freebase_id': '/m/02lbcq'},
    {'id': 164, 'name': 'Sandal', 'freebase_id': '/m/03nfch'},
    {'id': 165, 'name': 'Woman', 'freebase_id': '/m/03bt1vf'},
    {'id': 166, 'name': 'Stairs', 'freebase_id': '/m/01lynh'},
    {'id': 167, 'name': 'Harpsichord', 'freebase_id': '/m/03q5t'},
    {'id': 168, 'name': 'Stool', 'freebase_id': '/m/0fqt361'},
    {'id': 169, 'name': 'Bus', 'freebase_id': '/m/01bjv'},
    {'id': 170, 'name': 'Suitcase', 'freebase_id': '/m/01s55n'},
    {'id': 171, 'name': 'Human mouth', 'freebase_id': '/m/0283dt1'},
    {'id': 172, 'name': 'Juice', 'freebase_id': '/m/01z1kdw'},
    {'id': 173, 'name': 'Skull', 'freebase_id': '/m/016m2d'},
    {'id': 174, 'name': 'Door', 'freebase_id': '/m/02dgv'},
    {'id': 175, 'name': 'Violin', 'freebase_id': '/m/07y_7'},
    {'id': 176, 'name': 'Chopsticks', 'freebase_id': '/m/01_5g'},
    {'id': 177, 'name': 'Digital clock', 'freebase_id': '/m/06_72j'},
    {'id': 178, 'name': 'Sunflower', 'freebase_id': '/m/0ftb8'},
    {'id': 179, 'name': 'Leopard', 'freebase_id': '/m/0c29q'},
    {'id': 180, 'name': 'Bell pepper', 'freebase_id': '/m/0jg57'},
    {'id': 181, 'name': 'Harbor seal', 'freebase_id': '/m/02l8p9'},
    {'id': 182, 'name': 'Snake', 'freebase_id': '/m/078jl'},
    {'id': 183, 'name': 'Sewing machine', 'freebase_id': '/m/0llzx'},
    {'id': 184, 'name': 'Goose', 'freebase_id': '/m/0dbvp'},
    {'id': 185, 'name': 'Helicopter', 'freebase_id': '/m/09ct_'},
    {'id': 186, 'name': 'Seat belt', 'freebase_id': '/m/0dkzw'},
    {'id': 187, 'name': 'Coffee cup', 'freebase_id': '/m/02p5f1q'},
    {'id': 188, 'name': 'Microwave oven', 'freebase_id': '/m/0fx9l'},
    {'id': 189, 'name': 'Hot dog', 'freebase_id': '/m/01b9xk'},
    {'id': 190, 'name': 'Countertop', 'freebase_id': '/m/0b3fp9'},
    {'id': 191, 'name': 'Serving tray', 'freebase_id': '/m/0h8n27j'},
    {'id': 192, 'name': 'Dog bed', 'freebase_id': '/m/0h8n6f9'},
    {'id': 193, 'name': 'Beer', 'freebase_id': '/m/01599'},
    {'id': 194, 'name': 'Sunglasses', 'freebase_id': '/m/017ftj'},
    {'id': 195, 'name': 'Golf ball', 'freebase_id': '/m/044r5d'},
    {'id': 196, 'name': 'Waffle', 'freebase_id': '/m/01dwsz'},
    {'id': 197, 'name': 'Palm tree', 'freebase_id': '/m/0cdl1'},
    {'id': 198, 'name': 'Trumpet', 'freebase_id': '/m/07gql'},
    {'id': 199, 'name': 'Ruler', 'freebase_id': '/m/0hdln'},
    {'id': 200, 'name': 'Helmet', 'freebase_id': '/m/0zvk5'},
    {'id': 201, 'name': 'Ladder', 'freebase_id': '/m/012w5l'},
    {'id': 202, 'name': 'Office building', 'freebase_id': '/m/021sj1'},
    {'id': 203, 'name': 'Tablet computer', 'freebase_id': '/m/0bh9flk'},
    {'id': 204, 'name': 'Toilet paper', 'freebase_id': '/m/09gtd'},
    {'id': 205, 'name': 'Pomegranate', 'freebase_id': '/m/0jwn_'},
    {'id': 206, 'name': 'Skirt', 'freebase_id': '/m/02wv6h6'},
    {'id': 207, 'name': 'Gas stove', 'freebase_id': '/m/02wv84t'},
    {'id': 208, 'name': 'Cookie', 'freebase_id': '/m/021mn'},
    {'id': 209, 'name': 'Cart', 'freebase_id': '/m/018p4k'},
    {'id': 210, 'name': 'Raven', 'freebase_id': '/m/06j2d'},
    {'id': 211, 'name': 'Egg', 'freebase_id': '/m/033cnk'},
    {'id': 212, 'name': 'Burrito', 'freebase_id': '/m/01j3zr'},
    {'id': 213, 'name': 'Goat', 'freebase_id': '/m/03fwl'},
    {'id': 214, 'name': 'Kitchen knife', 'freebase_id': '/m/058qzx'},
    {'id': 215, 'name': 'Skateboard', 'freebase_id': '/m/06_fw'},
    {'id': 216, 'name': 'Salt and pepper shakers', 'freebase_id': '/m/02x8cch'},
    {'id': 217, 'name': 'Lynx', 'freebase_id': '/m/04g2r'},
    {'id': 218, 'name': 'Boot', 'freebase_id': '/m/01b638'},
    {'id': 219, 'name': 'Platter', 'freebase_id': '/m/099ssp'},
    {'id': 220, 'name': 'Ski', 'freebase_id': '/m/071p9'},
    {'id': 221, 'name': 'Swimwear', 'freebase_id': '/m/01gkx_'},
    {'id': 222, 'name': 'Swimming pool', 'freebase_id': '/m/0b_rs'},
    {'id': 223, 'name': 'Drinking straw', 'freebase_id': '/m/03v5tg'},
    {'id': 224, 'name': 'Wrench', 'freebase_id': '/m/01j5ks'},
    {'id': 225, 'name': 'Drum', 'freebase_id': '/m/026t6'},
    {'id': 226, 'name': 'Ant', 'freebase_id': '/m/0_k2'},
    {'id': 227, 'name': 'Human ear', 'freebase_id': '/m/039xj_'},
    {'id': 228, 'name': 'Headphones', 'freebase_id': '/m/01b7fy'},
    {'id': 229, 'name': 'Fountain', 'freebase_id': '/m/0220r2'},
    {'id': 230, 'name': 'Bird', 'freebase_id': '/m/015p6'},
    {'id': 231, 'name': 'Jeans', 'freebase_id': '/m/0fly7'},
    {'id': 232, 'name': 'Television', 'freebase_id': '/m/07c52'},
    {'id': 233, 'name': 'Crab', 'freebase_id': '/m/0n28_'},
    {'id': 234, 'name': 'Microphone', 'freebase_id': '/m/0hg7b'},
    {'id': 235, 'name': 'Home appliance', 'freebase_id': '/m/019dx1'},
    {'id': 236, 'name': 'Snowplow', 'freebase_id': '/m/04vv5k'},
    {'id': 237, 'name': 'Beetle', 'freebase_id': '/m/020jm'},
    {'id': 238, 'name': 'Artichoke', 'freebase_id': '/m/047v4b'},
    {'id': 239, 'name': 'Jet ski', 'freebase_id': '/m/01xs3r'},
    {'id': 240, 'name': 'Stationary bicycle', 'freebase_id': '/m/03kt2w'},
    {'id': 241, 'name': 'Human hair', 'freebase_id': '/m/03q69'},
    {'id': 242, 'name': 'Brown bear', 'freebase_id': '/m/01dxs'},
    {'id': 243, 'name': 'Starfish', 'freebase_id': '/m/01h8tj'},
    {'id': 244, 'name': 'Fork', 'freebase_id': '/m/0dt3t'},
    {'id': 245, 'name': 'Lobster', 'freebase_id': '/m/0cjq5'},
    {'id': 246, 'name': 'Corded phone', 'freebase_id': '/m/0h8lkj8'},
    {'id': 247, 'name': 'Drink', 'freebase_id': '/m/0271t'},
    {'id': 248, 'name': 'Saucer', 'freebase_id': '/m/03q5c7'},
    {'id': 249, 'name': 'Carrot', 'freebase_id': '/m/0fj52s'},
    {'id': 250, 'name': 'Insect', 'freebase_id': '/m/03vt0'},
    {'id': 251, 'name': 'Clock', 'freebase_id': '/m/01x3z'},
    {'id': 252, 'name': 'Castle', 'freebase_id': '/m/0d5gx'},
    {'id': 253, 'name': 'Tennis racket', 'freebase_id': '/m/0h8my_4'},
    {'id': 254, 'name': 'Ceiling fan', 'freebase_id': '/m/03ldnb'},
    {'id': 255, 'name': 'Asparagus', 'freebase_id': '/m/0cjs7'},
    {'id': 256, 'name': 'Jaguar', 'freebase_id': '/m/0449p'},
    {'id': 257, 'name': 'Musical instrument', 'freebase_id': '/m/04szw'},
    {'id': 258, 'name': 'Train', 'freebase_id': '/m/07jdr'},
    {'id': 259, 'name': 'Cat', 'freebase_id': '/m/01yrx'},
    {'id': 260, 'name': 'Rifle', 'freebase_id': '/m/06c54'},
    {'id': 261, 'name': 'Dumbbell', 'freebase_id': '/m/04h8sr'},
    {'id': 262, 'name': 'Mobile phone', 'freebase_id': '/m/050k8'},
    {'id': 263, 'name': 'Taxi', 'freebase_id': '/m/0pg52'},
    {'id': 264, 'name': 'Shower', 'freebase_id': '/m/02f9f_'},
    {'id': 265, 'name': 'Pitcher', 'freebase_id': '/m/054fyh'},
    {'id': 266, 'name': 'Lemon', 'freebase_id': '/m/09k_b'},
    {'id': 267, 'name': 'Invertebrate', 'freebase_id': '/m/03xxp'},
    {'id': 268, 'name': 'Turkey', 'freebase_id': '/m/0jly1'},
    {'id': 269, 'name': 'High heels', 'freebase_id': '/m/06k2mb'},
    {'id': 270, 'name': 'Bust', 'freebase_id': '/m/04yqq2'},
    {'id': 271, 'name': 'Elephant', 'freebase_id': '/m/0bwd_0j'},
    {'id': 272, 'name': 'Scarf', 'freebase_id': '/m/02h19r'},
    {'id': 273, 'name': 'Barrel', 'freebase_id': '/m/02zn6n'},
    {'id': 274, 'name': 'Trombone', 'freebase_id': '/m/07c6l'},
    {'id': 275, 'name': 'Pumpkin', 'freebase_id': '/m/05zsy'},
    {'id': 276, 'name': 'Box', 'freebase_id': '/m/025dyy'},
    {'id': 277, 'name': 'Tomato', 'freebase_id': '/m/07j87'},
    {'id': 278, 'name': 'Frog', 'freebase_id': '/m/09ld4'},
    {'id': 279, 'name': 'Bidet', 'freebase_id': '/m/01vbnl'},
    {'id': 280, 'name': 'Human face', 'freebase_id': '/m/0dzct'},
    {'id': 281, 'name': 'Houseplant', 'freebase_id': '/m/03fp41'},
    {'id': 282, 'name': 'Van', 'freebase_id': '/m/0h2r6'},
    {'id': 283, 'name': 'Shark', 'freebase_id': '/m/0by6g'},
    {'id': 284, 'name': 'Ice cream', 'freebase_id': '/m/0cxn2'},
    {'id': 285, 'name': 'Swim cap', 'freebase_id': '/m/04tn4x'},
    {'id': 286, 'name': 'Falcon', 'freebase_id': '/m/0f6wt'},
    {'id': 287, 'name': 'Ostrich', 'freebase_id': '/m/05n4y'},
    {'id': 288, 'name': 'Handgun', 'freebase_id': '/m/0gxl3'},
    {'id': 289, 'name': 'Whiteboard', 'freebase_id': '/m/02d9qx'},
    {'id': 290, 'name': 'Lizard', 'freebase_id': '/m/04m9y'},
    {'id': 291, 'name': 'Pasta', 'freebase_id': '/m/05z55'},
    {'id': 292, 'name': 'Snowmobile', 'freebase_id': '/m/01x3jk'},
    {'id': 293, 'name': 'Light bulb', 'freebase_id': '/m/0h8l4fh'},
    {'id': 294, 'name': 'Window blind', 'freebase_id': '/m/031b6r'},
    {'id': 295, 'name': 'Muffin', 'freebase_id': '/m/01tcjp'},
    {'id': 296, 'name': 'Pretzel', 'freebase_id': '/m/01f91_'},
    {'id': 297, 'name': 'Computer monitor', 'freebase_id': '/m/02522'},
    {'id': 298, 'name': 'Horn', 'freebase_id': '/m/0319l'},
    {'id': 299, 'name': 'Furniture', 'freebase_id': '/m/0c_jw'},
    {'id': 300, 'name': 'Sandwich', 'freebase_id': '/m/0l515'},
    {'id': 301, 'name': 'Fox', 'freebase_id': '/m/0306r'},
    {'id': 302, 'name': 'Convenience store', 'freebase_id': '/m/0crjs'},
    {'id': 303, 'name': 'Fish', 'freebase_id': '/m/0ch_cf'},
    {'id': 304, 'name': 'Fruit', 'freebase_id': '/m/02xwb'},
    {'id': 305, 'name': 'Earrings', 'freebase_id': '/m/01r546'},
    {'id': 306, 'name': 'Curtain', 'freebase_id': '/m/03rszm'},
    {'id': 307, 'name': 'Grape', 'freebase_id': '/m/0388q'},
    {'id': 308, 'name': 'Sofa bed', 'freebase_id': '/m/03m3pdh'},
    {'id': 309, 'name': 'Horse', 'freebase_id': '/m/03k3r'},
    {'id': 310, 'name': 'Luggage and bags', 'freebase_id': '/m/0hf58v5'},
    {'id': 311, 'name': 'Desk', 'freebase_id': '/m/01y9k5'},
    {'id': 312, 'name': 'Crutch', 'freebase_id': '/m/05441v'},
    {'id': 313, 'name': 'Bicycle helmet', 'freebase_id': '/m/03p3bw'},
    {'id': 314, 'name': 'Tick', 'freebase_id': '/m/0175cv'},
    {'id': 315, 'name': 'Airplane', 'freebase_id': '/m/0cmf2'},
    {'id': 316, 'name': 'Canary', 'freebase_id': '/m/0ccs93'},
    {'id': 317, 'name': 'Spatula', 'freebase_id': '/m/02d1br'},
    {'id': 318, 'name': 'Watch', 'freebase_id': '/m/0gjkl'},
    {'id': 319, 'name': 'Lily', 'freebase_id': '/m/0jqgx'},
    {'id': 320, 'name': 'Kitchen appliance', 'freebase_id': '/m/0h99cwc'},
    {'id': 321, 'name': 'Filing cabinet', 'freebase_id': '/m/047j0r'},
    {'id': 322, 'name': 'Aircraft', 'freebase_id': '/m/0k5j'},
    {'id': 323, 'name': 'Cake stand', 'freebase_id': '/m/0h8n6ft'},
    {'id': 324, 'name': 'Candy', 'freebase_id': '/m/0gm28'},
    {'id': 325, 'name': 'Sink', 'freebase_id': '/m/0130jx'},
    {'id': 326, 'name': 'Mouse', 'freebase_id': '/m/04rmv'},
    {'id': 327, 'name': 'Wine', 'freebase_id': '/m/081qc'},
    {'id': 328, 'name': 'Wheelchair', 'freebase_id': '/m/0qmmr'},
    {'id': 329, 'name': 'Goldfish', 'freebase_id': '/m/03fj2'},
    {'id': 330, 'name': 'Refrigerator', 'freebase_id': '/m/040b_t'},
    {'id': 331, 'name': 'French fries', 'freebase_id': '/m/02y6n'},
    {'id': 332, 'name': 'Drawer', 'freebase_id': '/m/0fqfqc'},
    {'id': 333, 'name': 'Treadmill', 'freebase_id': '/m/030610'},
    {'id': 334, 'name': 'Picnic basket', 'freebase_id': '/m/07kng9'},
    {'id': 335, 'name': 'Dice', 'freebase_id': '/m/029b3'},
    {'id': 336, 'name': 'Cabbage', 'freebase_id': '/m/0fbw6'},
    {'id': 337, 'name': 'Football helmet', 'freebase_id': '/m/07qxg_'},
    {'id': 338, 'name': 'Pig', 'freebase_id': '/m/068zj'},
    {'id': 339, 'name': 'Person', 'freebase_id': '/m/01g317'},
    {'id': 340, 'name': 'Shorts', 'freebase_id': '/m/01bfm9'},
    {'id': 341, 'name': 'Gondola', 'freebase_id': '/m/02068x'},
    {'id': 342, 'name': 'Honeycomb', 'freebase_id': '/m/0fz0h'},
    {'id': 343, 'name': 'Doughnut', 'freebase_id': '/m/0jy4k'},
    {'id': 344, 'name': 'Chest of drawers', 'freebase_id': '/m/05kyg_'},
    {'id': 345, 'name': 'Land vehicle', 'freebase_id': '/m/01prls'},
    {'id': 346, 'name': 'Bat', 'freebase_id': '/m/01h44'},
    {'id': 347, 'name': 'Monkey', 'freebase_id': '/m/08pbxl'},
    {'id': 348, 'name': 'Dagger', 'freebase_id': '/m/02gzp'},
    {'id': 349, 'name': 'Tableware', 'freebase_id': '/m/04brg2'},
    {'id': 350, 'name': 'Human foot', 'freebase_id': '/m/031n1'},
    {'id': 351, 'name': 'Mug', 'freebase_id': '/m/02jvh9'},
    {'id': 352, 'name': 'Alarm clock', 'freebase_id': '/m/046dlr'},
    {'id': 353, 'name': 'Pressure cooker', 'freebase_id': '/m/0h8ntjv'},
    {'id': 354, 'name': 'Human hand', 'freebase_id': '/m/0k65p'},
    {'id': 355, 'name': 'Tortoise', 'freebase_id': '/m/011k07'},
    {'id': 356, 'name': 'Baseball glove', 'freebase_id': '/m/03grzl'},
    {'id': 357, 'name': 'Sword', 'freebase_id': '/m/06y5r'},
    {'id': 358, 'name': 'Pear', 'freebase_id': '/m/061_f'},
    {'id': 359, 'name': 'Miniskirt', 'freebase_id': '/m/01cmb2'},
    {'id': 360, 'name': 'Traffic sign', 'freebase_id': '/m/01mqdt'},
    {'id': 361, 'name': 'Girl', 'freebase_id': '/m/05r655'},
    {'id': 362, 'name': 'Roller skates', 'freebase_id': '/m/02p3w7d'},
    {'id': 363, 'name': 'Dinosaur', 'freebase_id': '/m/029tx'},
    {'id': 364, 'name': 'Porch', 'freebase_id': '/m/04m6gz'},
    {'id': 365, 'name': 'Human beard', 'freebase_id': '/m/015h_t'},
    {'id': 366, 'name': 'Submarine sandwich', 'freebase_id': '/m/06pcq'},
    {'id': 367, 'name': 'Screwdriver', 'freebase_id': '/m/01bms0'},
    {'id': 368, 'name': 'Strawberry', 'freebase_id': '/m/07fbm7'},
    {'id': 369, 'name': 'Wine glass', 'freebase_id': '/m/09tvcd'},
    {'id': 370, 'name': 'Seafood', 'freebase_id': '/m/06nwz'},
    {'id': 371, 'name': 'Racket', 'freebase_id': '/m/0dv9c'},
    {'id': 372, 'name': 'Wheel', 'freebase_id': '/m/083wq'},
    {'id': 373, 'name': 'Sea lion', 'freebase_id': '/m/0gd36'},
    {'id': 374, 'name': 'Toy', 'freebase_id': '/m/0138tl'},
    {'id': 375, 'name': 'Tea', 'freebase_id': '/m/07clx'},
    {'id': 376, 'name': 'Tennis ball', 'freebase_id': '/m/05ctyq'},
    {'id': 377, 'name': 'Waste container', 'freebase_id': '/m/0bjyj5'},
    {'id': 378, 'name': 'Mule', 'freebase_id': '/m/0dbzx'},
    {'id': 379, 'name': 'Cricket ball', 'freebase_id': '/m/02ctlc'},
    {'id': 380, 'name': 'Pineapple', 'freebase_id': '/m/0fp6w'},
    {'id': 381, 'name': 'Coconut', 'freebase_id': '/m/0djtd'},
    {'id': 382, 'name': 'Doll', 'freebase_id': '/m/0167gd'},
    {'id': 383, 'name': 'Coffee table', 'freebase_id': '/m/078n6m'},
    {'id': 384, 'name': 'Snowman', 'freebase_id': '/m/0152hh'},
    {'id': 385, 'name': 'Lavender', 'freebase_id': '/m/04gth'},
    {'id': 386, 'name': 'Shrimp', 'freebase_id': '/m/0ll1f78'},
    {'id': 387, 'name': 'Maple', 'freebase_id': '/m/0cffdh'},
    {'id': 388, 'name': 'Cowboy hat', 'freebase_id': '/m/025rp__'},
    {'id': 389, 'name': 'Goggles', 'freebase_id': '/m/02_n6y'},
    {'id': 390, 'name': 'Rugby ball', 'freebase_id': '/m/0wdt60w'},
    {'id': 391, 'name': 'Caterpillar', 'freebase_id': '/m/0cydv'},
    {'id': 392, 'name': 'Poster', 'freebase_id': '/m/01n5jq'},
    {'id': 393, 'name': 'Rocket', 'freebase_id': '/m/09rvcxw'},
    {'id': 394, 'name': 'Organ', 'freebase_id': '/m/013y1f'},
    {'id': 395, 'name': 'Saxophone', 'freebase_id': '/m/06ncr'},
    {'id': 396, 'name': 'Traffic light', 'freebase_id': '/m/015qff'},
    {'id': 397, 'name': 'Cocktail', 'freebase_id': '/m/024g6'},
    {'id': 398, 'name': 'Plastic bag', 'freebase_id': '/m/05gqfk'},
    {'id': 399, 'name': 'Squash', 'freebase_id': '/m/0dv77'},
    {'id': 400, 'name': 'Mushroom', 'freebase_id': '/m/052sf'},
    {'id': 401, 'name': 'Hamburger', 'freebase_id': '/m/0cdn1'},
    {'id': 402, 'name': 'Light switch', 'freebase_id': '/m/03jbxj'},
    {'id': 403, 'name': 'Parachute', 'freebase_id': '/m/0cyfs'},
    {'id': 404, 'name': 'Teddy bear', 'freebase_id': '/m/0kmg4'},
    {'id': 405, 'name': 'Winter melon', 'freebase_id': '/m/02cvgx'},
    {'id': 406, 'name': 'Deer', 'freebase_id': '/m/09kx5'},
    {'id': 407, 'name': 'Musical keyboard', 'freebase_id': '/m/057cc'},
    {'id': 408, 'name': 'Plumbing fixture', 'freebase_id': '/m/02pkr5'},
    {'id': 409, 'name': 'Scoreboard', 'freebase_id': '/m/057p5t'},
    {'id'