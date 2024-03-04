import numpy as np
import json
import os
import cv2

def write_poses_bounds():
    # np.save('poses_bounds.npy')
    inp = '/storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/neuralangelo/data/cecum_t1_a/transforms.json'
    f = open(inp, 'r')
    data = json.load(f)
    img_dir = '/storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/neuralangelo/data/cecum_t1_a/images'
    f.close()

    num_imgs = len(os.listdir(img_dir))
    out = np.zeros((num_imgs, 17))
    f = (data['fl_x']+data['fl_y'])/2
    for i, frame in enumerate(data['frames']):
        bounds = np.array([frame['near'], frame['far']])
        # since its moving, lets just make all the poses identity for now
        pose = np.array([
            [1,0,0,0,data['h']],
            [0,1,0,0,data['w']],
            [0,0,1,0,f]
        ]).flatten()
        row = np.concatenate((pose, bounds), axis=0)
        out[i] = row
    np.save('ct1a/poses_bounds.npy', out)

def create_masks():
    # invert mask
    mask = cv2.imread('ct1a/rgb_mask.png')

    img_dir = '/storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/neuralangelo/data/cecum_t1_a/images'
    num_imgs = len(os.listdir(img_dir))
    for img_name in os.listdir(img_dir):
        # frame_name, color, png = img_name.split('.')
        # mask_name = [frame_name, 'mask', png].join('.')
        cv2.imwrite(os.path.join('ct1a', 'masks', img_name), mask)

depth = cv2.imread('ct1a/depth/0_color_depth.png')
print(depth)