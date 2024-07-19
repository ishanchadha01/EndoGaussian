import numpy as np
import json
import os
import cv2
from tqdm import tqdm

def write_poses_bounds(dataset_name):
    # np.save('poses_bounds.npy')
    base_dir = '/storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/REIM-NeRF/data_preprocessed'
    inp = os.path.join(base_dir, dataset_name, 'transforms_test.json')
    # inp = f'/storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/neuralangelo/data/{dataset_name}/transforms.json'
    f = open(inp, 'r')
    data = json.load(f)
    img_dir = os.path.join(base_dir, dataset_name, 'images')
    print(img_dir)
    # img_dir = f'/storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/neuralangelo/data/{dataset_name}/images'
    f.close()

    num_imgs = len(os.listdir(img_dir))
    out = np.zeros((num_imgs, 17))
    f = (data['fx']+data['fy'])/2
    for i, frame in tqdm(enumerate(data['frames'])):
        bounds = np.array([frame['near'], frame['far']])
        # since its moving, lets just make all the poses identity for now
        # pose = np.array([
        #     [1,0,0,0,data['h']],
        #     [0,1,0,0,data['w']],
        #     [0,0,1,0,f]
        # ]).flatten()
        pose = np.array(frame['transform_matrix']).astype(np.float32)[:3, :4] # using this line instead on ct1a to see if using actual pose will work
        ext = np.array([data['h'], data['w'], f]).reshape(3,-1)
        pose = np.concatenate((pose, ext), axis=1).flatten()

        row = np.concatenate((pose, bounds), axis=0)
        out[i] = row
    print(out)
    out_fp = os.path.join('/storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/Depth-Anything/c3vd_data', dataset_name, 'poses_bounds.npy')
    np.save(out_fp, out)

def create_masks(dataset_name):
    # invert mask
    mask = cv2.imread(f'/storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/REIM-NeRF/data_preprocessed/{dataset_name}/rgb_mask.png')

    if not os.path.exists(os.path.join('/storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/Depth-Anything/c3vd_data', dataset_name, 'masks')):
        os.makedirs(os.path.join('/storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/Depth-Anything/c3vd_data', dataset_name, 'masks'))

    base_dir = '/storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/REIM-NeRF/data_preprocessed'
    img_dir = os.path.join(base_dir, dataset_name, 'images')
    num_imgs = len(os.listdir(img_dir))
    for img_name in os.listdir(img_dir):
        # frame_name, color, png = img_name.split('.')
        # mask_name = [frame_name, 'mask', png].join('.')
        
        dirname = os.path.join('/storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/Depth-Anything/c3vd_data', dataset_name, 'masks', img_name)
        cv2.imwrite(dirname, mask)

# depth = cv2.imread('ct1a/depth/0_color_depth.png')
# print(depth)

def write_all_npy():
    for dataset_name in os.listdir('/storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/REIM-NeRF/data_preprocessed'):
        print(dataset_name)
        write_poses_bounds(dataset_name)

def create_all_masks():
    for dataset_name in os.listdir('/storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/REIM-NeRF/data_preprocessed'):
        print(dataset_name)
        create_masks(dataset_name)

# write_all_npy()
create_all_masks()