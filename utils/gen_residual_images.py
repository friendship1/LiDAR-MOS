#!/usr/bin/env python3
# Developed by Xieyuanli Chen
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script generates residual images
import cv2
import os
import sys
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
from utils import load_poses, load_calib, load_files, load_vertex

#from utils import range_projection
try:
  from c_gen_virtual_scan import gen_virtual_scan as range_projection
except:
  print("Using clib by $export PYTHONPATH=$PYTHONPATH:<path-to-library>")
  print("Currently using python-lib to generate range images.")
  from utils import range_projection

#from utils import range_projection
if __name__ == '__main__':
  #print('a')
  dataset = {
            'train' :['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10','11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
            #'train':['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
            #'train':['21']
            #'train':['19']
            #'train':['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
            }
  # load config file

  for category, scene_list in dataset.items():
    for scene in scene_list:
      for num_last_n in range(1,11):
        print(f'Scene# last# {scene}:{num_last_n}')
        config_filename = 'config/data_preparing.yaml'
        #yaml_filename = 'data_preparing_' + scene + '.yaml'
        #config_filename = os.path.join('config', yaml_filename)
        #print(yaml_filename)
        if len(sys.argv) > 1:
          config_filename = sys.argv[1]
        
        if yaml.__version__ >= '5.1':
          config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
          print('ver > 5.1 : {}'.format(config_filename))
        else:
          config = yaml.load(open(config_filename))
          #print('ver < 5.1 : {}'.format(config_filename))
        

        # specify parameters
        num_frames = config['num_frames']
        debug = config['debug']
        normalize = config['normalize']
        #num_last_n = config['num_last_n']
        visualize = config['visualize']
        try:
          base_folder = config['base_folder']
        except:
          base_folder = 'data/sequences'
        #
        #visualization_folder = config['visualization_folder']
        #visualization_folder: 'data/sequences/08/visualization_1'
        #residual_image_folder: 'data/sequences/08/residual_images_1'
        #visualization_folder = visualization_folder + '_' + str(num_last_n)
        visualization_folder = os.path.join(base_folder, scene, 'visualization_'+ str(num_last_n))
        
        #print(visualization_folder)
        # specify the output folders
        #residual_image_folder = config['residual_image_folder']
        residual_image_folder = os.path.join(base_folder, scene, 'residual_images_'+ str(num_last_n))
        #residual_image_folder = residual_image_folder + '_' + str(num_last_n)
        if not os.path.exists(residual_image_folder):
          os.makedirs(residual_image_folder)
          
        if visualize:
          if not os.path.exists(visualization_folder):
            os.makedirs(visualization_folder)
        
        # scan_folder: 'data/sequences/08/velodyne'
        # # ground truth poses file
        # pose_file: 'data/sequences/08/poses.txt'
        # # calibration file
        # calib_file: 'data/sequences/08/calib.txt'

        # load poses
        #pose_file = config['pose_file']
        pose_file = os.path.join(base_folder, scene, 'poses.txt')
        poses = np.array(load_poses(pose_file))
        inv_frame0 = np.linalg.inv(poses[0])
        
        # load calibrations
        #calib_file = config['calib_file']
        calib_file = os.path.join(base_folder, scene, 'calib.txt')
        T_cam_velo = load_calib(calib_file)
        T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
        T_velo_cam = np.linalg.inv(T_cam_velo)
        
        # convert kitti poses from camera coord to LiDAR coord
        new_poses = []
        for pose in poses:
          new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
        poses = np.array(new_poses)
        
        # load LiDAR scans
        #scan_folder = config['scan_folder']
        scan_folder = os.path.join(base_folder, scene, 'velodyne')
        scan_paths = load_files(scan_folder)
        
        # test for the first N scans
        if num_frames >= len(poses) or num_frames <= 0:
          print('generate training data for all frames with number of: ', len(poses))
        else:
          poses = poses[:num_frames]
          scan_paths = scan_paths[:num_frames]
        
        range_image_params = config['range_image']
        
        # generate residual images for the whole sequence
        for frame_idx in tqdm(range(len(scan_paths))):
          #print(frame_idx)
          file_name = os.path.join(residual_image_folder, str(frame_idx).zfill(6))
          diff_image = np.full((range_image_params['height'], range_image_params['width']), 0,
                                  dtype=np.float32)  # [H,W] range (0 is no data)
          
          # for the first N frame we generate a dummy file
          if frame_idx < num_last_n:
            np.save(file_name, diff_image)
          
          else:
            # load current scan and generate current range image
            current_pose = poses[frame_idx]
            current_scan = load_vertex(scan_paths[frame_idx])
            current_range = range_projection(current_scan.astype(np.float32),
                                            range_image_params['height'], range_image_params['width'],
                                            range_image_params['fov_up'], range_image_params['fov_down'],
                                            range_image_params['max_range'], range_image_params['min_range'])[:, :, 3]
            
            # load last scan, transform into the current coord and generate a transformed last range image
            last_pose = poses[frame_idx - num_last_n]
            last_scan = load_vertex(scan_paths[frame_idx - num_last_n])
            last_scan_transformed = np.linalg.inv(current_pose).dot(last_pose).dot(last_scan.T).T
            last_range_transformed = range_projection(last_scan_transformed.astype(np.float32),
                                                      range_image_params['height'], range_image_params['width'],
                                                      range_image_params['fov_up'], range_image_params['fov_down'],
                                                      range_image_params['max_range'], range_image_params['min_range'])[:, :, 3]
            
            # generate residual image
            valid_mask = (current_range > range_image_params['min_range']) & \
                        (current_range < range_image_params['max_range']) & \
                        (last_range_transformed > range_image_params['min_range']) & \
                        (last_range_transformed < range_image_params['max_range'])
            difference = np.abs(current_range[valid_mask] - last_range_transformed[valid_mask])
            
            if normalize:
              difference = np.abs(current_range[valid_mask] - last_range_transformed[valid_mask]) / current_range[valid_mask]

            diff_image[valid_mask] = difference

            

            #diff_image = np.minimum(diff_image, 5)
            #print('diff : max {}, min {}'.format(np.max(diff_image), np.min(diff_image)))

            if debug:
              fig, axs = plt.subplots(3)
              axs[0].imshow(last_range_transformed)
              axs[1].imshow(current_range)
              axs[2].imshow(diff_image, vmin=0, vmax=10)
              plt.show()
              
            if visualize:
              # fig = plt.figure(frameon=False, figsize=(16, 10))
              # fig.set_size_inches(20.48, 0.64)
              # ax = plt.Axes(fig, [0., 0., 1., 1.])
              # ax.set_axis_off()
              # fig.add_axes(ax)
              # ax.imshow(diff_image, vmin=0, vmax=1)
              image_name = os.path.join(visualization_folder, str(frame_idx).zfill(6) + '.png')
              # plt.savefig(image_name)
              #print(image_name)
              diff_saveimg = diff_image*100
              diff_saveimg = np.minimum(diff_saveimg, 255)
              cv2.imwrite(image_name, diff_saveimg)
              #print(np.max(diff_saveimg))
              #diff_image = np.minimum(diff_image, 255)
              # plt.close()
              # fig.clf()

            # save residual image
            np.save(file_name, diff_image)
