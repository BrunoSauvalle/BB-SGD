
import os
import torch
import torch.utils.data as data
import natsort
import cv2
import numpy as np
from tqdm import tqdm

class Image_dataset_with_optical_flow(data.Dataset):

    def load_image(self,idx):
        image_path = os.path.join(self.dir, self.sorted_image_names[idx])
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # range 0-255 shape HWC BGR
        return image

    def __init__(self, args, device):

        self.path = args.input_path
        if os.path.exists(os.path.join(args.input_path, 'input')):
            self.dir = os.path.join(args.input_path, 'input')
        else:
            self.dir = args.input_path
        print(f'creating image dataset from directory {self.dir}')

        image_names = os.listdir(self.dir)
        self.sorted_image_names = natsort.natsorted(image_names)
        n_images = len(self.sorted_image_names) # number of images before motionless images suppression
        first_image = self.load_image(0)
        self.image_height, self.image_width, self.nc_input = first_image.shape
        assert self.nc_input ==3 # only color images are supported
        self.nc =3

        # first creates numpy arrays on CPU on computes optical flows
        self.images = np.zeros((n_images,3, self.image_height, self.image_width),
                                  dtype=np.uint8 )
        self.optical_flows = np.zeros((n_images, self.image_height, self.image_width),
                               dtype=np.uint8)

        print(f'loading  new dataset with {n_images} frames and w = {self.image_width},h = {self.image_height}')
        print(f' computing optical flow using algorithm {args.OF_algorithm}...')
        if args.OF_algorithm == 'DIS_ULTRAFAST':
            optical_flow_method = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        elif args.OF_algorithm == 'DIS_FAST':
            optical_flow_method = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        else:
            print(f'optical_flow_algo {args.OF_algorithm} not implemented, available implementations are DIS_FAST and DIS_ULTRAFAST')
            exit(0)

        opencv_next_image = self.load_image(0)
        next_image_GRAY = cv2.cvtColor(opencv_next_image, cv2.COLOR_BGR2GRAY)
        next_image_RGB = np.asarray(cv2.cvtColor(opencv_next_image, cv2.COLOR_BGR2RGB))

        motionless_image_suppression_mask = np.ones(n_images,dtype =bool) # mask will be set to False for motionless frames

        for i in tqdm(range(n_images)): # sequential loading for optical flow computation

            # uses image loaded in last iteration as current image and loads next image
            current_image_RGB = next_image_RGB
            current_image_GRAY = next_image_GRAY

            if not i == n_images-1:
                opencv_next_image = self.load_image(i + 1)
            else:
                opencv_next_image = self.load_image(i - 1) # to compute the optical flow of the last image, we use the previous image

            next_image_RGB = np.asarray(cv2.cvtColor(opencv_next_image, cv2.COLOR_BGR2RGB))

            # calls optical flow method, which requires grey images
            next_image_GRAY = cv2.cvtColor(opencv_next_image, cv2.COLOR_BGR2GRAY)
            flow = optical_flow_method.calc(current_image_GRAY, next_image_GRAY, None)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # computes optical flow mask and check if the image is motionless
            optical_flow_mask = np.minimum(np.array((255*mag)/ (args.tau_2*self.image_width)), [255])  # shape HxW range 0-255
            if np.max(optical_flow_mask) < 255*args.tau_3 and n_images >10 and args.motionless_frames_suppression:
                motionless_image_suppression_mask[i] = False  # indicates this motionless image should be removed

            self.images[i] = np.transpose(current_image_RGB, (2, 0, 1))  # shape CHW 0-255 RGB
            self.optical_flows[i] = optical_flow_mask

        self.dataset_length = self.images.shape[0]

        # removes motionless frames if required and number of images is at least 10
        if args.motionless_frames_suppression and n_images >10:
            self.images = self.images[motionless_image_suppression_mask, ...]
            self.optical_flows = self.optical_flows[motionless_image_suppression_mask, ...]
            self.dataset_length = self.images.shape[0]
            print(f'dataset remaining length after motionless image suppression is {self.dataset_length}')

        # moves dataset and optical flows to GPU if available
        self.images = torch.from_numpy(self.images).to(device)
        self.optical_flows = torch.from_numpy(self.optical_flows).to(device)


    def __getitem__(self, idx):
        return self.images[idx].detach(), self.optical_flows[idx].detach()

    def __len__(self):
        return self.dataset_length

