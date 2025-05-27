import sys
import numpy as np
import pickle
from numpy.linalg import norm
# import open3d as o3d
import time

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt


from PIL import Image
import torch
import torchvision.transforms as transforms
import cv2
import matplotlib.patches as patches


class Dinov2Matcher:

    def __init__(self, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14", smaller_edge_size=448,
                 patch_size=14, device="cuda", ref_img_name='water_bottle.jpeg', ref_patch=(10, 16)):
        self.repo_name = repo_name
        self.model_name = model_name
        self.smaller_edge_size = smaller_edge_size
        self.patch_size = patch_size
        self.device = device    

        self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(size=smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC,
                              antialias=True),
            transforms.ToTensor(),
        ])

        self.ref_img_name = ref_img_name
        self.ref_patch = ref_patch

        # Prepare reference image
        self.ref_img = cv2.cvtColor(cv2.imread(self.ref_img_name, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # Extract features for the reference image
        self.ref_img_tensor, self.ref_grid, self.ref_scale = self.prepare_image(self.ref_img)
        self.ref_features = self.extract_features(self.ref_img_tensor).reshape(*self.ref_grid, -1)
        self.ref_norm = norm(self.ref_features[self.ref_patch])

        # Show the reference image and reference patch
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(self.ref_img_tensor.squeeze().permute(1, 2, 0))
        rect = patches.Rectangle(
            (ref_patch[1] * self.patch_size, ref_patch[0] * self.patch_size), self.patch_size, self.patch_size,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        plt.title('Reference Image with Reference Patch')
        plt.savefig('ref_patch_visualization.png')

    def prepare_image(self, rgb_image_numpy):
        image = Image.fromarray(rgb_image_numpy)
        image_tensor = self.transform(image)
        resize_scale = image.width / image_tensor.shape[2]

        height, width = image_tensor.shape[1:]  # C x H x W
        cropped_width, cropped_height = width - width % self.patch_size, height - height % self.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        grid_size = (cropped_height // self.patch_size, cropped_width // self.patch_size)
        return image_tensor, grid_size, resize_scale

    def extract_features(self, image_tensor):
        with torch.inference_mode():
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
        if self.model_name == "dino_vitb8":
            return tokens.cpu().numpy()[1:]
        return tokens.cpu().numpy()

    def calculate_heatmap(self, target_img):
        # Extract features for the target image
        target_img_tensor, target_grid, target_scale = self.prepare_image(target_img)
        target_features = self.extract_features(target_img_tensor).reshape(*target_grid, -1)

        # Calculate Heatmap Scores
        target_norms = norm(target_features, axis=2)
        heatmap_scores = np.tensordot(self.ref_features[self.ref_patch], target_features, axes=([0], [2])) / (self.ref_norm * target_norms)
        heatmap_scores = np.nan_to_num(heatmap_scores)
        

        # Apply threshold to filter out low similarity scores
        thresholded_scores = heatmap_scores.copy()
        thresholded_scores[thresholded_scores < 0.6] = 0.0

        # Find clusters and their centers
        image_corners = []
        non_zero_coords = np.nonzero(thresholded_scores)
        x = (non_zero_coords[1] / 42.0 * target_img.shape[1]) + self.patch_size/2
        y = (non_zero_coords[0] / 32.0 * target_img.shape[0]) + self.patch_size/2
        x = x.astype(int)
        y = y.astype(int)
        points = [(y[i], x[i]) for i in range(len(non_zero_coords[0]))]

        if len(non_zero_coords[0]) > 0:  # Check if there are any non-zero values
            # Create array of coordinates
            # points = np.column_stack((non_zero_coords[0], non_zero_coords[1]))
            
            # Use K-means clustering to find 4 clusters
            from sklearn.cluster import KMeans
            if len(points) >= 4:  # Make sure we have enough points for 4 clusters
                kmeans = KMeans(n_clusters=4, random_state=0).fit(points)
                centers = kmeans.cluster_centers_

                # Convert cluster centers to integer coordinates
                centers = centers.astype(int)
                
                # Convert to image coordinates
                image_corners = [(y, x ) for y, x in centers]
        
        # Normalize the heatmap scores
        if thresholded_scores.max() > thresholded_scores.min():
            thresholded_scores = (thresholded_scores - thresholded_scores.min()) / (thresholded_scores.max() - thresholded_scores.min())
        else:
            # Avoid division by zero if all values are below threshold
            thresholded_scores[:] = 0

        # Convert heatmap to image using OpenCV
        heatmap_img = cv2.applyColorMap((thresholded_scores * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        heatmap_img = cv2.resize(heatmap_img, (target_img.shape[1], target_img.shape[0]))
        
        target_heatmap_img = cv2.addWeighted(cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR), 0.4, heatmap_img, 0.6, 0)
        target_heatmap_img_test = target_heatmap_img.copy()

        # Draw cluster centers on the image
        if len(image_corners) > 0:
            # Draw points for each corner/cluster center
            for i, corner in enumerate(image_corners):
                cv2.circle(target_heatmap_img, (corner[1], corner[0]), 5, (255, 0, 0), -1)  # Green dots
                cv2.putText(target_heatmap_img, str(i), (corner[1]+10, corner[0]+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        x = (non_zero_coords[1] / 42.0 * target_img.shape[1]) + self.patch_size/2
        y = (non_zero_coords[0] / 32.0 * target_img.shape[0]) + self.patch_size/2
        x = x.astype(int)
        y = y.astype(int)
        coords_list = [(y[i], x[i]) for i in range(len(non_zero_coords[0]))]

        if len(non_zero_coords[0]) > 0:
            for i in range(len(non_zero_coords[0])):
                cv2.circle(target_heatmap_img_test, (coords_list[i][1], coords_list[i][0]), 2, (255, 0, 0), -1)  # Red dots
                cv2.putText(target_heatmap_img_test, str(i), (coords_list[i][1]+10, coords_list[i][0]+10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        plt.imsave('target_heatmap_img_test.png', target_heatmap_img_test)

        return thresholded_scores, target_heatmap_img, heatmap_img, image_corners


def find_highest_heatmap_point(heatmap_img):
    # Convert heatmap image to grayscale in order to find the whitest point
    gray_heatmap = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2GRAY)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray_heatmap)
    max_loc_flatten = heatmap_img.shape[0] * max_loc[0] + max_loc[1]
    return max_loc, max_val, max_loc_flatten


def main(target_img, source_img_path, patch_size, ref_patch):
    # Init Dinov2Matcher
    dm = Dinov2Matcher(repo_name='facebookresearch/dinov2', model_name='dinov2_vitb14', patch_size=patch_size, ref_img_name=source_img_path, ref_patch=ref_patch)


    # Load target image
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    heatmap_scores, target_heatmap_img, heatmap_img, image_corners = dm.calculate_heatmap(target_img)
    print(find_highest_heatmap_point(heatmap_img))
    plt.imsave('heatmap_img.png', heatmap_img)
    plt.imsave('target_heatmap_img.png', target_heatmap_img)
    print(f"image_corners: {image_corners}")
    print(f"Image saved to heatmap_img.png and target_heatmap_img.png")
    # plt.show()
    

if __name__ == "__main__":

    source_img_path = 'captured_img/og_frame_700.jpg'
    target_img = cv2.imread('new_pose_gt/og_frame_0.jpg')
    patch_size = 14
    ref_patch = (24, 19)
    
    main(target_img, source_img_path, patch_size, ref_patch)
