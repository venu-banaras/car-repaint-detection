import os
import json
import torch
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2
from sklearn.cluster import KMeans

import torchvision.models.detection as tmd
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from src import config



# def generate_random_colors(num_classes):
#     """Generate random colors for each class."""
#     return {i: [random.randint(0, 255) for _ in range(3)] for i in range(1, num_classes + 1)}
def load_model(checkpoint_path, num_classes, device):
    model = tmd.maskrcnn_resnet50_fpn(pretrained=False, trainable_backbone_layers=3)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = tmd.faster_rcnn.FastRCNNPredictor(in_features, int(num_classes))

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = tmd.mask_rcnn.MaskRCNNPredictor(
                                                                        in_features_mask,
                                                                        hidden_layer,
                                                                        int(num_classes)
                                                                    )    
    
    sizes = ((4,),(8,),(16,),(32,),(64,),(128,),(256,),)
    aspect_ratio = tuple([(0.25,0.5,1.0,1.25,1.5) for _ in range(5)])    
    anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratio)
    model.rpn.anchor_generator = anchor_generator
    model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    return model

def infer_image(model, image_path, device, threshold):
    model.eval()
    with torch.no_grad():
        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image).unsqueeze(0).to(device)
        outputs = model(image_tensor)[0]
        masks, boxes, scores, labels = (
            outputs["masks"], outputs["boxes"], outputs["scores"], outputs["labels"]
        )
        keep = scores > threshold
        return {
            "masks": masks[keep].cpu(),
            "boxes": boxes[keep].cpu(),
            "scores": scores[keep].cpu(),
            "labels": labels[keep].cpu(),
        }


def draw_predictions(image, predictions, class_colors):
    draw = ImageDraw.Draw(image, "RGBA")
    for mask, box, score, label in zip(
        predictions["masks"], predictions["boxes"], predictions["scores"], predictions["labels"]
    ):
        label = int(label)
        color = tuple(class_colors[label] + [128])  

        binary_mask = mask.squeeze(0) > 0.5
        binary_mask_np = binary_mask.cpu().numpy()  
        mask_array = np.array(binary_mask_np, dtype=np.uint8) * 255

        overlay = Image.fromarray(mask_array, mode="L").resize(image.size, Image.NEAREST)
        mask_overlay = Image.new("RGBA", image.size, color)
        image.paste(mask_overlay, (0, 0), overlay)

        x_min, y_min, x_max, y_max = box
        draw.rectangle([x_min, y_min, x_max, y_max], outline=tuple(class_colors[label]), width=3)

        label_text = f"{config.class_names.get(label, 'unknown')} {score:.2f}"
        draw.text((x_min, y_min), label_text, fill=(255, 255, 255, 255))

    return image

def dominant(image):
    '''Function to find Dominant colours by grouping into 4 clusters'''
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    pixels = image_rgb.reshape(-1, 3)   
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    dominant_colors = np.round(dominant_colors).astype(int)
    pixel_counts = np.array([np.sum(labels == i) for i in range(4)])
    total_pixels = pixels.shape[0]
    percentages = (pixel_counts / total_pixels) * 100
    return percentages

def make_mask(b_mask,image,xmin, xmax, ymin, ymax):
    '''Function to make masks using the binary mask, the crop and its bounding box coordinates'''

    cropped_image = image.crop((xmin, ymin, xmax, ymax))

    binary_mask_cropped = b_mask[ymin:ymax, xmin:xmax]

    mask_array = np.zeros((ymax - ymin, xmax - xmin), dtype=np.uint8)
    mask_array[binary_mask_cropped] = 255 

    mask_image = Image.fromarray(mask_array)

    cropped_image = cropped_image.convert("RGBA")
    mask_image = mask_image.convert("L") 

    
    cropped_image.putalpha(mask_image)
    return cropped_image

def extract_exact_shades_from_dominant_cluster(image, num_clusters=4):
 
    import collections
    
    
    image_np = np.array(image)
    
    original_rgb = image_np.copy()
    
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    pixels_hsv = image_hsv.reshape(-1, 3)
    
    height, width = image_np.shape[:2]
    total_pixels = height * width
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(pixels_hsv)
    
    labels = kmeans.labels_
    pixel_counts = np.array([np.sum(labels == i) for i in range(num_clusters)])
    percentages = (pixel_counts / total_pixels) * 100
    dominant_cluster_index = np.argmax(percentages)
    dominant_cluster_percentage = percentages[dominant_cluster_index]
    
    dominant_mask = (labels == dominant_cluster_index)
    
    y_coords = np.floor_divide(np.where(dominant_mask)[0], width).astype(int)
    x_coords = np.mod(np.where(dominant_mask)[0], width).astype(int)
    
    rgb_values = []
    for y, x in zip(y_coords, x_coords):
        rgb = tuple(original_rgb[y, x])
        rgb_values.append(rgb)
    
    rgb_counter = collections.Counter(rgb_values)
    
    unique_rgb_shades = dict(rgb_counter)
    
    sorted_shades = []
    for rgb, count in rgb_counter.most_common():
        shade_percentage = (count / pixel_counts[dominant_cluster_index]) * 100
        sorted_shades.append((rgb, count, shade_percentage))

    dominant_viz = visualize_dominant_cluster(image_np, labels, dominant_cluster_index, output_path="output/dominant_cluster.png")
    visualizations, composite = visualize_all_clusters(image_np, labels, num_clusters=4, output_folder="output/")
    
    return dominant_cluster_index, dominant_cluster_percentage, unique_rgb_shades, sorted_shades

def evaluate_model(
    model, image_folder, output_json_path, output_image_folder, CROPS_FOLDER, device, threshold=0.3, num_classes=95):
    class_colors = [config.class_color[i] for i in range(1, num_classes)]

    metallic_parts=[
    "right_front_door",
    "right_rear_door",
    "left_front_door",
    "left_rear_door",
    "right_qtr_panel",
    "left_qtr_panel",
    "tail_gate",                    #22 parts
    "right_pillar_a",
    "right_pillar_b",
    "right_pillar_c",
    "right_pillar_d",
    "left_pillar_a",
    "left_pillar_b",
    "left_pillar_c",
    "left_pillar_d",
    "right_fender_extender",
    "left_fender_extender",
    "right_qtr_extender",
    "left_qtr_extender",
    "right_fender",
    "left_fender",
    "hood",
]

    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": config.class_names.get(i, f"class_{i}")} for i in range(1, num_classes)],
    }
    annotation_id = 1

    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(CROPS_FOLDER, exist_ok=True)


    for i, image_name in tqdm(enumerate(os.listdir(image_folder))):
        image_path = os.path.join(image_folder, image_name)
        image_id = i
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error reading image {image_path}", e)
            continue

        coco_output["images"].append({"id": image_id, "file_name": image_name, "width": width, "height": height})

        prediction = infer_image(model, image_path, device, threshold)


        for mask, bbox, score, label in zip(
            prediction["masks"], prediction["boxes"], prediction["scores"], prediction["labels"]
        ):
            binary_mask = mask.squeeze(0) > 0.5
            area = binary_mask.sum().item()

            x_min, y_min, x_max, y_max = bbox

            coco_output["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "category_name": config.class_names.get(int(label), f"class_{int(label)}"),
                    "bbox": [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)],
                    "score": float(score),
                    "area": float(area),
                }
            )
            annotation_id += 1

            part = config.class_names.get(int(label), f"class_{int(label)}")
            
            if part in metallic_parts:
                x_min, y_min, x_max, y_max = map(int, bbox)  # Converting tensors to integers

                with Image.open(image_path) as img:

                    cropped_img=make_mask(binary_mask,img,x_min,x_max,y_min,y_max)

                    print(f"for {part}/n")

                    dominant_idx, dominant_pct, unique_shades, sorted_shades = extract_exact_shades_from_dominant_cluster(cropped_img)
                    print(f"Dominant cluster: {dominant_idx}, covering {dominant_pct:.2f}% of the image")
                    print(f"Found {len(unique_shades)} unique RGB shades in the dominant cluster")
                    print("\nTop 10 most common shades:")
                    for i, (rgb, count, percentage) in enumerate(sorted_shades[:10]):
                         print(f"Shade #{i+1}: RGB{rgb}, Count: {count} pixels ({percentage:.2f}% of dominant cluster)")
                    
                    export_shades_to_csv(sorted_shades, 'car_part_shades.csv')
                    cropped_img.save(os.path.join(CROPS_FOLDER, f"{image_name}_{annotation_id}.png"))
                    # percentages = dominant(cropped_img)

                    # if any(percentages > 60):
                    #     print(f'{part} is repainted')
                    


        with Image.open(image_path).convert("RGB") as img:
            visualized_image = draw_predictions(img, prediction, class_colors)
            visualized_image.save(os.path.join(output_image_folder, image_name))

    with open(output_json_path, "w") as f:
        json.dump(coco_output, f, indent=4)



def export_shades_to_csv(sorted_shades, filename='shade_analysis.csv'):
    import csv
    std=0
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Shade_Number', 'R', 'G', 'B', 'Pixel_Count', 'Percentage', 'Manhattan_Distance','Weighted_Squared_Distance'])
        
        for i, (rgba, count, percentage) in enumerate(sorted_shades):
            r = int(rgba[0])
            g = int(rgba[1])
            b = int(rgba[2])
            manhattan_Distance = abs(255 - r) + abs(255 - g) + abs(255 - b)
            Weighted_Squared_Distance = (manhattan_Distance**2) * (percentage/100)
            std=std+Weighted_Squared_Distance
            csv_writer.writerow([i+1, r, g, b, count, f"{percentage:.2f}", manhattan_Distance, Weighted_Squared_Distance])
    std=np.sqrt(std)
    print(f"Deviation: {std}")
    print(f"CSV file created successfully: {filename}")
    return filename

def visualize_dominant_cluster(image, labels, dominant_cluster_index, output_path="dominant_cluster.png"):
    
    visualization = np.zeros_like(image)
    
    height, width = image.shape[:2]
    flat_image = image.reshape(-1, image.shape[2])
    
    for i in range(len(labels)):
        if labels[i] == dominant_cluster_index:
            y = i // width
            x = i % width
            if y < height and x < width: 
                visualization[y, x] = image[y, x]
    
    # Save the visualization
    if output_path:
        if isinstance(visualization, np.ndarray):
            Image.fromarray(visualization).save(output_path)
        print(f"Saved dominant cluster visualization to {output_path}")
    
    return visualization

def visualize_all_clusters(image, labels, num_clusters=4, output_folder="./"):
    """
    Visualize each cluster separately and create a composite image showing all clusters
    
    Parameters:
    image - original image (numpy array in RGB format)
    labels - cluster labels for each pixel
    num_clusters - number of clusters
    output_folder - folder to save output images
    
    Returns:
    visualizations - list of cluster visualizations
    composite - composite visualization with all clusters
    """
    
    height, width = image.shape[:2]
    visualizations = []
    
    composite = np.zeros_like(image)
    
    cluster_colors = [
        [255, 0, 0],      # Red for cluster 0
        [0, 255, 0],      # Green for cluster 1
        [0, 0, 255],      # Blue for cluster 2
        [255, 255, 0],    # Yellow for cluster 3
        [255, 0, 255],    # Magenta for cluster 4
        [0, 255, 255],    # Cyan for cluster 5
        [255, 128, 0],    # Orange for cluster 6
        [128, 0, 255]     # Purple for cluster 7
    ]
    
    while len(cluster_colors) < num_clusters:
        cluster_colors.append([np.random.randint(0, 256) for _ in range(3)])
    
    for cluster_idx in range(num_clusters):
        cluster_viz = np.zeros_like(image)
        
        mask = np.zeros((height, width), dtype=bool)
        
        for i in range(len(labels)):
            if labels[i] == cluster_idx:
                y = i // width
                x = i % width
                if y < height and x < width:  
                    
                    cluster_viz[y, x] = image[y, x]
                    

                    composite[y, x] = np.append(cluster_colors[cluster_idx], 255)
                    
                    mask[y, x] = True
        
        if np.any(mask):
            kernel = np.ones((3, 3), np.uint8)
            dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
            border_mask = dilated_mask.astype(bool) & ~mask
            
            for y in range(height):
                for x in range(width):
                    if border_mask[y, x]:
                        composite[y, x] = [255, 255, 255,255]  # White border
        
        visualizations.append(cluster_viz)
        
        if output_folder:
            output_path = os.path.join(output_folder, f"cluster_{cluster_idx}.png")
            Image.fromarray(cluster_viz).save(output_path)
            print(f"Saved cluster {cluster_idx} visualization to {output_path}")
    
    if output_folder:
        composite_path = os.path.join(output_folder, "all_clusters_composite.png")
        Image.fromarray(composite).save(composite_path)
        print(f"Saved composite visualization to {composite_path}")
    
    return visualizations, composite


if __name__ == "__main__":
    IMAGE_FOLDER = "test_imgs"
    OUTPUT_JSON_PATH = "output_predictions.json"
    OUTPUT_IMAGE_FOLDER = "visualized_images"
    crops_folder="repainted_crops"
    MODEL_PATH = "car_part_v0.pth"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    THRESHOLD = 0.5
    NUM_CLASSES = 81

    model = load_model(MODEL_PATH, NUM_CLASSES, DEVICE)

    evaluate_model(model, IMAGE_FOLDER, OUTPUT_JSON_PATH, OUTPUT_IMAGE_FOLDER, crops_folder, DEVICE, THRESHOLD, NUM_CLASSES)
