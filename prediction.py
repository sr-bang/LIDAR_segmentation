from torch.utils.data import dataset
from tqdm import tqdm
import sys
import utils
import os
import random
import argparse
import numpy as np
from torch.utils import data
import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

sys.path.insert(0, '/home/shreya/Projects/lidar_segmentation/DeepLabV3Plus-Pytorch')

from datasets.cityscapes import Cityscapes
from metrics.stream_metrics import StreamSegMetrics
import network.modeling as modeling

def predict_from_checkpoint(checkpoint_path, rgb_data_path):
    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = modeling.__dict__['deeplabv3plus_resnet101'](num_classes=19, output_stride=16)  # Example model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model).to(device)
    model.eval()

    # Prepare the transformation
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    predictions = []
    img = Image.open(rgb_data_path).convert('RGB')
    img_transformed = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(img_transformed).max(1)[1].cpu().numpy()[0]  # HW
        predictions.append(pred)
        
        # Optional: Convert predictions to RGB (if decode_fn is available)
        colorized_pred = Cityscapes.decode_target(pred).astype('uint8')  # Example decode function
    return predictions, colorized_pred