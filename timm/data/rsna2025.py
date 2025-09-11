import os
import numpy as np
import pydicom
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from scipy import ndimage
import warnings
import gc
import torch
from torch.utils.data import Dataset
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
warnings.filterwarnings('ignore')

class DICOMPreprocessorKaggle:
    """
    DICOM preprocessing system for Kaggle Code Competition
    Converts original DICOMPreprocessor logic to single series processing
    """
    
    def __init__(self, target_shape: Tuple[int, int, int] = (32, 384, 384)):
        self.target_depth, self.target_height, self.target_width = target_shape
        
    def load_dicom_series(self, series_path: str) -> Tuple[List[pydicom.Dataset], str]:
        """
        Load DICOM series
        """
        series_path = Path(series_path)
        series_name = series_path.name
        
        # Search for DICOM files
        dicom_files = []
        for root, _, files in os.walk(series_path):
            for file in files:
                if file.endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {series_path}")
        
        #print(f"Found {len(dicom_files)} DICOM files in series {series_name}")
        
        # Load DICOM datasets
        datasets = []
        for filepath in dicom_files:
            try:
                ds = pydicom.dcmread(filepath, force=True)
                datasets.append(ds)
            except Exception as e:
                #print(f"Failed to load {filepath}: {e}")
                continue
        
        if not datasets:
            raise ValueError(f"No valid DICOM files in {series_path}")
        
        return datasets, series_name
    
    def extract_slice_info(self, datasets: List[pydicom.Dataset]) -> List[Dict]:
        """
        Extract position information for each slice
        """
        slice_info = []
        
        for i, ds in enumerate(datasets):
            info = {
                'dataset': ds,
                'index': i,
                'instance_number': getattr(ds, 'InstanceNumber', i),
            }
            
            # Get z-coordinate from ImagePositionPatient
            try:
                position = getattr(ds, 'ImagePositionPatient', None)
                if position is not None and len(position) >= 3:
                    info['z_position'] = float(position[2])
                else:
                    # Fallback: use InstanceNumber
                    info['z_position'] = float(info['instance_number'])
                    #print("ImagePositionPatient not found, using InstanceNumber")
            except Exception as e:
                info['z_position'] = float(i)
                #print(f"Failed to extract position info: {e}")
            
            slice_info.append(info)
        
        return slice_info
    
    def sort_slices_by_position(self, slice_info: List[Dict]) -> List[Dict]:
        """
        Sort slices by z-coordinate
        """
        # Sort by z-coordinate
        sorted_slices = sorted(slice_info, key=lambda x: x['z_position'])
        
        #print(f"Sorted {len(sorted_slices)} slices by z-position")
        #print(f"Z-range: {sorted_slices[0]['z_position']:.2f} to {sorted_slices[-1]['z_position']:.2f}")
        
        return sorted_slices
    
    def get_windowing_params(self, ds: pydicom.Dataset, img: np.ndarray = None) -> Tuple[Optional[float], Optional[float]]:
        """
        Get windowing parameters based on modality
        """
        modality = getattr(ds, 'Modality', 'CT')
        
        if modality == 'CT':
            # For CT, apply CTA (angiography) settings
            center, width = (50, 350)
            #print(f"Using CTA windowing for CT: Center={center}, Width={width}")
            # return center, width
            return "CT", "CT"
            
        elif modality == 'MR':
            # For MR, skip windowing (statistical normalization only)
            #print("MR modality detected: skipping windowing, using statistical normalization")
            return None, None
            
        else:
            # Unexpected modality (safety measure)
            #print(f"Unexpected modality '{modality}', using CTA windowing")
            #return (50, 350)
            return None, None
    
    def apply_windowing_or_normalize(self, img: np.ndarray, center: Optional[float], width: Optional[float]) -> np.ndarray:
        """
        Apply windowing or statistical normalization
        """
        if center is not None and width is not None:
            # # Windowing processing (for CT/CTA)
            # img_min = center - width / 2
            # img_max = center + width / 2
            
            # windowed = np.clip(img, img_min, img_max)
            # windowed = (windowed - img_min) / (img_max - img_min + 1e-7)
            # result = (windowed * 255).astype(np.uint8)
            
            # #print(f"Applied windowing: [{img_min:.1f}, {img_max:.1f}] → [0, 255]")
            # return result
            
            # Statistical normalization (for CT as well)
            # Normalize using 1-99 percentiles
            p1, p99 = np.percentile(img, [1, 99])
            p1, p99 = 0, 500
            
            if p99 > p1:
                normalized = np.clip(img, p1, p99)
                normalized = (normalized - p1) / (p99 - p1)
                result = (normalized * 255).astype(np.uint8)
                
                #print(f"Applied statistical normalization: [{p1:.1f}, {p99:.1f}] → [0, 255]")
                return result
            else:
                # Fallback: min-max normalization
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    normalized = (img - img_min) / (img_max - img_min)
                    result = (normalized * 255).astype(np.uint8)
                    #print(f"Applied min-max normalization: [{img_min:.1f}, {img_max:.1f}] → [0, 255]")
                    return result
                else:
                    # If image has no variation
                    #print("Image has no variation, returning zeros")
                    return np.zeros_like(img, dtype=np.uint8)
        
        else:
            # Statistical normalization (for MR)
            # Normalize using 1-99 percentiles
            p1, p99 = np.percentile(img, [1, 99])
            
            if p99 > p1:
                normalized = np.clip(img, p1, p99)
                normalized = (normalized - p1) / (p99 - p1)
                result = (normalized * 255).astype(np.uint8)
                
                #print(f"Applied statistical normalization: [{p1:.1f}, {p99:.1f}] → [0, 255]")
                return result
            else:
                # Fallback: min-max normalization
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    normalized = (img - img_min) / (img_max - img_min)
                    result = (normalized * 255).astype(np.uint8)
                    #print(f"Applied min-max normalization: [{img_min:.1f}, {img_max:.1f}] → [0, 255]")
                    return result
                else:
                    # If image has no variation
                    #print("Image has no variation, returning zeros")
                    return np.zeros_like(img, dtype=np.uint8)
    
    def extract_pixel_array(self, ds: pydicom.Dataset) -> np.ndarray:
        """
        Extract 2D pixel array from DICOM and apply preprocessing (for 2D DICOM series)
        """
        # Get pixel data
        img = ds.pixel_array.astype(np.float32)
        
        # For 3D volume case (multiple frames) - select middle frame
        if img.ndim == 3:
            #print(f"3D DICOM in 2D processing - using middle frame from shape: {img.shape}")
            frame_idx = img.shape[0] // 2
            img = img[frame_idx]
            #print(f"Selected frame {frame_idx} from 3D DICOM")
        
        # Convert color image to grayscale
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            #print("Converted color image to grayscale")
        
        # Apply RescaleSlope and RescaleIntercept
        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        slope, intercept = 1, 0
        if slope != 1 or intercept != 0:
            img = img * float(slope) + float(intercept)
            #print(f"Applied rescaling: slope={slope}, intercept={intercept}")
        
        return img
    
    def resize_volume_3d(self, volume: np.ndarray) -> np.ndarray:
        """
        Resize 3D volume to target size
        """
        current_shape = volume.shape
        target_shape = (self.target_depth, self.target_height, self.target_width)
        
        if current_shape == target_shape:
            return volume
        
        #print(f"Resizing volume from {current_shape} to {target_shape}")
        
        # 3D resizing using scipy.ndimage
        zoom_factors = [
            target_shape[i] / current_shape[i] for i in range(3)
        ]
        
        # Resize with linear interpolation
        resized_volume = ndimage.zoom(volume, zoom_factors, order=1, mode='nearest')
        
        # Clip to exact size just in case
        resized_volume = resized_volume[:self.target_depth, :self.target_height, :self.target_width]
        
        # Padding if necessary
        pad_width = [
            (0, max(0, self.target_depth - resized_volume.shape[0])),
            (0, max(0, self.target_height - resized_volume.shape[1])),
            (0, max(0, self.target_width - resized_volume.shape[2]))
        ]
        
        if any(pw[1] > 0 for pw in pad_width):
            resized_volume = np.pad(resized_volume, pad_width, mode='edge')
        
        #print(f"Final volume shape: {resized_volume.shape}")
        return resized_volume.astype(np.uint8)
    
    def process_series(self, series_path: str) -> np.ndarray:
        """
        Process DICOM series and return as NumPy array (for Kaggle: no file saving)
        """
        try:
            # 1. Load DICOM files
            datasets, series_name = self.load_dicom_series(series_path)
            
            # Check first DICOM to determine 3D/2D
            first_ds = datasets[0]
            first_img = first_ds.pixel_array
            
            if len(datasets) == 1 and first_img.ndim == 3:
                # Case 1: Single 3D DICOM file
                #print(f"Processing single 3D DICOM with shape: {first_img.shape}")
                return self._process_single_3d_dicom(first_ds, series_name)
            else:
                # Case 2: Multiple 2D DICOM files
                #print(f"Processing {len(datasets)} 2D DICOM files")
                return self._process_multiple_2d_dicoms(datasets, series_name)
            
        except Exception as e:
            #print(f"Failed to process series {series_path}: {e}")
            raise
    
    def _process_single_3d_dicom(self, ds: pydicom.Dataset, series_name: str) -> np.ndarray:
        """
        Process single 3D DICOM file (for Kaggle: no file saving)
        """
        # Get pixel array
        volume = ds.pixel_array.astype(np.float32)
        
        # Apply RescaleSlope and RescaleIntercept
        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        slope, intercept = 1, 0
        if slope != 1 or intercept != 0:
            volume = volume * float(slope) + float(intercept)
            # #print(f"Applied rescaling: slope={slope}, intercept={intercept}")
        
        # Get windowing settings
        window_center, window_width = self.get_windowing_params(ds)
        
        # Apply windowing to each slice
        processed_slices = []
        for i in range(volume.shape[0]):
            slice_img = volume[i]
            processed_img = self.apply_windowing_or_normalize(slice_img, window_center, window_width)
            processed_slices.append(processed_img)
        
        volume = np.stack(processed_slices, axis=0)
        ##print(f"3D volume shape after windowing: {volume.shape}")
        
        # 3D resize
        final_volume = self.resize_volume_3d(volume)
        
        ##print(f"Successfully processed 3D DICOM series {series_name}")
        return final_volume
    
    def _process_multiple_2d_dicoms(self, datasets: List[pydicom.Dataset], series_name: str) -> np.ndarray:
        """
        Process multiple 2D DICOM files (for Kaggle: no file saving)
        """
        slice_info = self.extract_slice_info(datasets)
        sorted_slices = self.sort_slices_by_position(slice_info)
        first_img = self.extract_pixel_array(sorted_slices[0]['dataset'])
        window_center, window_width = self.get_windowing_params(sorted_slices[0]['dataset'], first_img)
        processed_slices = []
        
        for slice_data in sorted_slices:
            ds = slice_data['dataset']
            img = self.extract_pixel_array(ds)
            processed_img = self.apply_windowing_or_normalize(img, window_center, window_width)
            resized_img = cv2.resize(processed_img, (self.target_width, self.target_height))
            
            processed_slices.append(resized_img)

        volume = np.stack(processed_slices, axis=0)
        ##print(f"2D slices stacked to volume shape: {volume.shape}")
        final_volume = self.resize_volume_3d(volume)
        
        ##print(f"Successfully processed 2D DICOM series {series_name}")
        return final_volume

def process_dicom_series_kaggle(series_path: str, target_shape: Tuple[int, int, int] = (32, 384, 384)) -> np.ndarray:
    """
    DICOM processing function for Kaggle inference (single series)
    
    Args:
        series_path: Path to DICOM series
        target_shape: Target volume size (depth, height, width)
    
    Returns:
        np.ndarray: Processed volume
    """
    preprocessor = DICOMPreprocessorKaggle(target_shape=target_shape)
    return preprocessor.process_series(series_path)

# Safe processing function with memory cleanup
def process_dicom_series_safe(series_path: str, target_shape: Tuple[int, int, int] = (32, 384, 384)) -> np.ndarray:
    """
    Safe DICOM processing with memory cleanup
    
    Args:
        series_path: Path to DICOM series
        target_shape: Target volume size (depth, height, width)
    
    Returns:
        np.ndarray: Processed volume
    """
    try:
        volume = process_dicom_series_kaggle(series_path, target_shape)
        return volume
    finally:
        # Memory cleanup
        gc.collect()

# 你可以在这个地方定义 CFG 和尺寸等全局配置
class CFG:
    size = 384  # 假设目标尺寸为 256，按照需要修改

def get_inference_transform():
    """获取推理时的预处理变换"""
    return A.Compose([
        A.Resize(CFG.size, CFG.size),  # 调整图像大小
        A.Normalize(),  # 归一化
        ToTensorV2(),  # 转为 PyTorch Tensor
    ])

class RSNADataset(Dataset):
    def __init__(self, train_csv, series_dir, num_limit=-1, use_cache=False, target_shape=(32, 384, 384), transform=None):
        # 读取 CSV 文件
        self.train_df = pd.read_csv(train_csv)
        
        # 如果样本数大于 num_limit，应用限制
        if len(self.train_df) > num_limit and num_limit != -1:
            self.sample_df = self.train_df.head(num_limit)
        else:
            self.sample_df = self.train_df
        
        # 从 DataFrame 中选择 SeriesInstanceUID
        self.selected_uids = self.sample_df['SeriesInstanceUID'].tolist()
        
        # 定义标签列
        self.LABEL_COLS = [
            'Left Infraclinoid Internal Carotid Artery',
            'Right Infraclinoid Internal Carotid Artery',
            'Left Supraclinoid Internal Carotid Artery',
            'Right Supraclinoid Internal Carotid Artery',
            'Left Middle Cerebral Artery',
            'Right Middle Cerebral Artery',
            'Anterior Communicating Artery',
            'Left Anterior Cerebral Artery',
            'Right Anterior Cerebral Artery',
            'Left Posterior Communicating Artery',
            'Right Posterior Communicating Artery',
            'Basilar Tip',
            'Other Posterior Circulation',
            'Aneurysm Present',
        ]
        
        # 初始化一些变量
        self.series_paths = []  # 存储图像序列的路径
        self.labels = []        # 存储标签
        self.use_cache = use_cache
        self.target_shape = target_shape
        
        # 默认使用的变换
        self.transform = transform if transform else get_inference_transform()
        
        # 加载图像和标签
        for uid in tqdm(self.selected_uids, desc='Loading DICOM series'):
            image = self._load_image(os.path.join(series_dir, uid))  
            labels = self.sample_df[self.sample_df['SeriesInstanceUID'] == uid][self.LABEL_COLS].values
            
            # 如果使用缓存，则将图像和标签存储到内存
            if self.use_cache:
                self.series_paths.append(image)
                self.labels.append(labels)
            else:
                # 如果不使用缓存，仅存储路径
                self.series_paths.append(os.path.join(series_dir, uid))
                self.labels.append(labels)
        
    def _load_image(self, image_path):
        """加载图像的辅助函数。"""
        image = process_dicom_series_safe(image_path, self.target_shape)
        return image
    
    def __len__(self):
        """返回数据集中的样本数量。"""
        return len(self.series_paths)
    
    def __getitem__(self, idx):
        """根据索引返回一个样本。"""
        series_path = self.series_paths[idx]
        labels = self.labels[idx]
        
        if self.use_cache:
            # 如果使用缓存，直接从内存中加载图像
            image = self.series_paths[idx]
        else:
            # 如果不使用缓存，加载图像
            image = self._load_image(series_path)
        
        image = image.transpose(1, 2, 0)  # 转置以适应 PyTorch 的通道顺序 (C, H, W)
        # 如果有 transform，应用变换
        if self.transform:
            volume = self.transform(image=image)["image"]
        
        # 将标签转换为 tensor
        labels_tensor = torch.tensor(labels, dtype=torch.float32).squeeze(0)

        return volume, labels_tensor
