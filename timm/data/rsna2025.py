import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm

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
        # if not os.path.exists('kaggle/working/preprogress/series'):
        #     os.makedirs('kaggle/working/preprogress/series')
        for uid in tqdm(self.selected_uids, desc='Loading DICOM series'):
            image = self._load_image(os.path.join(series_dir, f'{uid}.npy'))
            # image = self._load_image(os.path.join(series_dir, uid))
            # np.save(f'kaggle/working/preprogress/series/{uid}.npy', image)  
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
        # image = process_dicom_series_safe(image_path, self.target_shape)
        image = np.load(image_path)
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
