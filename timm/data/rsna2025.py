import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold

class CFG:
    size = 384  # 假设目标尺寸为 256，按照需要修改

def get_inference_transform():
    """获取推理时的预处理变换"""
    return A.Compose([
        A.Resize(CFG.size, CFG.size),  # 调整图像大小
        A.Normalize(),  # 归一化
        ToTensorV2(),  # 转为 PyTorch Tensor
    ])

def get_train_transform():
    """获取训练时的预处理变换"""
    return A.Compose([
        A.Resize(CFG.size, CFG.size),  # 调整图像大小
        A.Blur(blur_limit=7, p=0.5),  # 随机模糊
        A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),  # 随机光学畸变
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),  # 随机平移、缩放和旋转
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5),  # 随机丢失区域
        A.Normalize(),  # 归一化
        ToTensorV2(),  # 转为 PyTorch Tensor
    ])

class RSNADataset(Dataset):
    def __init__(self, train_csv, series_dir, labels=None, split='validation', use_3d = False, fold=0, num_limit=-1, use_cache=False, target_shape=(32, 384, 384), transform=None):
        # 读取 CSV 文件
        self.train_df = pd.read_csv(train_csv)
        self.use_3d = use_3d
        
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
        if labels is not None:
            self.LABEL_COLS = [self.LABEL_COLS[i] for i in labels]
        # 初始化一些变量
        self.series_paths = []  # 存储图像序列的路径
        self.labels = []        # 存储标签
        self.use_cache = use_cache
        self.target_shape = target_shape
        
        # 默认使用的变换
        if transform:
            self.transform = transform
        else:
            if split == 'train' or split == 'all':
                self.transform = get_train_transform()
            else:
                self.transform = get_inference_transform()
        
        # 5-fold交叉验证
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        self.fold_splits = list(kf.split(self.selected_uids))
        
        # 获取当前 fold 的训练集和测试集索引
        train_idx, test_idx = self.fold_splits[fold]
        
        # 根据参数选择训练集或测试集
        if split == 'train':
            selected_idx = train_idx
        elif split == 'validation':
            selected_idx = test_idx
        elif split == 'all':
            selected_idx = np.arange(len(self.selected_uids))
        else:
            raise ValueError(f"train_or_test must be 'train' or 'validation', but got {split}")

        # 获取当前fold的UIDs
        self.selected_uids = [self.selected_uids[i] for i in selected_idx]
        
        # 加载图像和标签
        for uid in tqdm(self.selected_uids, desc='Loading DICOM series'):
            image = self._load_image(os.path.join(series_dir, f'{uid}.npy'))
            labels = self.sample_df[self.sample_df['SeriesInstanceUID'] == uid][self.LABEL_COLS].values

            # 如果使用缓存，则将图像和标签存储到内存
            if self.use_cache:
                self.series_paths.append(image)
                self.labels.append(labels)
            else:
                # 如果不使用缓存，仅存储路径
                self.series_paths.append(os.path.join(series_dir, f'{uid}.npy'))
                self.labels.append(labels)
        
    def _load_image(self, image_path):
        """加载图像的辅助函数。"""
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
        
        # Albumentations 期望输入为 (H, W, C)
        image = image.transpose(1, 2, 0)
        # 如果有 transform，应用变换
        if self.transform:
            volume = self.transform(image=image)["image"]
        if self.use_3d:
            volume = volume.unsqueeze(0)  # 添加一个额外的维度以适应 3D CNN
        
        # 将标签转换为 tensor
        labels_tensor = torch.tensor(labels, dtype=torch.float32).squeeze(0)
        
        # 返回仅 (volume, labels)，不再包含 age
        return volume, labels_tensor