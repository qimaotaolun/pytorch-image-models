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
    """获取训练时的预处理变换（参考 DINOv3 的增强思想，结合医学影像适配）
    
    对齐思路：
    - 使用 RandomResizedCrop 模拟 DINO 的多尺度随机裁剪（全局裁剪比例约 (0.25, 1.0)）
    - 去除大角度旋转与强几何畸变，保持与 DINO 一致的几何增广（裁剪 + 可选翻转）
    - 使用两条强度增广路径的 OneOf，分别模拟 DINO 的 global_transfo1 / global_transfo2
      • 路径A：强对比/亮度抖动 + 高概率模糊（对应 DINO 的 global_transfo1 + GaussianBlur(p=1.0)）
      • 路径B：弱模糊 + 伽马变化（对应 DINO 的 global_transfo2 中的轻模糊 + 太阳化的强度近似）
    - 保留 CoarseDropout 作为 Cutout 类似正则
    - 禁用水平翻转以避免“左/右”标签语义错误（如脑血管部位）
    """
    global_scale = (0.25, 1.0)  # DINO 常用的全局裁剪比例区间
    local_like_pipelines = A.OneOf([
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
            A.GaussianBlur(blur_limit=(3, 9), sigma_limit=(0.1, 2.0), p=1.0),
        ]),
        A.Compose([
            A.GaussianBlur(blur_limit=(3, 9), sigma_limit=(0.1, 2.0), p=0.1),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
        ]),
    ], p=1.0)

    return A.Compose([
        # 多尺度随机裁剪到目标尺寸，匹配 DINO 的 RandomResizedCrop 行为
        A.RandomResizedCrop(height=CFG.size, width=CFG.size, scale=global_scale, ratio=(0.75, 1.3333333333)),
        # 医学任务含左右标签，禁用水平翻转以避免标签语义错误（如需启用，需同步交换左右相关标签）
        # A.HorizontalFlip(p=0.0),
        # DINO 两条强度路径的近似（OneOf 强制二选一）
        local_like_pipelines,
        # 轻度高斯噪声
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        # Cutout 风格的随机丢失区域，按尺寸自适应
        A.CoarseDropout(max_holes=8, max_height=CFG.size // 24, max_width=CFG.size // 24, p=0.5),
        # 归一化 + 转张量
        A.Normalize(),
        ToTensorV2(),
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
                self.series_paths.append(os.path.join(series_dir, f'{uid}.npy'))
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
        if self.use_3d:
            volume = volume.unsqueeze(0)  # 添加一个额外的维度以适应 3D CNN
        
        # 将标签转换为 tensor
        labels_tensor = torch.tensor(labels, dtype=torch.float32).squeeze(0)
        
        return volume, labels_tensor