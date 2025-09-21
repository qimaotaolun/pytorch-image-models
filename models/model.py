import timm
import torch
from torch.functional import F
from vit_pytorch.cross_vit import CrossViT

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.timm = CrossViT(
            image_size = 256,
            num_classes = 14,
            depth = 4,               # number of multi-scale encoding blocks
            sm_dim = 192,            # high res dimension
            sm_patch_size = 16,      # high res patch size (should be smaller than lg_patch_size)
            sm_enc_depth = 2,        # high res depth
            sm_enc_heads = 8,        # high res heads
            sm_enc_mlp_dim = 2048,   # high res feedforward dimension
            lg_dim = 384,            # low res dimension
            lg_patch_size = 64,      # low res patch size
            lg_enc_depth = 3,        # low res depth
            lg_enc_heads = 8,        # low res heads
            lg_enc_mlp_dim = 2048,   # low res feedforward dimensions
            cross_attn_depth = 2,    # cross attention rounds
            cross_attn_heads = 8,    # cross attention heads
            dropout = 0.1,
            emb_dropout = 0.1,
            channels= 32,
        )
        
    def forward(self, x):
        if x.shape[0] != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')
        x = self.timm(x)
        return x

if __name__ == '__main__':
    torch.manual_seed(42)
    
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 打印当前GPU的使用情况
    if device.type == 'cuda':
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
        print(f"Allocated Memory: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
        print(f"Cached Memory: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")
    else:
        print("CUDA not available, using CPU.")

    # Initialize model and move to appropriate device
    model = MyModel()
    model = model.to(device)

    # Load the saved model weights
    # model.load_state_dict(torch.load('model_best.pth'))

    # Set the model to evaluation mode
    model.eval()

    # Check memory usage before forward pass
    if device.type == 'cuda':
        print(f"Allocated Memory before operation: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
        print(f"Cached Memory before operation: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")
    
    # Create a dummy image tensor
    image = torch.randn(1, 32, 384, 384).to(device)

    # Run forward pass
    output = model(image)
    
    # Apply sigmoid activation
    print(F.sigmoid(output))

    # Check and print GPU memory usage after the operation
    if device.type == 'cuda':
        print(f"Allocated Memory after operation: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
        print(f"Cached Memory after operation: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")
