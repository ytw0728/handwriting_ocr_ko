import src.numpy as np


def convert_image_to_patches(imgs, n_h, n_w, patch_h, patch_w) -> np.ndarray:
    """Convert image into patches.

    Args:
        images: input images.
        num_patches_1d: total number of patches in one direction, same is replicated in other.

    Returns:
        images converted to patches.
    """
    B, C, H, W = imgs.shape
    
    # 1. (B, C, H, W) -> (B, C, n_h, patch_h, n_w, patch_w)
    patches = imgs.reshape(B, C, n_h, patch_h, n_w, patch_w)
    
    # 2. (B, C, n_h, patch_h, n_w, patch_w) -> (B, n_h, n_w, C, patch_h, patch_w)
    #    패치 그리드 (n_h, n_w)를 앞으로, 패치 내부 (C, patch_h, patch_w)를 뒤로 보냄
    patches = patches.transpose(0, 2, 4, 1, 3, 5)
    # 3. (B, n_h, n_w, C, patch_h, patch_w) -> (B, n_h, n_w, D)
    #    패치 내부 차원을 하나로 평탄화 (D = C*patch_h*patch_w)
    patches_grid = patches.reshape(B, n_h, n_w, -1)
    # 4. 세로 축(n_h, axis=1)을 따라 평균 계산
    # (B, n_h, n_w, D) -> (B, n_w, D)
    vertical_mean = patches_grid.mean(axis=1)
    return vertical_mean
