import src.numpy as np


def get_positional_embeddings(n_w: int, dimension: int) -> np.ndarray:
    """Create 1D positional embeddings for the n_w columns (after vertical mean).

    Args:
        n_w: number of patches (columns) in width direction after vertical mean.
        dimension: dimension of the position embedding for a single feature vector.

    Returns:
        computed position embedding. Shape: [n_w, dimension]
    """
    assert dimension % 2 == 0, "Dimension must be even for sin/cos split."
    
    # 출력 형태: [n_w, dimension]
    pos_embed = np.zeros((n_w, dimension))
    
    # sin/cos 항을 계산하는 부분은 유지
    div_term = np.exp(np.arange(0, dimension, 2) * -(np.log(10000.0) / dimension))

    # 높이(y) 반복문을 제거하고, 너비(x)에 대해서만 반복
    for x in range(n_w):
        # 짝수 인덱스 (0::2)는 sin 대신 cos(x) 사용
        pos_embed[x, 0::2] = np.cos(x * div_term) 
        
        # 홀수 인덱스 (1::2)는 sin(x) 사용
        pos_embed[x, 1::2] = np.sin(x * div_term) 
        
    # 결과 형태는 이미 [n_w, dimension]이므로 추가 flatten 불필요
    return pos_embed
