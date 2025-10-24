import src.numpy as np
from src.layers.linear import Linear
from src.layers.parameter import Parameter
from src.utils.patch import convert_image_to_patches
from src.utils.position_embedding import get_positional_embeddings
from src.layers.vit_block import ViTBlock
from src.data.output_encoder import output_size


def ctc_greedy_decode(output_logits: np.ndarray, blank_idx: int = 0) -> list[int]:
    """
    CTC Greedy Decoding을 수행하여 인덱스 시퀀스를 반환합니다.
    Args:
        output_logits: 단일 인스턴스의 Logits (T, C).
    Returns:
        list[int]: 디코딩된 문자 인덱스 시퀀스 (Blank 및 반복 제거됨).
    """
    if output_logits.ndim == 2:
        output_logits = np.expand_dims(output_logits, axis=0)  # (1, T, C)
    
    batch_decoded = []
    for sample_logits in output_logits:
        best_path = np.argmax(sample_logits, axis=-1)  # Shape (T,)
        decoded = []
        prev_char = None
        for char_idx in best_path:
            if char_idx != blank_idx and char_idx != prev_char:
                decoded.append(int(char_idx))
            prev_char = char_idx
        batch_decoded.append(decoded)
    return batch_decoded

def calculate_cer(decoded_sequence: list[int], target_sequence: list[int]) -> float:
    """
    Levenshtein Distance를 사용하여 Character Error Rate (CER)을 계산합니다.
    (간단한 NumPy 환경을 위해 Levenshtein Distance 계산 코드를 포함합니다.)
    """
    n, m = len(decoded_sequence), len(target_sequence)
    if n == 0 and m == 0: return 0.0
    if m == 0: return 1.0 # Target이 빈 문자열이면 오류율 100%

    # 편집 거리 (Edit Distance) 테이블 초기화
    dp = np.zeros((n + 1, m + 1), dtype=np.int64)
    for i in range(n + 1): dp[i, 0] = i
    for j in range(m + 1): dp[0, j] = j

            
    edit_distance = dp[n, m]
    return edit_distance / m # CER = Edit Distance / Target Length

class ViT:
    """Vision Transformer"""

    def __init__(self, chw: tuple, n_patches: int, hidden_d: int, n_heads: int, num_blocks: int, out_classses: int):
        """Initialize.

        Args:
            chw: dimension (C H W).
            n_patches: number of patches.
            hidden_d: hidden dimension.
            n_heads: number of heads.
            num_blocks: number of blocks.
            out_classses: total number of output classes.
        """
        self.chw = chw
        self.n_patches_h, self.n_patches_w = n_patches
        self.patch_size = (
            chw[1] // self.n_patches_h,   # 세로 크기
            chw[2] // self.n_patches_w    # 가로 크기
        )
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.hidden_d = hidden_d
        self.linear_mapper = Linear(self.input_d, self.hidden_d)
        self.pos_embed = get_positional_embeddings(self.n_patches_w, self.hidden_d)
        self.blocks = [ViTBlock(hidden_d, n_heads) for _ in range(num_blocks)]
        self.mlp = Linear(self.hidden_d, out_classses)
        self.class_token = Parameter(np.random.rand(1, self.hidden_d))


    def forward(self, images: np.ndarray) -> np.ndarray:
        """Forward propagation.

        Args:
            images: input array.

        Returns:
            computed linear layer output.
        """
        patches = convert_image_to_patches(
            images,
            self.n_patches_h,
            self.n_patches_w,
            self.patch_size[0],
            self.patch_size[1]
        )
        tokens = self.linear_mapper(patches)
        out = tokens + self.pos_embed
        for block in self.blocks:
            out = block.forward(out)

        out = self.mlp(out)
        return out

    def set_optimizer(self, optimizer_algo: object) -> None:
        """Set optimizer.

        Args:
            optimizer: optimizer.
        """
        self.linear_mapper.set_optimizer(optimizer_algo)
        for block in self.blocks:
            block.set_optimizer(optimizer_algo)
        self.mlp.set_optimizer(optimizer_algo)

    def backward(self, error: np.ndarray) -> np.ndarray:
        """Backward propagation.

        Args:
            grad: represents the gradient w.r.t. the output. Defaults to None.

        Returns:
            the gradients w.r.t. the input.
        """
        error = self.mlp.backward(error)

        for block in self.blocks[::-1]:
            error = block.backward(error)
        removed_cls = error[:, 1:, :]
        _ = self.linear_mapper.backward(removed_cls)
        self.class_token.backward(error[:, 0, :])

    def update_weights(self) -> None:
        """Update weights based on the calculated gradients."""
        self.mlp.update_weights()
        for block in self.blocks[::-1]:
            block.update_weights()
        self.linear_mapper.update_weights()
        self.class_token.update_weights()
    
    def accuracy(self, imgs_batch: np.ndarray, answer_batch: np.ndarray, answer_length_batch: np.ndarray) -> float:
        """
        배치에 대한 CER (1 - Accuracy)을 계산합니다.
        """
        # 1. Logits 예측
        # y_hat: (B, T, C) - Logits (또는 Log_P)가 이미 계산된 경우
        # Logits가 아닌 Softmax 확률이 필요하다면 여기서 Softmax를 적용
        y_hat = self.forward(imgs_batch) 
        
        batch_cer = []
        B, T, C = y_hat.shape

        for b in range(B):
            # 2. Greedy Decoding
            decoded_indices = ctc_greedy_decode(y_hat[b], blank_idx=output_size)
            
            # 3. Target Sequence 추출 (패딩 제거)
            L = int(answer_length_batch[b].item())
            target_indices = answer_batch[b:(b+L-1)].tolist()
            
            # 4. CER 계산
            cer = calculate_cer(decoded_indices, target_indices)
            batch_cer.append(cer)
        # 5. 평균 CER을 반환 (1 - CER이 Accuracy가 됩니다)

        mean = 0.0
        for v in batch_cer:
            mean += v
        mean /= len(batch_cer)

        return 1.0 - mean

    def save_weights(self, filepath: str) -> None:
        """Save weights to a file.

        Args:
            filepath: path to the file.
        """
        weights = {
            "linear_mapper": self.linear_mapper.get_weights(),
            "blocks": [block.get_weights() for block in self.blocks],
            "mlp": self.mlp.get_weights(),
            "class_token": self.class_token.get_weights()
        }
        import pickle
        with open(filepath, "wb") as f:
            pickle.dump(weights, f)
    
    def load_weights(self, filepath: str) -> None:
        """Import weights from a file.

        Args:
            filepath: path to the file.
        """
        import pickle
        with open(filepath, "rb") as f:
            weights = pickle.load(f)
        
        self.linear_mapper.set_weights(weights["linear_mapper"])
        for block, block_weights in zip(self.blocks, weights["blocks"]):
            block.set_weights(block_weights)
        self.mlp.set_weights(weights["mlp"])
        self.class_token.set_weights(weights["class_token"])