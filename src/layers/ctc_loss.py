import src.numpy as np

def log_add_exp(log_a: np.float64, log_b: np.float64) -> np.float64:
    """Computes log(exp(log_a) + exp(log_b)) in a numerically stable way."""
    if log_a == -np.inf:
        return log_b
    if log_b == -np.inf:
        return log_a
    
    # Log-Sum-Exp Trick: max(a, b) + log(1 + exp(-|a - b|))
    max_val = max(log_a, log_b)
    min_val = min(log_a, log_b)
    # np.log1p(x) = log(1+x)는 x가 0에 가까울 때 더 정밀함
    return max_val + np.log1p(np.exp(min_val - max_val))

# Assuming CharacterVocabulary is loaded elsewhere and has ind_to_char/char_to_ind logic.
# For simplicity, we use the indices from the previous working context: BLANK=0, UNK=1, other classes > 1

class CTCLoss:
    """Computes the CTC loss and its gradient using the Forward-Backward algorithm (Log Domain)."""

    def __init__(self, blank_idx: int = 0) -> None:
        """Initialize.
        Args:
            blank_idx: Index of the CTC Blank Token (default: 0).
        """
        self.blank_idx = blank_idx
        self.cache = {}

    def _pad_label_to_prime(self, target_indices: np.ndarray, L: int) -> np.ndarray:
        """Converts [c1, c2, ...] to [b, c1, b, c2, b, ...]."""
        L_prime = 2 * L + 1
        targets_prime = np.full(L_prime, fill_value=self.blank_idx, dtype=np.int32)
        targets_prime[1::2] = target_indices[:L]
        return targets_prime

    def _forward_backward_single(self, log_P: np.ndarray, targets_prime: np.ndarray, T: int, L_prime: int):
        """Calculates alpha, beta, and the total log-likelihood for a single instance."""
        
        # -------------------- 1. Alpha (Forward) Calculation (Log Domain) --------------------
        alpha = np.full((T, L_prime), fill_value=-np.inf, dtype=np.float64)
        
        # t=0 초기화
        alpha[0, 0] = log_P[0, self.blank_idx]
        if L_prime > 1:
            # alpha[0, 1] = P(c1|t=0)
            alpha[0, 1] = log_P[0, targets_prime[1]]

        # 재귀 계산
        for t in range(1, T):
            for u in range(L_prime):
                # (1) 이전 상태 (u)에서 현재 u로 전이
                log_add_sum = alpha[t-1, u]

                # (2) 이전 상태 (u-1)에서 현재 u로 전이
                if u > 0:
                    log_add_sum = log_add_exp(log_add_sum, alpha[t-1, u-1])

                # (3) 이전 상태 (u-2)에서 현재 u로 전이 (blank skip / repeated char skip)
                if u > 1 and targets_prime[u] != self.blank_idx and targets_prime[u] != targets_prime[u-2]:
                    log_add_sum = log_add_exp(log_add_sum, alpha[t-1, u-2])
                
                # alpha[t, u] = log(sum of incoming paths) + log P(y_u|t)
                alpha[t, u] = log_add_sum + log_P[t, targets_prime[u]]

        # -------------------- 2. Beta (Backward) Calculation (Log Domain) --------------------
        beta = np.full((T, L_prime), fill_value=-np.inf, dtype=np.float64)
        
        # t=T-1 초기화
        beta[T-1, L_prime - 1] = 0.0 # log(1)
        if L_prime > 1:
            beta[T-1, L_prime - 2] = 0.0 # log(1)

        # 역방향 재귀 계산 (t=T-2 부터 0까지)
        for t in range(T - 2, -1, -1):
            for u in range(L_prime):
                # (1) 이전 상태 (u)에서 전이 (t+1, u)
                log_add_sum = beta[t+1, u]
                
                # (2) 이전 상태 (u+1)에서 전이 (t+1, u+1)
                if u < L_prime - 1:
                    log_add_sum = log_add_exp(log_add_sum, beta[t+1, u+1])

                # (3) 이전 상태 (u+2)에서 전이 (t+1, u+2)
                if u < L_prime - 2 and targets_prime[u] != self.blank_idx and targets_prime[u] != targets_prime[u+2]:
                    log_add_sum = log_add_exp(log_add_sum, beta[t+1, u+2])
                
                # beta[t, u] = log(sum of outgoing paths) + log P(y_u|t)
                beta[t, u] = log_add_sum + log_P[t, targets_prime[u]]
        
        # -------------------- 3. Total Log-Likelihood --------------------
        # Log P(y|X) = alpha[T-1, L'-1] + alpha[T-1, L'-2]
        log_P_y = log_add_exp(alpha[T-1, L_prime - 1], alpha[T-1, L_prime - 2] if L_prime > 1 else -np.inf)
        
        return alpha, beta, log_P_y


    def forward(self, y_pred: np.ndarray, y_true: np.ndarray, input_lengths: np.ndarray, target_lengths: np.ndarray) -> np.ndarray:
        """Computes the CTC loss (Negative Log Likelihood).

        y_pred: Logits (B, T, C).
        y_true: Padded target indices (B, L_max).
        input_lengths: Actual sequence lengths T (B,).
        target_lengths: Actual target lengths L (B,).
        """
        self.cache = {}
        
        # 💡 y_pred를 float64로 변환하여 수치적 안정성 확보
        y_pred = y_pred.astype(np.float64)
        
        batch_size, T_max, C = y_pred.shape
        
        # 1. Log Softmax 계산 (Log Probabilities)
        max_val = np.amax(y_pred, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(y_pred - max_val), axis=-1, keepdims=True)) + max_val
        log_P = np.log(np.exp(y_pred)) - log_sum_exp
        
        self.cache['y_pred'] = y_pred
        self.cache['log_P'] = log_P
        self.cache['C'] = C
        self.cache['target_lengths'] = target_lengths
        self.cache['input_lengths'] = input_lengths

        batch_losses = []
        
        # 2. 배치별로 CTC Forward 알고리즘 적용
        for b in range(batch_size):
            # T와 L을 순수 Python 정수로 추출 (이전 오류 해결을 위해)
            T = int(input_lengths[b].item())
            L = int(target_lengths[b].item())
            
            if L == 0:
                continue

            targets_prime = self._pad_label_to_prime(y_true, L)
            L_prime = 2 * L + 1
            log_P_b = log_P[b, :T]
            alpha, beta, log_P_y = self._forward_backward_single(log_P_b, targets_prime, T, L_prime)
            batch_losses.append(-log_P_y.item())
            
            # Backward를 위해 캐시
            self.cache[f'alpha_{b}'] = alpha
            self.cache[f'beta_{b}'] = beta
            self.cache[f'targets_prime_{b}'] = targets_prime
            self.cache[f'log_P_y_{b}'] = log_P_y
        return np.mean(np.array(batch_losses))


    def backward(self) -> np.ndarray:
        """Backward propagation: Computes the gradient w.r.t. the input logits (y_pred)."""
        
        y_pred = self.cache['y_pred']
        log_P = self.cache['log_P']
        input_lengths = self.cache['input_lengths']
        target_lengths = self.cache['target_lengths']
        C = self.cache['C']
        
        B, T_max, C = y_pred.shape
        gradients = np.zeros_like(y_pred, dtype=np.float64)

        for b in range(B):
            T = int(input_lengths[b].item())
            L = int(target_lengths[b].item())
            if L == 0: continue
            
            targets_prime = self.cache[f'targets_prime_{b}']
            L_prime = 2 * L + 1
            alpha = self.cache[f'alpha_{b}']
            beta = self.cache[f'beta_{b}']
            log_P_y = self.cache[f'log_P_y_{b}']
            
            # 1. Posterior Probability 계산 (Log Domain)
            # log(alpha[t, u]) + log(beta[t, u]) - log(P(y|X))
            log_posterior = alpha[:T] + beta[:T] - log_P_y
            
            # 2. Gradient 계산 (Logits에 대한 미분)
            # dL/d(y_tk) = P_t(k) - sum_{u: l'_u = k} exp(log(alpha_t(u)) + log(beta_t(u)) - log(P(y|X)))
            
            for t in range(T):
                # Probabilities P_t(k) (Logits의 Softmax)
                P_t = np.exp(log_P[b, t, :])
                
                # Sum of log-alpha-beta products for character k
                log_sum_k = np.full(C, fill_value=-np.inf, dtype=np.float64)
                
                # targets_prime의 인덱스 u와 k를 매핑
                for u in range(L_prime):
                    k = targets_prime[u]
                    # log_add_exp를 사용하여 log_sum_k[k]에 log_posterior[t, u]를 누적
                    log_sum_k[k] = log_add_exp(log_sum_k[k], log_posterior[t, u])
                
                # 3. 최종 Gradient 계산 (Formula 15/16의 로그 도메인 버전)
                # dL/d(y_tk) = P_t(k) - exp(log_sum_k[k])
                
                # log_sum_k가 -inf인 경우 (k가 targets_prime에 없는 경우) exp는 0이 됩니다.
                gradients[b, t, :] = P_t - np.exp(log_sum_k)

        # 모든 배치에 대한 평균 Gradient (1 / B)
        return gradients / B

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """Defining __call__ method to enable function like call."""
        return self.forward(*args, **kwargs)