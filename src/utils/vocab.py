import src.numpy as np

# 특수 토큰 정의
# 0번 인덱스는 CTC Blank Token으로 예약합니다.
BLANK_TOKEN = '<blank>'
# 1번 인덱스는 인식을 못 한 문자를 위한 Unknown Token으로 예약합니다.
UNKNOWN_TOKEN = '<unk>'

class CharacterVocabulary:
    def __init__(self, vocab_file_path: str):
        """
        문자 파일로부터 Vocab을 로드하고, CTC 학습에 필요한 매핑을 생성합니다.
        
        Args:
            vocab_file_path: 문자 목록이 담긴 파일 경로 (예: 'chars.txt')
        """
        # 0: <blank>, 1: <unk> 를 제외한 나머지 문자를 로드합니다.
        
        with open(vocab_file_path, 'r', encoding='utf-8') as f:
            # 줄바꿈 및 공백을 제거하고 유니크한 문자만 추출 (중복 방지)
            raw_chars = [c.strip() for c in f.readlines() if c.strip()]
            unique_chars = sorted(list(set(raw_chars)))

        # 1. 인덱스 -> 문자 (디코딩용)
        self.idx_to_char = {0: BLANK_TOKEN, 1: UNKNOWN_TOKEN}
        # CTC Blank Token과 Unknown Token 다음인 2번부터 문자를 할당
        for i, char in enumerate(unique_chars):
            self.idx_to_char[i + 2] = char

        # 2. 문자 -> 인덱스 (인코딩/학습용)
        self.char_to_idx = {char: idx for idx, char in self.idx_to_char.items()}
        
        # 3. 최종 클래스 개수 (ViT Linear Layer의 out_features)
        self.num_classes = len(self.idx_to_char)
        
        # print(f"Vocabulary loaded. Total classes (including <blank>, <unk>): {self.num_classes}")

    def encode(self, text: str, max_length: int) -> np.ndarray:
        """
        텍스트를 정수 인덱스 시퀀스로 변환합니다.
        
        Args:
            text: 정답 문자열 (예: "안녕하세요")
            max_length: 정답 시퀀스의 최대 길이 (패딩을 위해 사용)
            
        Returns:
            np.ndarray: 패딩된 정수 인덱스 배열 (max_length,)
        """
        # 문자열을 인덱스로 변환하며, 없는 문자는 UNKNOWN_TOKEN(1)로 처리
        indices = [self.char_to_idx.get(char, self.char_to_idx[UNKNOWN_TOKEN]) for char in text]
        
        # 최대 길이에 맞게 패딩 (CTC에서는 보통 0번 인덱스인 <blank> 대신, 
        # 사용되지 않는 다른 값(예: -1) 또는 일반적인 패딩 인덱스(1)를 사용 후 
        # CTC Loss 계산 시 실제 시퀀스 길이를 별도로 제공합니다.)
        
        # 여기서는 편의상 UNKNOWN_TOKEN 인덱스(1)로 패딩합니다.
        # 실제 CTC 구현에서는 패딩 인덱스를 Blank Token(0)과 헷갈리지 않게 관리해야 합니다.
        
        if len(indices) > max_length:
            indices = indices[:max_length]

        padding_needed = max_length - len(indices)
        if padding_needed > 0:
            # 여기서는 UNK 토큰 인덱스(1)로 패딩하고, 실제 길이를 별도로 관리한다고 가정
            padded_indices = indices + [self.char_to_idx[UNKNOWN_TOKEN]] * padding_needed
        else:
            padded_indices = indices

        return np.array(padded_indices, dtype=np.int32)