import string

# 1. 문자 집합 정의
# 한글 완성형 범위
korean_start = 0xAC00
korean_end = 0xD7A3
korean_chars = [chr(i) for i in range(korean_start, korean_end + 1)]

# 영어 + 숫자 + 특수문자
english_digits = list(string.ascii_letters + string.digits + string.punctuation)

# 최종 문자 집합
characters = korean_chars + english_digits

# 2. 출력층 크기
output_size = len(characters)

# 3. 인덱스 ↔ 문자 매핑
idx_to_char = {i: ch for i, ch in enumerate(characters)}
char_to_idx = {ch: i for i, ch in enumerate(characters)}
