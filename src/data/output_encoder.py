idx_to_char = {}
char_to_idx = {}

with open(r"src/utils/chars.txt", 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()
    idx_to_char = {idx: char for idx, char in enumerate(lines) if char}
    char_to_idx = {char: idx for idx, char in idx_to_char.items()}

output_size = len(idx_to_char.keys()) # 0 ~ output_size-1 = idx, output_size = blank
