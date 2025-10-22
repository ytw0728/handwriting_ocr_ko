
file_path = "./range-of-hanja.txt"

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

hanja = [c for c in text if '\u4e00' <= c <= '\u9fff']

codepoints = [ord(c) for c in hanja]

min_cp = min(codepoints)
max_cp = max(codepoints)

count = max_cp - min_cp + 1

print(f"최소 코드포인트: U+{min_cp:X}")
print(f"최대 코드포인트: U+{max_cp:X}")
print(f"가능한 코드포인트 가짓수: {count}")
