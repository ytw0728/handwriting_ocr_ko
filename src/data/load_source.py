import src.numpy as np
from PIL import Image
from .dataset_label import Label
from src.training.output_encoder import char_to_idx

def load_source(split: str, label: Label, output_size:int, max_answer_length:int, resize=(256, 128), include_hanja=False, max_length=40) -> tuple[np.ndarray[np.float64, np.dtype[np.float64]], np.ndarray[(int,int)], np.ndarray[np.ndarray[int]]]:
    image_filename = f"data/{split}/sources/{label.Images.identifier}.png"
    pen_color = label.Images.pen_color
    pen_type = label.Images.pen_type

    src_img = Image.open(image_filename).convert('RGB')

    imgs: list[np.ndarray[np.float64]] = []
    metas: list[(int,int)] = []
    answers: list[list[int]] = []

    cnt = 0
    # 한 파일 당 40개 bbox만 처리
    for bbox in label.bbox:
        if include_hanja == False and any(not ch in char_to_idx for ch in bbox.data):
            continue

        cropped = src_img.crop((bbox.x[0], bbox.y[0], bbox.x[3], bbox.y[3]))
        resized = cropped.resize(resize)
        numpied = np.array(resized).transpose(2, 0, 1) / 255.0

        imgs.append(numpied)
        metas.append((pen_color, pen_type))

        ans = np.zeros((max_answer_length, output_size), dtype=np.float32)
        for i, ch in enumerate(bbox.data[:max_answer_length]):
            if ch in char_to_idx:
                ans[i, char_to_idx[ch]] = 1.0

        answers.append(ans.flatten())

        cnt += 1
        if cnt >= max_length:
            break

    pad = np.zeros((3, resize[1], resize[0]), dtype=np.float64)
    imgs.extend([pad.copy() for _ in range(max_length - len(imgs))])
    metas.extend([(-1, -1) for _ in range(max_length - len(metas))])
    answers.extend(
        np.array([
            (np.zeros(max_answer_length * output_size)) for _ in range(max_length - len(answers))
        ])
    )

    return np.array(imgs, dtype=np.float64), np.array(metas), np.array(answers)
