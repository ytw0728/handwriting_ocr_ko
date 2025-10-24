import src.numpy as np
from PIL import Image
from .dataset_label import Label
from src.data.output_encoder import char_to_idx

def load_source(split: str, label: Label, resize=(256, 128), max_length=40):
    image_filename = f"data/{split}/sources/{label.Images.identifier}.png"
    pen_color = label.Images.pen_color
    pen_type = label.Images.pen_type

    src_img = Image.open(image_filename).convert('RGB')

    imgs: list[np.ndarray[np.float64]] = []
    # metas: list[(int,int)] = []
    answers: list[int]= []
    answer_lengths: list[int] =[]

    cnt = 0
    # 한 파일 당 40개 bbox만 처
    for bbox in label.bbox:
        ans = []
        for ch in bbox.data:
            if ch in char_to_idx:
                ans.append(char_to_idx[ch])
            else:
                continue

        cropped = src_img.crop((bbox.x[0], bbox.y[0], bbox.x[3], bbox.y[3]))
        resized = cropped.resize(resize)
        numpied = np.array(resized).transpose(2, 0, 1) / 255.0

        imgs.append(numpied)
        # metas.append((pen_color, pen_type))


        answers.extend(ans)
        answer_lengths.append(len(ans))

        cnt += 1
        if cnt >= max_length:
            break

    # metas.extend([(-1, -1) for _ in range(max_length - len(metas))])

    return np.array(imgs, dtype=np.float64), np.array(answers, dtype=np.int32), np.array(answer_lengths, dtype=np.int32)