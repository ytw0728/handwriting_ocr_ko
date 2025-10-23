import json
from .dataset_label import Label
from pydantic import ValidationError


split = 'train'
sample = "IMG_OCR_53_4PR_09180"

def load_label(split: str, label_path: str) -> Label:
    label_filename = f"data/{split}/labels/{label_path}.json"
    with open(label_filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    # parse the label using Pydantic
    try:
        label = Label(**data)
        # print(label)
        # print(label.Annotation)
        # print(label.Dataset)
        # print(label.Dataset.category)
        # print(label.Dataset.identifier)
        # print(label.Images)
        # print('writer age:', label.Images.writer_age)
        # print('writer sex:', label.Images.writer_sex)
        # print(f'{len(label.bbox)} bbox')
        return label
    except ValidationError as e:
        print("Validation errors:")
        print(e)
