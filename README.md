# Korean Handwriting OCR 

## Prerequisites

- [mise-en-place](https://mise.jdx.dev/)

## How to

```bash
// init with mise or install uv manually
$ mise trust
$ mise install

// install the environments
$ uv install

$ uv run train.py
$ uv run test.py
```

## Data Set

https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=605

```

./data/(train,test)/labels/*.json
./data/(train,test)/sources/*.png

```