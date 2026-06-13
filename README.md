# Simple-PP-DocLayoutV3

Simple document layout analysis based on [PP-DocLayoutV3 of PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3) with fewer dependencies. 

<div align="center">
<img width="800" height="1041" alt="detection result example" src="https://github.com/user-attachments/assets/bcff3168-9015-4db0-8296-f789b129070f" />
</div>

# Example

```python
from simplelayout import SimplePPDocLayoutV3

model = SimplePPDocLayoutV3()

paths = ["testimage.png"]

results = model.run(paths)

import matplotlib.pyplot as plt

# plot results
for path, result in zip(paths, results):
    plt.figure(figsize=(7, 10))

    plt.title(path)
    image = result["image"]
    plt.imshow(image)

    crops = []

    # plot detected objects in image
    for bbox, outline, score, label in zip(
        result["boxes"],
        result["outlines"],
        result["scores"],
        result["labels"]
    ):
        # position of detection result in image
        x0, y0, x1, y1 = bbox
        plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0])

        # crop detected region for future use
        crop = image.crop(bbox)
        crops.append(crop)

        # outline of result["mask"] (more precise than bbox for rotated text)
        x, y = outline.T
        plt.fill(x, y, alpha=0.1, edgecolor=(0, 0, 0, 0.5))

        # score describes certainty of detection result
        style = dict(fc="blanchedalmond", ec="orange", alpha=0.8)
        plt.text(x0, y0, f"{score:.2f}, {label}", bbox=style)
    plt.show()

    # show cropped regions
    for i, crop in enumerate(crops[:48], 1):
        plt.subplot(8, 6, i)
        plt.imshow(crop)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
```

The detected regions can then be used for e.g. Optical Character Recognition (OCR) with [Simple-GLM-OCR](https://github.com/99991/Simple-GLM-OCR).

```python
from simplelayout import SimplePPDocLayoutV3
from simpleglmocr import SimpleGlmOcr

layout_model = SimplePPDocLayoutV3()
ocr_model = SimpleGlmOcr()

result, = layout_model.run(["testimage.png"])

for bbox, label in zip(result["boxes"], result["labels"]):
    crop = result["image"].crop(bbox)

    text = ocr_model.run("Text Recognition:", crop)

    print(f"{label}:\n\n{text}")
    print("-" * 80)
```

# Installation

```bash
pip install torch numpy pillow matplotlib # matplotlib is optional, only used for example.py
git clone https://github.com/99991/Simple-PP-DocLayoutV3.git
cd Simple-PP-DocLayoutV3
python3 example.py
```
