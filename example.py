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
