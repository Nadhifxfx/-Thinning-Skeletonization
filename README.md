# Hit or Miss-Thinning & Skeletonization
```python
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import erosion, skeletonize, thin
from skimage.util import invert
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from google.colab import files
from PIL import Image
import io

# Fungsi Hit-or-Miss
def hit_or_miss(image, se_foreground, se_background):
    image_complement = invert(image)
    eroded_foreground = erosion(image, se_foreground)
    eroded_background = erosion(image_complement, se_background)
    return eroded_foreground & eroded_background

# Upload Gambar
uploaded = files.upload()

# Ambil file pertama yang diupload
filename = next(iter(uploaded))
img = Image.open(io.BytesIO(uploaded[filename])).convert('L')  # grayscale
img = np.array(img)

# Konversi ke citra biner
thresh = threshold_otsu(img)
binary_image = img > thresh

# Structuring Elements (SE) untuk Hit-or-Miss
se_foreground = np.array([[0, 1, 0],
                          [0, 1, 0],
                          [0, 0, 0]], dtype=bool)

se_background = np.array([[1, 0, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=bool)

# Proses Hit-or-Miss
hitmiss_result = hit_or_miss(binary_image, se_foreground, se_background)

# Proses Thinning & Skeletonization
thin_result = thin(binary_image)
skeleton_result = skeletonize(binary_image)

# Visualisasi
fig, axs = plt.subplots(1, 5, figsize=(18, 4))
axs[0].imshow(binary_image, cmap='gray')
axs[0].set_title("Original Binary Image")

axs[1].imshow(invert(binary_image), cmap='gray')
axs[1].set_title("Complement Image")

axs[2].imshow(hitmiss_result, cmap='gray')
axs[2].set_title("Hit-or-Miss Result")

axs[3].imshow(thin_result, cmap='gray')
axs[3].set_title("Thinning")

axs[4].imshow(skeleton_result, cmap='gray')
axs[4].set_title("Skeletonization")

for ax in axs:
    ax.axis('off')

plt.tight_layout()
plt.show()
```
**Output :**
![download](https://github.com/user-attachments/assets/4def1768-2b57-4c0c-b256-f90c656e70c2)

