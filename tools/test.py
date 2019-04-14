
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import patches

id = "P05979301-3752"
image_path = "/Users/jiangyy/voc2007.xoy/JPEGImages/{}.jpg".format(id)
box_path = "/Users/jiangyy/voc2007.xoy/labels/{}.txt".format(id)
# image_path = "/Users/jiangyy/Desktop/009167.jpg"
fig, ax = plt.subplots(1)
image = Image.open(image_path)
ax.imshow(image)
box = [155, 101, 179, 128]
# box = [600, 86, 610, 109]
rect = patches.Rectangle((box[0], box[1]), (box[2] - box[0]), (box[3] - box[1]), linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)
plt.show()
