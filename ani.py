import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from vo import Visual_Odometry

plt.style.use("seaborn-whitegrid")


vo = Visual_Odometry("./data_1", "Orb", 500)
for index, (__, __) in enumerate(vo.RotationalAndTranslational()):
    for query, train in zip(vo.keypointsMatching()[0], vo.keypointsMatching()[1]):
        plt.plot(query[0], query[1], marker="o")
        plt.plot(train[0], train[1], marker="+")
    plt.savefig("./pictures/plot_{}.png".format(index))

import cv2
import numpy as np
import glob

img_array = []
for filename in glob.glob("./pictures/*.png"):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)


out = cv2.VideoWriter("project.avi", cv2.VideoWriter_fourcc(*"DIVX"), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
