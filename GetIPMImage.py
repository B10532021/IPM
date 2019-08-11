from GetInfo import GetInfo
from GetVanishingPoint import GetVanishingPoint
from TransformImage2Ground import TransformImage2Ground
from TransformGround2Image import TransformGround2Image
import cv2
import numpy as np


class Info(object):
    def __init__(self, dct):
        self.dct = dct

    def __getattr__(self, name):
        return self.dct[name]


I = cv2.imread('Images/pic1.jpg')
R = I[:, :, 1]
height = int(I.shape[0]) # row y
width = int(I.shape[1]) # col x

cameraInfo = Info({
    "focalLengthX": 700, # 1200.6831,         # focal length x
    "focalLengthY": 700, # 1200.6831,         # focal length y
    "opticalCenterX": int(width / 2), # 638.1608,        # optical center x
    "opticalCenterY": int(height / 2), # 738.8648,       # optical center y
    "cameraHeight": 1500, # 1879.8,  # camera height in `mm`
    "pitch": 15.5,           # rotation degree around x
    "yaw": 0.0,              # rotation degree around y
    "roll": 0              # rotation degree around z
})
ipmInfo = Info({
    "inputWidth": width,
    "inputHeight": height,
    "outWidth": int(width*3/4),
    "outHeight": int(height*3/4),
    "left": 128,
    "right": width-128,
    "top": 450,
    "bottom": height
})
# IPM
vpp = GetVanishingPoint(cameraInfo)
vp_x = vpp[0][0]
vp_y = vpp[1][0]
ipmInfo.top = float(max(int(vp_y), ipmInfo.top))
print(vp_y)
uvLimitsp = np.array([[vp_x, ipmInfo.right, ipmInfo.left, vp_x],
             [ipmInfo.top, ipmInfo.top, ipmInfo.top, ipmInfo.bottom]], np.float32)

xyLimits = TransformImage2Ground(uvLimitsp, cameraInfo)
row1 = xyLimits[0, :]
row2 = xyLimits[1, :]
xfMin = min(row1)
xfMax = max(row1)
yfMin = min(row2)
yfMax = max(row2)
xyRatio = int((xfMax - xfMin)/(yfMax - yfMin))
outImage = np.zeros((640, xyRatio*640), np.float32)
outRow = int(outImage.shape[0])
outCol = int(outImage.shape[1])
stepRow = (yfMax - yfMin)/outRow
stepCol = (xfMax - xfMin)/outCol
xyGrid = np.zeros((2, outRow*outCol), np.float32)
y = yfMax-0.5*stepRow

for i in range(0, outRow):
    x = xfMin+0.5*stepCol
    for j in range(0, outCol):
        xyGrid[0, (i-1)*outCol+j] = x
        xyGrid[1, (i-1)*outCol+j] = y
        x = x + stepCol
    y = y - stepRow

# TransformGround2Image
uvGrid = TransformGround2Image(xyGrid, cameraInfo)
# mean value of the image
means = np.mean(R)/255
RR = R.astype(float)/255
for i in range(0, outRow):
    for j in range(0, outCol):
        ui = uvGrid[0, i*outCol+j]
        vi = uvGrid[1, i*outCol+j]
        #print(ui, vi)
        if ui < ipmInfo.left or ui > ipmInfo.right or vi < ipmInfo.top or vi > ipmInfo.bottom:
            outImage[i, j] = means
        else:
            x1 = np.int32(ui)
            x2 = np.int32(ui+0.5)
            y1 = np.int32(vi)
            y2 = np.int32(vi+0.5)
            x = ui-float(x1)
            y = vi-float(y1)
            val = float(RR[y1, x1])*(1-x)*(1-y)+float(RR[y1, x2])*x*(1-y)+float(RR[y2, x1])*(1-x)*y+float(RR[y2, x2])*x*y
            outImage[i, j] = val

# show the result
while True:
    cv2.imshow('img', outImage)
    if cv2.waitKey(20) & 0xFF == 27:
        break
# save image
# cv2.imwrite('pic_ipm.jpg',outImage)
