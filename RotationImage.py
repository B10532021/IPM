
import cv2

def RotationImage(image, cameraInfo):
	rows, cols, _ = image.shape
	M = cv2.getRotationMatrix2D(
		(cameraInfo.opticalCenterX, cameraInfo.opticalCenterY), cameraInfo.roll, 1)
	image = cv2.warpAffine(image, M, (cols, rows))
	return image
