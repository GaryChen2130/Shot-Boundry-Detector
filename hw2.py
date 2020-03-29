import os
import cv2
import copy
import math
import matplotlib.pyplot as plt

BIN_SIZE = 16
THRESHOLD = 10000
VIDEO_PATH = '.\\ironman.mpg'
#VIDEO_PATH = '.\\news.mpg'
#VIDEO_PATH = '.\\baseball.mpg'

def read_video(path):
	frames = []
	cap = cv2.VideoCapture(path)
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	for i in range(length):
		ret,frame = cap.read()
		frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
		frames.append(frame)
	return frames,fps


def compute_histogram(frame,bin_size):
	hist = [0]*bin_size
	bin_width = 256/bin_size
	for row in frame:
		for pixel in row:
			hist[int(pixel/bin_width)] += 1
	return hist


def compute_dis(hist1,hist2):
	dis = 0
	for i in range(len(hist1)):
		if hist2[i] > 0:
			dis += pow(hist1[i] - hist2[i],2)/hist2[i]
		else:
			dis += pow(hist1[i] - hist2[i],2)
	return dis


def detect_shots(video):
	frame_num = 0
	boundary = []
	dis_list = []
	for frame in video:
		hist = compute_histogram(frame,BIN_SIZE)
		frame_num += 1
		if frame_num == 1:
			hist_prev = copy.deepcopy(hist)
			continue
		dis = compute_dis(hist_prev,hist)
		dis_list.append(dis)
		if dis > THRESHOLD:
			print('frame number: ' + str(frame_num - 1) + ', distance: ' + str(dis))
			boundary.append(frame_num - 1)
		hist_prev = copy.deepcopy(hist)

	bins = [i for i in range(0,len(video) - 1)]
	plt.bar(bins,dis_list,width=10)
	plt.savefig('.\\output\\' + VIDEO_PATH[2:-4] + '_hist_' + str(BIN_SIZE) + '.jpg',bbox_inches = 'tight',pad_inches = 0.0)
	plt.show()
	plt.close()
	return boundary


if __name__ == '__main__':
	video,fps = read_video(VIDEO_PATH)
	print('fps: ' + str(fps))
	print('video length: ' + str(len(video)))
	boundry = detect_shots(video)
	print(boundry)
