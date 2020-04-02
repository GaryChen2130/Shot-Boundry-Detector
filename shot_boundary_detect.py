import os
import cv2
import copy
import math
import matplotlib.pyplot as plt

BIN_SIZE = 8
THRESHOLD = 6000
VIDEO_PATH = '.\\ironman.mpg'
#VIDEO_PATH = '.\\news.mpg'
#VIDEO_PATH = '.\\baseball.mpg'
ZEROBOUND = 150
LOWERBOUND = 250
HARDCUT_CHECK = 5
DISSOLVE_CHECK = 9

def read_video(path):
	frames = []
	cap = cv2.VideoCapture(path)
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	for i in range(length):
		ret,frame = cap.read()
		frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		frames.append(frame)
	return frames,fps


def compute_histogram(frame,bin_size):
	hist = [0]*bin_size
	bin_width = 256/bin_size
	for row in frame:
		for pixel in row:
			hist[int(pixel/bin_width)] += 1
	return hist


def flatten(list_in):
	list_out = []
	for row in list_in:
		list_out.append(row[0])
	return list_out


def compute_dis(hist1,hist2):
	dis = 0
	for i in range(len(hist1)):
		if hist2[i] > 0:
			dis += pow(hist1[i] - hist2[i],2)/hist2[i]
		else:
			dis += pow(hist1[i] - hist2[i],2)
	return dis


def detect_shots(video,cd):
	frame_num = 0
	dissolve_sum = 0
	dissolve_cnt = 0
	boundary = []
	dis_list = []
	for frame in video:
		#hist = compute_histogram(frame,BIN_SIZE)
		hist = cv2.calcHist([frame],[0],None,[BIN_SIZE],[0.0,255.0])
		hist = flatten(hist)
		frame_num += 1
		if frame_num == 1:
			hist_prev = copy.deepcopy(hist)
			continue

		dis = compute_dis(hist_prev,hist)
		dis_list.append(dis)

		if (dis > THRESHOLD) and (dissolve_cnt <= HARDCUT_CHECK):
			print('Hard Cut! frame number: ' + str(frame_num - 1) + ', distance: ' + str(dis))
			if (len(boundary) == 0) or (frame_num - boundary[-1][0] > cd):
				boundary.append((frame_num - 1,dis))
			elif dis > boundary[-1][1]:
				boundary[-1] = (frame_num - 1,dis)
		
		elif dis > LOWERBOUND:
			dissolve_sum += dis
			dissolve_cnt += 1
			#print('frame number: ' + str(frame_num - 1) + ', distance: ' + str(dis))

		elif dis < ZEROBOUND:
			if dissolve_cnt >= DISSOLVE_CHECK:
				print('Dissolve! frame number: ' + str(frame_num - 2) + ', distance: ' + str(dissolve_sum) + ', count: ' + str(dissolve_cnt))
				if (len(boundary) == 0) or (frame_num - boundary[-1][0] - 1 > cd):
					boundary.append((frame_num - 2,dissolve_sum))
				elif dissolve_sum > boundary[-1][1]:
					boundary[-1] = (frame_num - 2,dissolve_sum)
			dissolve_sum = 0
			dissolve_cnt = 0

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
	boundry = detect_shots(video,9)
	print(boundry)
