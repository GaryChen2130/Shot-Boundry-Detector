import os
import cv2
import copy
import math
import re
import argparse
import matplotlib.pyplot as plt

VIDEO_PATH = '.\\baseball.mpg'
GROUND_PATH = '.\\baseball_ground.txt'

BIN_SIZE = 8
THRESHOLD = 6000
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


def read_ground_truth(path):
	ground = []
	f = open(path,'r')
	lines = f.readlines()
	for i in range(4,len(lines)):
		line = lines[i].strip()
		line = re.split('~|-',line)
		if len(line) > 1:
			ground.append((eval(line[0]),eval(line[1])))
		else:
			ground.append((eval(line[0]),eval(line[0])))
	#print(ground)
	return ground


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
		hist = cv2.calcHist([frame],[0],None,[BIN_SIZE],[0.0,255.0])
		hist = flatten(hist)
		frame_num += 1
		if frame_num == 1:
			hist_prev = copy.deepcopy(hist)
			continue

		dis = compute_dis(hist_prev,hist)
		dis_list.append(dis)

		if (dis > THRESHOLD) and (dissolve_cnt <= HARDCUT_CHECK):
			#print('Hard Cut! frame number: ' + str(frame_num - 1) + ', distance: ' + str(dis))
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
				#print('Dissolve! frame number: ' + str(frame_num - 2) + ', distance: ' + str(dissolve_sum) + ', count: ' + str(dissolve_cnt))
				if (len(boundary) == 0) or (frame_num - boundary[-1][0] - 1 > cd):
					boundary.append((frame_num - 2,dissolve_sum))
				elif dissolve_sum > boundary[-1][1]:
					boundary[-1] = (frame_num - 2,dissolve_sum)
			dissolve_sum = 0
			dissolve_cnt = 0

		hist_prev = copy.deepcopy(hist)

	'''
	bins = [i for i in range(0,len(video) - 1)]
	plt.bar(bins,dis_list,width=10)
	plt.savefig('.\\output\\' + VIDEO_PATH[2:-4] + '_hist_' + str(BIN_SIZE) + '.jpg',bbox_inches = 'tight',pad_inches = 0.0)
	plt.show()
	plt.close()
	'''
	return boundary


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--video',help = 'path of input video')
	parser.add_argument('--ground',help = 'path of ground truth file')
	args = parser.parse_args()

	if args.video:
		VIDEO_PATH = args.video
	if args.ground:
		GROUND_PATH = args.ground

	video,fps = read_video(VIDEO_PATH)
	#print('fps: ' + str(fps))
	#print('video length: ' + str(len(video)))
	boundary = detect_shots(video,9)
	#print(boundary)

	tp = 0
	fp = 0
	ground = read_ground_truth(GROUND_PATH)
	match = [0]*len(ground)
	print('Detected shot boundaries:')
	for (frame_index,dis) in boundary:
		flag = False
		index = 0
		print(frame_index)
		for (index_min,index_max) in ground:
			if (match[index] == 0) and (frame_index >= index_min) and (frame_index <= index_max):
				tp += 1
				flag = True
				match[index] = 1
				break
			index += 1

		if flag is not True:
			fp += 1

	print('\nprecision: ' + str(round(tp/len(boundary),2)) + ', recall: ' + str(round(tp/len(ground),2)))

