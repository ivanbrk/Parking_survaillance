import glob
import cv2
import geopandas as gpd 
from shapely.geometry import Point 
from shapely.affinity import scale
import torch
import numpy as np 
import pandas as pd 
import time
import warnings
warnings.filterwarnings('ignore')

model_path = "best.pt"
shape_path = "shp/slots.shp"
input_video_path = "video/video_input.mp4"
output_video_path = "video/video_output.mp4"

threshold = 0.5 # threshold for yolo confidence score

slots = gpd.read_file(shape_path) #read slots geometry from shapefile

slots['geometry'] = slots.apply(lambda row: scale(row.geometry, xfact=1.0, yfact=-1.0, zfact=1.0, origin=(0,0)), axis = 1) #convert world coordinates (origin left lower) to image coordinates (origin left upper)
slots['center'] = slots.apply(lambda row: row.geometry.centroid, axis = 1) #define slots center point as geometry


model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path) #load yolo weights already trained for cars from UAVs

vidcap = cv2.VideoCapture(input_video_path) # load input_video


fourcc = cv2.VideoWriter_fourcc(*'mp4v') # define encoding
video=cv2.VideoWriter(detected_video_path, fourcc, vidcap.get(5), (vidcap.get(3),vidcap.get(4))) # define parameters of output_video (path, encoding, fps, shape)

j = 0 # initialize j just to know where script is come from
while True:

	st = time.time() #start time
	success,image = vidcap.read() # read next frame of input video

	image1 = model(image) # find cars on loaded image
	df = image1.pandas().xyxy[0] #present detected cars on image by dataframe

	df = df.loc[df.confidence>=threshold] # filter cars with confidence larger than threshold

	df['geom'] = df.apply(lambda row: Point(row.xmin+((row.xmax - row.xmin)/2), row.ymin+((row.ymax - row.ymin)/2)), axis = 1) # define centroid of detected cars as point geometry

	df = df[['confidence', 'geom', 'name']] # filter only main values for every detected car

	cars = gpd.GeoDataFrame(df, geometry = df['geom'], crs = "EPSG:4326") # define geodataframe of detected cars with point geometry
	cars.drop("geom", inplace = True, axis = 1) #drop geom column

	joined = gpd.sjoin(cars, slots, 'left') # spatial join between cars and slots by left join

	joined['colors'] = joined.apply(lambda row: (0,0,255) if pd.isnull(row.id)==True else (255,0,0), axis = 1) # define blue color if car is in slot else red color if car does not in any slot
	joined.drop(["index_right", "center"], axis = 1, inplace = True) # drop unnecessary columns
	empty_slots = slots.loc[~slots.id.isin(list(joined.id.unique()))] # filter slots which does not have any car - empty slots

	joined.reset_index(drop = True, inplace = True) # reset inidices in joined gdf

	empty_slots['colors'] = empty_slots.apply(lambda row: (0,255,0) , axis = 1) # set green color for empty slots
	empty_slots.drop(["geometry"], axis = 1, inplace = True) #drop geomtry column from empty slots
	empty_slots.reset_index(drop = True, inplace = True) #reset empty slots indices
	
	#visualize cars by cv2 circle
	for i in range(0,len(joined)):

		xx = int(joined.at[i,'geometry'].x)
		yy = int(joined.at[i,'geometry'].y)
		col = joined.at[i,'colors']

		cv2.circle(image,(xx,yy), 7, col, -1)

	#visualize empty slots by cv2 circle

	for i in range(0,len(empty_slots)):

		xx = int(empty_slots.at[i,'center'].x)
		yy = int(empty_slots.at[i,'center'].y)
		col = empty_slots.at[i,'colors']

		cv2.circle(image,(xx,yy), 7, col, -1)

	free_slots = len(empty_slots) #calculate number of empty slots
	unparked = len(joined.loc[joined.id.isnull()]) #calculate number of cars which are looking for slot
	parked = len(joined.loc[joined.id.isnull()==False]) #calculate number of parked cars

	#draw text with numbers of free slots, cars which are looking for slot and parked cars
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(image,"Free slots: "+str(free_slots),(10,20), font, 1,(0,255,0),2,cv2.LINE_AA)
	cv2.putText(image,"Looking for slot: "+str(unparked),(10,50), font, 1,(0,0,255),2,cv2.LINE_AA)
	cv2.putText(image,"Parked: "+str(parked),(10,80), font, 1,(255,0,0),2,cv2.LINE_AA)

	#add edited image to output video
	video.write(image)

	j = j+1

	sto = time.time() # stop time
	print(j, "/", vidcap.get(7), "|", np.round(sto-st,3), "s !") # print info about number of image in video and time for editing it
