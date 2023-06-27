#import
import pandas as panda 
import cv2
import time
from datetime import datetime

#initialize
initialState = None
motionTrackList= [ None, None ] 
motionTime = []
dataFrame = panda.DataFrame(columns = ["Initial", "Final"])

#capture start
video = cv2.VideoCapture(0)

#catch frames
while True:  

   # frame creating and reading  
   check, cur_frame = video.read()  
   var_motion = 0  
   gray_image = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)  
   gray_frame = cv2.GaussianBlur(gray_image, (21, 21), 0)

   #assign grayFrame to initialState if its none
   if initialState is None:  
       initialState = gray_frame  
       continue  
      
   #difference 
   differ_frame = cv2.absdiff(initialState, gray_frame)  

   #change between static or initial background 
   thresh_frame = cv2.threshold(differ_frame, 30, 255, cv2.THRESH_BINARY)[1]  
   thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)  

   #аor the moving object in the frame finding the coutours 
   cont,_ = cv2.findContours(thresh_frame.copy(),   
                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
   for cur in cont:  
       if cv2.contourArea(cur) < 1000:  
           continue  
       var_motion = 1  
       (cur_x, cur_y,cur_w, cur_h) = cv2.boundingRect(cur)
       
       #red rectangle around object 
       cv2.rectangle(cur_frame, (cur_x, cur_y), (cur_x + cur_w, cur_y + cur_h), (255,0,0), 3)  


   #motion status   
   motionTrackList.append(var_motion)  
   motionTrackList = motionTrackList[-2:]  

   #start time of the motion 
   if motionTrackList[-1] == 1 and motionTrackList[-2] == 0:  
       motionTime.append(datetime.now())  

   #end time of the motion 
   if motionTrackList[-1] == 0 and motionTrackList[-2] == 1:  
       motionTime.append(datetime.now())   

   # Through the colour frame displaying the contour of the object
   cv2.imshow("show:", cur_frame)   

   #key  
   wait_key = cv2.waitKey(1)    
   # With the help of the 'm' key ending the whole process of our system   
   if wait_key == ord('m'):  
       # adding the motion variable value to motiontime list when something is moving on the screen  
       if var_motion == 1:  
           motionTime.append(datetime.now())  
       break

#adding the time of motion or var_motion inside the data frame  
for a in range(0, len(motionTime), 2):  
   dataFrame = dataFrame.append({"Initial" : time[a], "Final" : motionTime[a + 1]}, ignore_index = True) 
# To record all the movements, creating a CSV file  
dataFrame.to_csv("Record.csv")  
# Releasing the video   
video.release() 
#closing or destroying all the open windows
cv2.destroyAllWindows()
