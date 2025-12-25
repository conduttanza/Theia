#code by conduttanza
#
#created the 17/12/2025

import math, numpy as np
from threading import Lock
import webbrowser as wb
import pyautogui as pg
import time

class Config:
    
    #universal values
    stream_url = "http://192.168.1.3:8080/video"
    web_url = 'https://'
    threshold_value = 0.01   # threshold for change detection
    
    #config values
    fps = 30
    delay = 1 / (5*fps)
    side_x = 900
    side_y = int(side_x * (768/1366))
    size_tolerance = 100
    gimBallRadius = side_x / 10
    gimbalDownArrowLen = 1/5
    #FUNCTIONS TO ACTIVATE
    doImageScaling = False
    handCommands = True
    doGimbalReader = True
    

class Logic(Config):
    def __init__(self):
        self.side_x = Config.side_x
        self.side_y = Config.side_y
        self.app = None
            
    def scaling(self, scale):
        if self.side_x and self.side_y and scale and Config.doImageScaling == True:
            #print('ts is doing something')
            return self.side_x * scale, self.side_y * scale
        else:
            return self.side_x, self.side_y
        
    def openWebApps(self):
        #just call the function
        self.app = input('input the web app name: ').lower()
        No = 'no'
        Exit = 'exit'
        if self.app == No or self.app == Exit:
            return
        if self.app != None:
            url = Config.web_url + self.app
            print(url)
            wb.open(url, new=2)
            
    def writeText(self):
        print('cant write rn, ty for the patience')
    
    def gimbalReader(self, hand_lmks):
        if Config.doGimbalReader == True and hand_lmks != None:
            x_len = hand_lmks.landmark[20].x - hand_lmks.landmark[4].x
            y_len = hand_lmks.landmark[20].y - hand_lmks.landmark[4].y
            if x_len > 0.05 or x_len < -0.05:
                IndexThumbAngle = math.atan(y_len/x_len)# - math.pi/6
            else:
                x_len += 0.01
                IndexThumbAngle = math.atan(y_len/x_len)# - math.pi/6
            return IndexThumbAngle
        else:
            #print('no handlanmarks')
            pass