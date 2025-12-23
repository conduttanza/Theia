#code by conduttanza
#
#created the 17/12/2025

import math, numpy as np
from threading import Thread, Lock
import webbrowser as wb

class Config:
    
    #universal values
    stream_url = None
    web_url = 'https://'
    threshold_value = 0.01   # threshold for change detection
    
    #config values
    fps = 30
    delay = 1 / (5*fps)
    side_x = 800
    side_y = int(side_x * (768/1366))
    size_tolerance = 100
    #FUNCTIONS TO ACTIVATE
    doImageScaling = True
    openWebApps = True

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
