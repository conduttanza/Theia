#code by conduttanza
#
#created the 17/12/2025

#imports
import torch
import time
from threading import Thread, Lock
import math
import numpy as np
import pygame

#mediapipe imports
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
    

#self made imports
from functions_logic import Config, Logic
from image_capture import Image
image = Image()
config = Config()
logic = Logic()

# Module-level state for change detection

class Hands_Reckon:
    
    def __init__(self):
        #time.sleep(Config.delay)
        self.hand_landmarks = None
        self.ret = False
        self.running = True
        self.whatToMove = []
        self.indexThumbDistance = 0
        self.hand_scale = 0
        self.newXcopy = 0
        self.newYcopy = 0
        self.new_side_x = 0
        self.new_side_y = 0
        self.handAngle = None
        self.lock = Lock()
        self.frame_is_rgb = False
        self.frame = image.get_frame()
        Thread(target=self.update, daemon=True).start()
        
    def update(self):
        with mp_hands.Hands(
            model_complexity = 0, 
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5,
            max_num_hands = 1) as hands:
            while self.running:
                frame = image.get_frame()
                if frame is None:
                    time.sleep(config.delay)
                    continue
                frame.flags.writeable = False #improved performance
                # OpenCV returns frames in BGR. MediaPipe expects RGB, so convert once here
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame)
                if results.multi_hand_landmarks:
                    #print('existent results')
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    #----------------------------------------------------------------------------------------#
                    self.hand_landmarks = hand_landmarks
                    #----------------------------------------------------------------------------------------#
                    self.Wrist = [self.hand_landmarks.landmark[0].x, self.hand_landmarks.landmark[0].y]
                    self.Thumb = [self.hand_landmarks.landmark[4].x, self.hand_landmarks.landmark[4].y]
                    self.I_finger = [self.hand_landmarks.landmark[8].x, self.hand_landmarks.landmark[8].y]
                    self.M_finger = [self.hand_landmarks.landmark[12].x, self.hand_landmarks.landmark[12].y]
                    self.R_finger = [self.hand_landmarks.landmark[16].x, self.hand_landmarks.landmark[16].y]
                    self.P_finger = [self.hand_landmarks.landmark[20].x, self.hand_landmarks.landmark[20].y]
                    #----------------------------------------------------------------------------------------#
                    if config.doImageScaling == True:
                        self.scaling()
                    if config.handCommands == True:
                        self.handCommands()
                    if config.doGimbalReader == True:
                        self.gimbalReader()
                    if config.doParticles == True:
                        #WIP
                        #self.particleSim()
                        pass
                    if config.tracker == True:
                        self.tracker()
                self.ret = True
                self.frame = frame.copy()
                self.frame_is_rgb = True
                #time.sleep(Config.delay)
    
    def showStream(self):         
        frame = self.frame
        if frame is None:
            return None
        # Prefer the most recently processed frame (already converted to RGB in update)
        if self.ret and self.frame is not None:
            frame = self.frame.copy()
            frame_rgb = frame if self.frame_is_rgb else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            ret, frame = self.cap.read()
            if not ret:
                return
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_rgb = cv2.flip(frame_rgb, 1)
        frame_rgb = cv2.resize(frame_rgb, (config.side_x, config.side_y))
        pygameSurface = pygame.image.frombuffer(
        frame_rgb.tobytes(),
        frame_rgb.shape[1::-1],
        'RGB'
        )
        return pygameSurface
    
    def tracker(self):
        handCenter = self.hand_landmarks.landmark[9] #middle finger metacarpus
        center = [config.side_x/2, config.side_y/2]
        distance_x = center[0]-handCenter.x*config.side_x
        distance_y = center[1]-handCenter.y*config.side_y
        #print(distance_x, distance_y)
        if distance_y > 50:
            self.move = 'up'
            self.speed = math.atan(distance_y/config.side_y)
            self.whatToMove = [self.move, self.speed]
        elif distance_y < -50:
            self.move = 'down'
            self.speed = -math.atan(distance_y/config.side_y)
            self.whatToMove = [self.move, self.speed]
        else:
            self.move = 'idle'
            self.speed = -math.atan(distance_y)/config.side_y
            self.whatToMove = [self.move, self.speed]
            
        
    
    def returnTrackerSpeed(self):
        return self.whatToMove
            
    def handCommands(self):
        #----------------------------------------------------------------------------------------#
        #GET THE DISTANCE OF EACH FINGER FROM THE THUMB
        #----------------------------------------------------------------------------------------#
        self.indexThumbDistance = math.sqrt(((self.Thumb[0]-self.I_finger[0])*config.side_x)**2+
            ((self.Thumb[1]-self.I_finger[1])*config.side_y)**2)
        self.middleThumbDistance = math.sqrt(((self.Thumb[0]-self.M_finger[0])*config.side_x)**2+
            ((self.Thumb[1]-self.M_finger[1])*config.side_y)**2)
        self.ringThumbDistance = math.sqrt(((self.Thumb[0]-self.R_finger[0])*config.side_x)**2+
            ((self.Thumb[1]-self.R_finger[1])*config.side_y)**2)
        self.pinkyThumbDistance = math.sqrt(((self.Thumb[0]-self.P_finger[0])*config.side_x)**2+
            ((self.Thumb[1]-self.P_finger[1])*config.side_y)**2)
        #----------------------------------------------------------------------------------------#
        #GET THE DISTANCE OF EACH FINGER FROM THE WRIST
        #----------------------------------------------------------------------------------------#
        self.thumbWristDistance = math.sqrt(((self.Wrist[0]-self.Thumb[0])*config.side_x)**2+
            ((self.Wrist[1]-self.Thumb[1])*config.side_y)**2)
        self.indexWristDistance = math.sqrt(((self.Wrist[0]-self.I_finger[0])*config.side_x)**2+
            ((self.Wrist[1]-self.I_finger[1])*config.side_y)**2)
        self.middleWristDistance = math.sqrt(((self.Wrist[0]-self.M_finger[0])*config.side_x)**2+
            ((self.Wrist[1]-self.M_finger[1])*config.side_y)**2)
        self.ringWristDistance = math.sqrt(((self.Wrist[0]-self.R_finger[0])*config.side_x)**2+
            ((self.Wrist[1]-self.R_finger[1])*config.side_y)**2)
        self.pinkyWristDistance = math.sqrt(((self.Wrist[0]-self.P_finger[0])*config.side_x)**2+
            ((self.Wrist[1]-self.P_finger[1])*config.side_y)**2)
        #----------------------------------------------------------------------------------------#
        #YES, IT IS REPEATED BUT, SHOULD BE BETTER IF OTHER FUNCTS DONT RUN IT TOO
        #also yes, all written by hand
        #----------------------------------------------------------------------------------------#
        mScale = self.hand_landmarks.landmark[9]
        hand_scale = [
            self.Wrist[0]*config.side_x, 
            self.Wrist[1]*config.side_y,
            mScale.x*config.side_x,
            mScale.y*config.side_y
            ]
        scale_for_hand = math.sqrt((hand_scale[0]-hand_scale[2])**2+(hand_scale[1]-hand_scale[3])**2)
        open = 1.5*scale_for_hand
        close = scale_for_hand
        '''
        print('below')
        print(self.thumbWristDistance)
        print(self.indexWristDistance)
        print(self.middleWristDistance)
        print(self.ringWristDistance)
        print(self.pinkyWristDistance)
        '''
        self.countOne = (self.thumbWristDistance > open and self.indexWristDistance < close and self.middleWristDistance < close and self.ringWristDistance < close and self.pinkyWristDistance < close)
        self.countTwo =  (self.thumbWristDistance > open and self.indexWristDistance > open and self.middleWristDistance < close and self.ringWristDistance < close and self.pinkyWristDistance < close)
        self.countThree =  (self.thumbWristDistance > open and self.indexWristDistance > open and self.middleWristDistance > open and self.ringWristDistance < close and self.pinkyWristDistance < close)
        self.countFour =  (self.thumbWristDistance < close and self.indexWristDistance > open and self.middleWristDistance > open and self.ringWristDistance > open and self.pinkyWristDistance > open)
        self.countFive =  (self.thumbWristDistance > open and self.indexWristDistance > open and self.middleWristDistance > open and self.ringWristDistance > open and self.pinkyWristDistance > open)
        
        self.openMidFinger = (self.middleWristDistance > open)

        text = None
        if self.countOne:
            text = 'one'
        if self.countTwo:
            text = 'two'
        if self.countThree:
            text = 'three'
        if self.countFour:
            text = 'four'
        if self.countFive:
            text = 'five'
        #print(text)
        logic.writeText(text)
    
    def scaling(self):
        self.indexThumbDistance = math.sqrt(
            ((self.Thumb.x-
            self.I_finger.x)*config.side_x)**2+
            ((self.Thumb.y-
            self.I_finger.y)*config.side_y)**2
            )
        #print(indexThumbDistance)
        mScale = self.hand_landmarks.landmark[9]
        self.hand_scale = [
            self.Wrist.x*config.side_x, 
            self.Wrist.y*config.side_y,
            mScale.x*config.side_x,
            mScale.y*config.side_y
            ]
        self.scale_for_hand = math.sqrt((self.hand_scale[0]-self.hand_scale[2])**2+(self.hand_scale[1]-self.hand_scale[3])**2)
        scale = self.indexThumbDistance/self.scale_for_hand
        x, y = logic.scaling(scale) # y would be the y height though not used to keep the best ratio
        self.newXcopy = self.new_side_x
        self.newYcopy = self.new_side_y
        self.new_side_x = int(x)
        self.new_side_y = int(x*(config.side_y/config.side_x)) # y height calculated with the x length and its window ratio
    
    def gimbalReader(self):
        #print('imgreclogic im activating')
        self.handAngle = logic.gimbalReader(self.hand_landmarks)
        if self.handAngle != None:
            #print('angle (in rad): ',round(self.handAngle,4))
            math.radians(self.handAngle)
            self.gimbalx = math.cos(self.handAngle) * config.gimBallRadius
            self.gimbaly = math.sin(self.handAngle) * config.gimBallRadius
            return self.gimbalx,self.gimbaly
        else:
            return 1, 1
    
    def returnLandmarks(self):
        #print('sending landmarks')
        return self.hand_landmarks if self.ret else None
    
    def stop(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()

            
class torchProcess:
    old_frame = None

    def __init__(self,frame):
        self.frame = frame
        self.old_frame = None
    
    def process_frame(self):
        """Process a single frame for change detection.

        Call this from your main loop and pass the latest frame. This avoids
        instantiating Image() repeatedly (which opens the camera multiple times).
        """
        if self.frame is None:
            return

        if self.old_frame is None:
            self.old_frame = self.frame.copy()
            return

        self.frame_tensor = torch.from_numpy(self.frame).permute(2, 0, 1).float() / 255.0
        self.old_frame_tensor = torch.from_numpy(self.old_frame).permute(2, 0, 1).float() / 255.0
        self.diff = torch.abs(self.frame_tensor - self.old_frame_tensor)
        if self.diff.mean().item() > Config.threshold_value:
            print("Significant change detected in the frame.")
        else:
            #print("No significant change.")
            pass
        self.old_frame = self.frame.copy()
