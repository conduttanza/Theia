#code by conduttanza
#
#created the 17/12/2025

#simple torch based change detection for image recognition
import torch
import time
from threading import Thread, Lock
import math
import numpy as np

#hand recognition imports
#
#using the mediapipe library 
#
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
    

#self made imports
from window_logic import Config, Logic
from inputs import Image
image = Image()
config = Config()
logic = Logic()

# Module-level state for change detection

class Hands_Reckon:
    
    def __init__(self):
        #time.sleep(Config.delay)
        self.stream_url = Config.stream_url
        self.cap = cv2.VideoCapture(self.stream_url or 0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot use camera")
        self.hand_landmarks = None
        self.ret = False
        self.frame = None
        self.running = True
        self.indexThumbDistance = 0
        self.hand_scale = 0
        self.newXcopy = 0
        self.newYcopy = 0
        self.new_side_x = 0
        self.new_side_y = 0
        self.handAngle = None
        self.lock = Lock
        Thread(target=self.update, daemon=True).start()
        
    def update(self):
        with mp_hands.Hands(
            model_complexity = 0, 
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5,
            max_num_hands = 2) as hands:
            while self.cap.isOpened() and self.running:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(config.delay)
                    print('Empty frame', '\n', 'skipping...')
                    continue
                frame.flags.writeable = False #improved performance
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
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
                self.ret = True
                self.frame = frame.copy()
                #time.sleep(Config.delay)
    
    def show_recon(self):
        if not self.ret:
            return
        if self.new_side_x and self.new_side_y:
            if (self.new_side_x - self.newXcopy) > config.size_tolerance and (self.new_side_y - self.newYcopy) > config.size_tolerance:
                size = [self.new_side_x, self.new_side_y]
            else:
                size = [self.newXcopy, self.newYcopy]
        else:
            size = [config.side_x, config.side_y]
        frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (size[0], size[1]))
        cv2.imshow('tracker', cv2.flip(frame_rgb, 1))
    
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
        self.new_side_y = int(x*(768/1366)) # y height calculated with the x length and its window ratio
    
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
        if self.middleWristDistance < 50  and self.indexThumbDistance < 10 and self.pinkyWristDistance > 175:
            #----------------------------------------------------------------------------------------#
            #print('middle ',self.middleWristDistance,' pinky' , self.pinkyWristDistance) 
            #----------------------------------------------------------------------------------------#
            #DEBUG print() - this is a contortion to not get interrupted while keeping it active
            logic.openWebApps()
        if self.middleWristDistance > 180  and self.indexThumbDistance > 150 and self.pinkyWristDistance < 60 and self.ringWristDistance < 60 and self.thumbWristDistance > 130:
            #----------------------------------------------------------------------------------------#
            #print('> 180 ',self.middleWristDistance,' > 150' , self.indexThumbDistance,' < 60' , self.pinkyWristDistance,' > 130' , self.thumbWristDistance, ) 
            #----------------------------------------------------------------------------------------#
            #DEBUG print() - this is a contortion to not get interrupted while keeping it active
            logic.writeText()
        else:
            #----------------------------------------------------------------------------------------#
            #print(' else middle ',self.middleWristDistance,' pinky' , self.pinkyWristDistance)
            #print('> 180 ',self.middleWristDistance,' > 150' , self.indexThumbDistance,' < 60' , self.pinkyWristDistance,' > 130' , self.thumbWristDistance, ) 
            #----------------------------------------------------------------------------------------#
            #DEBUG print()
            pass
        
    def gimbalReader(self):
        #print('imgreclogic im activating')
        self.handAngle = logic.gimbalReader(self.hand_landmarks)
        if self.handAngle != None:
            #print('angle (in rad): ',round(self.handAngle,4))
            math.radians(self.handAngle)
            gimbalx = math.cos(self.handAngle) * config.gimBallRadius
            gimbaly = math.sin(self.handAngle) * config.gimBallRadius
            return gimbalx,gimbaly
        else:
            return 1, 1
    
    def returnLandmarks(self):
        #print('sending landmarks')
        return self.hand_landmarks if self.ret else None
    
    def stop(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
'''

    
def handRecognition():
    #time.sleep(Config.delay) 
    
    cap = cv2.VideoCapture(Config.stream_url or 0)
    with mp_hands.Hands(
        model_complexity = 0, 
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print('Empty frame, ', '\n', 'Ignoring...')
                continue
            
            # To de-improve performance, optionally mark the image as writeable to
            # pass by reference.
            # 
            
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = True
            
            
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = hands.process(frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    hand_landmarks.landmark[8] 
                    hand_landmarks.landmark[4]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('MediaPipe Hands', cv2.flip(rgb_frame, 1))
            #process_frame(rgb_frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()


'''


old_frame = None
   
def process_frame(frame):
    """Process a single frame for change detection.

    Call this from your main loop and pass the latest frame. This avoids
    instantiating Image() repeatedly (which opens the camera multiple times).
    """
    global old_frame
    if frame is None:
        return

    if old_frame is None:
        old_frame = frame.copy()
        return

    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    old_frame_tensor = torch.from_numpy(old_frame).permute(2, 0, 1).float() / 255.0
    diff = torch.abs(frame_tensor - old_frame_tensor)
    if diff.mean().item() > Config.threshold_value:
        print("Significant change detected in the frame.")
    else:
        print("No significant change.")
    old_frame = frame.copy()
