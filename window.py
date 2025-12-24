#code by conduttanza
#
#created the 17/12/2025

#universal imports
import pygame, numpy as np, cv2, math


#self made imports
import window_logic, inputs, imagRec_logic

Logic = window_logic.Logic()
config = window_logic.Config()
code = window_logic.Logic()
fps = config.fps
side_x = config.side_x
side_y = config.side_y
image = inputs
recognition = imagRec_logic
recognizer = recognition.Hands_Reckon()

center = np.array((
    [side_x/2, side_y/2]
))

pygame.init()
pygame.display.set_caption('hand simulation')
screen = pygame.display.set_mode((side_x, side_y), pygame.RESIZABLE)
clock = pygame.time.Clock()

def Point(input):
    coords = code.outPut(input)
    return coords

running = True
R = G = B = 255

try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                print('quitting...')
            
        screen.fill((0,0,0))
        
        
        
        #webcam stream
        #image.Image.show_stream(screen)
        #frame = image.Image.get_frame()
        #if frame is not None:
            #recognition.process_frame(frame)
        
        #hand recon
        
        recognizer.show_recon()
        landmarks = recognizer.returnLandmarks()
        
        #xtremes of the hand
        if landmarks:
            
            Thumb = landmarks.landmark[4]
            #print(Thumb)
            tx = side_x - int(Thumb.x * side_x)
            ty = int(Thumb.y * side_y)
            tz = Thumb.z * side_x
            
            I_finger = landmarks.landmark[8]
            #print(I_finger)
            ix = side_x - int(I_finger.x * side_x)
            iy = int(I_finger.y * side_y)
            iz = I_finger.z * side_x
            
            M_finger = landmarks.landmark[12]
            #print(M_finger)
            mx = side_x - int(M_finger.x * side_x)
            my = int(M_finger.y * side_y)
            mz = M_finger.z * side_x
            
            R_finger = landmarks.landmark[16]
            #print(R_finger)
            rx = side_x - int(R_finger.x * side_x)
            ry = int(R_finger.y * side_y)
            rz = R_finger.z * side_x
            
            P_finger = landmarks.landmark[20]
            #print(P_finger)
            px = side_x - int(P_finger.x * side_x)
            py = int(P_finger.y * side_y)
            pz = P_finger.z * side_x
            
            Wrist = landmarks.landmark[0]
            #print(Wrist)
            wx = side_x - int(Wrist.x * side_x)
            wy = int(Wrist.y * side_y)
            wz = Wrist.z * side_x
            
            shape = np.array((
                [tx,ty],
                [ix,iy],
                [mx,my],
                [rx,ry],
                [px,py],
                [wx,wy]
            ))
            
            pygame.draw.polygon(screen,(0,255,0), shape, 3)
            pygame.draw.line(screen,(0,0,B),(tx,ty),(ix,iy),3)
            

            
            for point in shape:
                pygame.draw.circle(screen, (R,G,B), point, 10)
                
        if config.doGimbalReader == True:
            gimbalx, gimbaly = recognizer.gimbalReader()
            if gimbalx != None and gimbaly != None:
                
                gimbalPoint = (gimbalx,-gimbaly)
                gimbalPointSupp = (-gimbalx,gimbaly)
                gimbalDownArrow = (gimbaly * config.gimbalDownArrowLen,gimbalx * config.gimbalDownArrowLen)
                
                pygame.draw.line(screen,(255,255,255),gimbalPoint+center,center,3)
                pygame.draw.line(screen,(255,255,255),gimbalPointSupp+center,center,3)
                pygame.draw.line(screen,(255,0,0),gimbalDownArrow+center,center,2)
                pygame.draw.circle(screen,(0,255,0),center,config.gimBallRadius,1)
            
            
        clock.tick(fps)
        pygame.display.flip()
        if running == False:
            print('\n'+'pygame quit successfully')

except KeyboardInterrupt:
    pygame.quit()
    print('\n'+'kb quit successful')
    running = False