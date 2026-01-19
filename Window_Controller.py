#code by conduttanza
#
#created the 17/12/2025

#imports
import pygame, numpy as np, math

#local imports
import window_logic, inputs, Hand_Recognition_RasPi, PrimaryServo

Logic = window_logic.Logic()
config = window_logic.Config()
code = window_logic.Logic()
fps = config.fps
side_x = config.side_x
side_y = config.side_y
image = inputs
recognition = Hand_Recognition_RasPi
recognizer = recognition.Gestures()
gpio = PrimaryServo.GPIO()

def Point(input):
    coords = code.outPut(input)
    return coords

def Fingers(landmarks):
    
    Wrist = landmarks.landmark[0]
    Thumb = landmarks.landmark[4]   
    I_finger = landmarks.landmark[8]
    M_finger = landmarks.landmark[12]  
    R_finger = landmarks.landmark[16]
    P_finger = landmarks.landmark[20]
    
    return {
        'Wrist':Wrist,
        'Thumb':Thumb,
        'Index':I_finger,
        'Middle':M_finger,
        'Ring':R_finger,
        'Pinky':P_finger,
        }

def main():
    center = np.array((
        [side_x/2, side_y/2]
    ))
    pygame.init()
    pygame.display.set_caption('hand recognition stream')
    screen = pygame.display.set_mode((side_x, side_y), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    
    running = True
    R = G = B = 255
    last_landmarks = None
    current_size = (side_x,side_y)
    
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    print('quitting...')
            
            screen.fill((0,0,0))
            if config.stream == True:
                pygameSurface = recognizer.showStream()
                if pygameSurface is not None:
                    screen.blit(pygameSurface, (0,0))
                
            #LOGIC
            newSize = None #recognizer.imageHandScaling()
            if newSize and newSize != current_size:
                current_size = newSize
                screen = pygame.display.set_mode(current_size, pygame.RESIZABLE)
                        
            #hand recognizer
            #recognizer.showStream()
            landmarks = recognizer.returnLandmarks()
            
            if landmarks:
                last_landmarks = landmarks
            
            if last_landmarks:
                notableFingers = Fingers(last_landmarks)
            
            #xtremes of the hand
            if landmarks:
                
                #NOTABLE LANDMARKS i.e. fingertips, thumb, wrist..
                
                notableFingers = Fingers(landmarks)
                
                shape = np.array((
                    [config.side_x - notableFingers['Thumb'].x * side_x,notableFingers['Thumb'].y * side_y],
                    [config.side_x - notableFingers['Index'].x * side_x,notableFingers['Index'].y * side_y],
                    [config.side_x - notableFingers['Middle'].x * side_x,notableFingers['Middle'].y * side_y],
                    [config.side_x - notableFingers['Ring'].x * side_x,notableFingers['Ring'].y * side_y],
                    [config.side_x - notableFingers['Pinky'].x * side_x,notableFingers['Pinky'].y * side_y],
                    [config.side_x - notableFingers['Wrist'].x * side_x,notableFingers['Wrist'].y * side_y]
                ))
                
                shape = shape.astype(int)
                '''
                pygame.draw.polygon(screen,(0,G,0), shape, 3)
                pygame.draw.line(screen,(0,0,B),shape[0],shape[1],3)
                             
                for point in shape:
                    pygame.draw.circle(screen, (R,G,B), point, 10)
                    '''
            if config.doGimbalReader == True:
                
                gimbalx, gimbaly = recognizer.gimbalReader()
                
                if gimbalx != None and gimbaly != None:
                    gimbalCenter = np.array((
                        [config.side_x - (config.gimBallRadius + 10),config.side_y - (config.gimBallRadius + 10)]
                    ))
                    gimbalPoint = (gimbalx,-gimbaly)
                    gimbalPointSupp = (-gimbalx,gimbaly)
                    gimbalDownArrow = (gimbaly * config.gimbalDownArrowLen,gimbalx * config.gimbalDownArrowLen)
                    
                    pygame.draw.circle(screen,(0,0,0),gimbalCenter,config.gimBallRadius)
                    pygame.draw.line(screen,(R,G,B),gimbalPoint+gimbalCenter,gimbalCenter,3)
                    pygame.draw.line(screen,(R,G,B),gimbalPointSupp+gimbalCenter,gimbalCenter,3)
                    pygame.draw.line(screen,(R,0,0),gimbalDownArrow+gimbalCenter,gimbalCenter,2)
                    pygame.draw.circle(screen,(0,G,0),gimbalCenter,config.gimBallRadius,1)
                
                
            clock.tick(fps)
            pygame.display.flip()
            if running == False:
                gpio.cleanup()
                print('\n'+'pygame quit successfully')

    except KeyboardInterrupt:
        pygame.quit()
        gpio.cleanup()
        print('\n'+'kb quit successful')
        running = False



main()