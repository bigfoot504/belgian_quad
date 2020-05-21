'''
Generates some blobs and lets them float around randomly.
Blobs are intended to simulate UxVs.
This script renders one episode and lets UxVs drift around without enemy, objective, nor food.
'''

import random
import pygame
import numpy as np

STARTING_BLUE_BLOBS = 10
STARTING_RED_BLOBS = 10

WIDTH = 800
HEIGHT = 600
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)


# Blob parent class
class Blob:
    
    # Blob is born (when a Blob is created, this is going to run)
    def __init__(self, color, x_boundary, y_boundary, size_range=(4,8), movement_range=(-1,1)):
        # size_range & movement_range can be specified or go to those defaults
        self.size = random.randint(size_range[0],size_range[1])
        self.color = color # assigns color attribute to self object
        self.x_boundary = x_boundary # from window WIDTH
        self.y_boundary = y_boundary # from window HEIGHT
        self.x = random.randrange(0, self.x_boundary) # like randint(0,WIDTH-1)
        self.y = random.randrange(0, self.y_boundary)
        '''
        if color == BLUE: # assign locations based on color
            self.x = random.randrange(0, self.x_boundary / 4)
            self.y = random.randrange(0, self.y_boundary)
        elif color == RED:
            self.x = random.randrange(self.x_boundary*3/4, self.x_boundary)
            self.y = random.randrange(0, self.y_boundary)
        '''
        self.movement_range = movement_range

    def __str__(self): # return position
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other): # subtract one blob from another (get (x,y) distance)
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other): # check if two blobs over each other
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        '''
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)
        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)
        elif choice == 8:
            self.move(x=0, y=0)

    def move(self, move_x=False, move_y=False):
        # If no value for x, move randomly
        if not move_x:
            self.x += random.randint(self.movement_range[0],self.movement_range[1])
        else:
            self.x += move_x

        # If no value for y, move randomly
        if not move_y:
            self.y += random.randint(self.movement_range[0],self.movement_range[1])
        else:
            self.y += move_y

        # If we are out of bounds, fix!
        if self.x < 0: self.x = 0
        elif self.x > self.x_boundary - 1: self.x = self.x_boundary - 1
        if self.y < 0: self.y = 0
        elif self.y > self.y_boundary - 1: self.y = self.y_boundary - 1

    def move_drift(self, drift_dir, speed=1):
        # specify a random drift direction
        # drift_dir in degrees 0-360
        move_x =  int(round(np.sin(drift_dir * np.pi / 180))) + random.randint(-1,1) * speed
        move_y = -int(round(np.cos(drift_dir * np.pi / 180))) + random.randint(-1,1) * speed
        self.move(move_x, move_y)

    # End of Blob class


# BlueBlob class inherits from parent Blob class
class BlueBlob(Blob):

    def __init__(self, color, x_boundary, y_boundary, size_range=(4,8), movement_range=(-1,1)):
        # Overwrites __init__, but keep default values here so they can be specified when super init is called
        super().__init__(color, x_boundary, y_boundary, size_range, movement_range)
        self.color = BLUE
        # Overwrites x,y coord's specified by super init called above
        self.x = random.randrange(0, self.x_boundary / 4)
        self.y = random.randrange(0, self.y_boundary)

    def move_drift(self, drift_dir, speed=1):
        # drift randomly in a direction
        # drift_dir in degrees 0-360
        super().move_drift(drift_dir, speed)


# RedBlob class inherits from parent Blob class
class RedBlob(Blob):

    def __init__(self, color, x_boundary, y_boundary, size_range=(4,8), movement_range=(-1,1)):
        # Overwrites __init__, but keep default values here so they can be specified when super init is called
        super().__init__(color, x_boundary, y_boundary, size_range, movement_range)
        self.color = RED
        # Overwrites x,y coord's specified by super init called above
        self.x = random.randrange(self.x_boundary * 3 / 4, self.x_boundary)
        self.y = random.randrange(0, self.y_boundary)

    def move_drift(self, drift_dir, speed=1):
        # drift randomly in a direction
        # drift_dir in degrees 0-360
        super().move_drift(drift_dir, speed)


# define function to get new coord's from old coords based on direction (degrees) and distance
# rounded to nearest pixel values
def xy2xy(old_xy, direction, distance):
    # old_xy must be passed as an (x,y) coordinate tuple

    old_x, old_y = old_xy[0], old_xy[1]
    move_x =  int(round(np.sin(direction * np.pi / 180)*distance))
    move_y = -int(round(np.cos(direction * np.pi / 180)*distance))
    # likewise, returns new_xy an (x,y) coordinate tuple
    new_xy = (old_x + move_x, old_y + move_y)

    return new_xy

# define function to fly in circles
def fly_circles(center, radius):
    # receives center coord's as tuple

    center_x, center_y = center[0], center[1]


game_display = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("UxV Blob World")
clock = pygame.time.Clock()

def draw_environment(blob_list, t):
    game_display.fill(WHITE) # clears out frame, so we can redraw new frame on top
    
    for blob_dict in blob_list:
    # there's a blob_dict for blue blobs and one for red blobs
    # blob_list consists of 2 things: the blue dict and the red dict
        for blob_id in blob_dict:
            blob = blob_dict[blob_id]
            pygame.draw.circle(game_display, blob.color, [blob.x, blob.y], blob.size)

            blob.move_drift(t,1)

    pygame.display.update()  # updates our screen; backend builds screen; update sends build to the screen
    
def main():
    blue_blobs = dict(enumerate([BlueBlob(BLUE,WIDTH,HEIGHT,(8,8)) for i in range(STARTING_BLUE_BLOBS)]))
    red_blobs = dict(enumerate([RedBlob(RED,WIDTH,HEIGHT,(8,8)) for i in range(STARTING_RED_BLOBS)]))
    t = 0 # use t for time construct
    while True:
        for event in pygame.event.get(): # grabs event from pygame's events
            if event.type == pygame.QUIT: # pygame QUIT event (like clicking "X" in corner)
                pygame.quit()
                quit()

        t += 2  # move the needle so that blobs go in circles
        draw_environment([blue_blobs,red_blobs], t)
        clock.tick(200) # 60fps cap
        #print(red_blob.x, red_blob.y)

if __name__ == '__main__':
    main()

