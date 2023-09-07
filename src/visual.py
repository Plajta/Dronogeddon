import cv2
import numpy as np
import data

drone_last_pos = (0, 0)
map = np.full((1000, 1000, 3), 255)

def map_init():
    data.map_data

    #draw drone position (in center)
    pass


def update_map(drone_commands, curr_angle, dist):
    pass



if __name__ == "__main__":
    map_init()