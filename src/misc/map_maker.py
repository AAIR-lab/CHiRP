import numpy as np
import pickle as pk
from PIL import Image
import os


def get_map(file_name):
    print(os.getcwd())
    image_path = "environments/maps/" + file_name + ".png"
    color_array = {}
    count = 2 
    image = Image.open(image_path)
    image = image.convert("RGB")
    old_map = np.asarray(image)
    map = np.zeros((old_map.shape[0], old_map.shape[1]))
    for x in range(old_map.shape[0]):
        for y in range(old_map.shape[1]):
            if np.array_equal(old_map[x, y], np.array((0,0,0))):
                map[x,y] = 1
            #adds any other color than black or white as pitfall, can add additional colors here for different enviroments
            elif np.array_equal(old_map[x,y], np.array((255,0,0))):
                map[x,y] = -1
            elif not np.array_equal(old_map[x, y], np.array((255,255,255))):
                map_bytes = old_map[x,y].tobytes()
                if map_bytes in color_array.keys():
                    map[x,y] = color_array[map_bytes]
                else:
                    map[x,y] = count
                    color_array[map_bytes] = count
                    count +=1
    return map

def get_map_rooms(file_name):
    basepath = os.getcwd()
    image_path = basepath+"/environments/maps/" + file_name + ".map"
    f = open(image_path)
    string = ""
    for line in f:
        string += line
    env_map = string_to_bool_map(string)
    return env_map

def string_to_bool_map(str_map):
    bool_map = []
    for row in str_map.split('\n')[:]:
        bool_map.append([int(not (r==' ')) for r in row])
    return np.array(bool_map)
