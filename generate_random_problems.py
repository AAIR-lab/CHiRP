import random
import os

import hyper_param
from src.misc import utils
import src.misc.map_maker as map_maker

def generate_random_problems(map_name):
    maze = map_maker.get_map(map_name)
    dimension = maze.shape

    string = 'problem_args = {\n'
    for i in range(40):
        random_row = random.randint(0, len(maze) - 1)
        random_col = random.randint(0, len(maze[0]) - 1)
        while(maze[random_row][random_col]!=0):
            random_row = random.randint(0, len(maze) - 1)
            random_col = random.randint(0, len(maze[0]) - 1)

        start = [random_row, random_col]

        if "rooms" in map_name:
            random_row = random.randint(0, len(maze) - 1)
            random_col = random.randint(0, len(maze[0]) - 1)
            while(maze[random_row][random_col]!=0):
                random_row = random.randint(0, len(maze) - 1)
                random_col = random.randint(0, len(maze[0]) - 1)
            end = [random_row, random_col]
            state = start + end

        elif "taxi" in map_name:
            locations = {}
            locations[0] = []
            locations[1] = [0,0]
            locations[2] = [0,dimension[1]-1]
            locations[3] = [dimension[0]-1,0]
            locations[4] = [dimension[0]-1,dimension[1]-1]
            locations[5] = [0,dimension[1]//2]
            locations[6] = [dimension[0]//2,dimension[1]-1]
            locations[7] = [dimension[0]-1,dimension[1]//2]
            locations[8] = [dimension[0]//2,0]

            pickup = [random.randint(1,4)]
            dropoff = [random.randrange(1,4)]
            while pickup[0] == dropoff[0]:
                dropoff = [random.randrange(1,4)]

            init_vars = start + locations[pickup[0]] 
            goal_vars = locations[dropoff[0]]
        string += "     "+ f'{i}' +': { "init_vars": '+f'{init_vars}' + ', "goal_vars": '+f'{goal_vars}' + '},\n'
    string += '}'
    return string

def plot_random_problems(problem_args):
    env = hyper_param.env
    for env_id in problem_args:
        env.initialize_problem(problem_args[env_id], hyper_param.step_max)
        env_directory = hyper_param.directory+"/env_"+str(env_id)
        if not os.path.exists(env_directory+"/heatmaps/"):
            os.makedirs(env_directory+"/heatmaps/")
        utils.plot_option_path([], None, env, env_id, env_directory)

if __name__=="__main__":
    # map_name = "rooms_33x33"
    map_name = "taxi_10x10"
    problem_args = generate_random_problems(map_name)
    print(problem_args)
    # plot_random_problems(problem_args)