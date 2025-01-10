import pickle as pk
from numpy import array_equal, random
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.pyplot import figure
import numpy as np
from copy import copy, deepcopy
import os
import re
from PIL import Image, ImageOps, ImageDraw, ImageFont
from urllib.request import urlopen

class Results:
  
    def __init__(self, map_name, exp_number, baseline_included = True):
        self._experiment_number = exp_number
        self._data_abs = []
        self._data_baseline = []

        root = os.getcwd()
        path_abs = root+"/transfer_learning/results/"+map_name+"_"
        path_base = root+"/transfer_learning/results/base_"+map_name+"_"
        # path_abs = root + '\\' + 'Abstraction' + '\\' + 'results' + '\\' + map_name + '_'
        #path_base = root + '\\' + 'Abstraction' + '\\' + 'results' + '\\' + 'base_' + map_name + "_"

        for i in range (1,self._experiment_number + 1):

            with open(path_abs + str(i), "rb") as file_name:
                self._data_abs.append( pk.load(file_name) )
            if baseline_included:
                with open(path_base + str(i), "rb") as file_name:
                    self._data_baseline.append (pk.load(file_name))
        
        self._color_list = []
        self._color_bounds = [-0.5]
        self._color_assignment = {}
        self._color_map = None
        self._color_track = 0
        self._sub_a = 0
        self._sub_b = 0
        self._abs_plot_dim = None
        self._counter = 0
        self._color_maps = []
        self._phase_success = []
        self._maze = self._data_abs[0]
        self._baseline_included = baseline_included
        self.data_extraction()
        self._episode_number = len(self._rewards_abs[0])

    def adjust_rate (self, data):
        epi_counter = 0
        success_counter = 0
        data_adjusted = []
        for i in range (len(data)):
            epi_counter += 1
            if data[i] == 1: success_counter +=1
            rate = success_counter / epi_counter
            data_adjusted.append(rate)
        return data_adjusted

    def data_extraction (self):
        self._rewards_abs = []
        self._rewards_base = []
        self._success_abs = []
        self._success_base = []
        self._abstraction_result = []
        self._qtables = []
        self._map = self._data_abs[0][0]

        for i in range ( self._experiment_number):
            temp_abs = []
            self._abstraction_result.append(self._data_abs[i][1][-1])
            self._qtables.append (self._data_abs[i][3])
            for item in self._data_abs[i][2][1]:
                temp_abs = temp_abs + item
            self._rewards_abs.append(temp_abs)

            temp_abs = []
            for item in self._data_abs[i][2][2]:
                temp_abs = temp_abs + item
            self._success_abs.append(self.adjust_rate(temp_abs))
            if self._baseline_included:
                self._rewards_base.append (self._data_baseline[i][2][1][0])
                self._success_base.append (self.adjust_rate(self._data_baseline[i][2][2][0]))
            if self._baseline_included: self._episodes = self._data_baseline[i][2][0][0]
   
        return None

    def prepare_avg_bound (self, param):
        abs = np.zeros ((3,self._episode_number))
        base = np.zeros ((3,self._episode_number))
        
        for i in range (self._episode_number):
            epi = self._episodes
            temp_abs = []
            temp_base = []
            data_dict_abs = {"rewards": self._rewards_abs, "success rate": self._success_abs}
            data_dict_base = {"rewards": self._rewards_base, "success rate": self._success_base}
  
            for j in range (self._experiment_number):
                
                temp_abs.append( data_dict_abs[param][j][i] )
                temp_base.append( data_dict_base[param][j][i] )

            temp_abs = np.array(temp_abs)
            temp_base = np.array(temp_base)
            abs_avg = np.average(temp_abs)
            base_avg = np.average(temp_base)
            abs_std = np.std(temp_abs)
            base_std = np.std(temp_base)
            abs[0][i] = abs_avg + abs_std
            abs[1][i] = abs_avg
            abs[2][i] = abs_avg - abs_std

            base[0][i] = base_avg + base_std
            base[1][i] = base_avg
            base[2][i] = base_avg - base_std
        return abs, base, epi

    def compare_bound (self, param, moving_number):
        abs, base, epi = self.prepare_avg_bound (param)
        abs_smooth, base_smooth, epi_smooth = [], [], []
        if moving_number != None:
            for i in range (3):
                temp_base, epi_smooth = self.moving_average(moving_number, base[i,::], epi)
                base_smooth.append(temp_base)
                temp_abs, aaa = self.moving_average(moving_number, abs[i,::], epi)
                abs_smooth.append(temp_abs)

        line_w = 1
        plt.plot(epi_smooth, abs_smooth[1], color='#009dff', linestyle='solid',
            linewidth=line_w,label='AD Q-learning')
        plt.fill_between(epi_smooth, abs_smooth[0], abs_smooth[2],
            alpha=0.5, edgecolor='#CC4F1B', facecolor='#b3e2ff',linewidth=0)
        
        plt.plot(epi_smooth, base_smooth[1], color='#ff6600', linestyle='solid',
            linewidth=line_w, label='Q-Learning')
        plt.fill_between(epi_smooth, base_smooth[0], base_smooth[2],
            alpha=0.5, edgecolor='#CC4F1B', facecolor='#ffd0b0ff',linewidth=0)
        plt.xlabel("episodes")
        plt.ylabel(param)
        plt.legend()
        plt.show()

    def moving_average (self, moving_number,y, x):
        y_m = []
        x_m = []
        if moving_number != np.inf:
            for i in range (moving_number, len(y)):
                sum_temp = 0
                for j in range (i - moving_number, i):
                    sum_temp += y[j]
                sum_temp /= moving_number
                y_m.append(sum_temp)
                x_m.append(i)
        else:
            for i in range (len(y)):
                sum_temp = 0
                for j in range (0, i):
                    sum_temp += y[j]
                sum_temp /= i + 1
                y_m.append(sum_temp)
                x_m.append(i)
        return y_m, x_m

    # Add your code here #
    def make_symbols(image_arr, map_arr, scale, thickness):
        for x in range(map_arr.shape[0]):
            for y in range(map_arr.shape[1]):
                if map_arr[x,y] == -1: 
                    image_arr[x*scale : x*scale + scale, y*scale + int(scale/2) -int(thickness/2):y*scale + int(scale/2) +int(thickness/2)] = np.array((255,0,0,255))
                    image_arr[x*scale + int(scale/2) -int(thickness/2): x*scale + int(scale/2) +int(thickness/2), y*scale : y*scale + scale] = np.array((255,0,0,255))
                elif map_arr[x,y] == 1:
                    image_arr[x*scale : x*scale + scale, y* scale : y*scale + scale] = np.array((0,0,0,255))
        return image_arr
    
    def mark_destination(image_arr, map_arr, scale, thickness, init, goal, p_locs, label):
        image_arr = np.copy(image_arr)
        for x in range(map_arr.shape[0]):
            for y in range(map_arr.shape[1]):
                if map_arr[x,y] == 1:
                    image_arr[x*scale : x*scale + scale, y* scale : y*scale + scale] = np.array((0,0,0,255))
                if label and tuple([x,y]) == tuple(goal): 
                    image_arr[x*scale : x*scale + scale, y*scale + int(scale/2) -int(thickness):y*scale + int(scale/2) +int(thickness)] = np.array((240,110,24,255))
                    image_arr[x*scale + int(scale/2) -int(thickness): x*scale + int(scale/2) +int(thickness), y*scale : y*scale + scale] = np.array((240,110,24,255))
                if label and tuple([x,y]) == tuple(init): 
                    image_arr[x*scale : x*scale + scale, y*scale + int(scale/2) -int(thickness):y*scale + int(scale/2) +int(thickness)] = np.array((235,227,12,255))
                    image_arr[x*scale + int(scale/2) -int(thickness): x*scale + int(scale/2) +int(thickness), y*scale : y*scale + scale] = np.array((235,227,12,255))
                if label and tuple([x,y]) in [tuple(loc) for loc in p_locs]: 
                    image_arr[x*scale : x*scale + scale, y*scale + int(scale/2) -int(thickness/2):y*scale + int(scale/2) +int(thickness/2)] = np.array((234,63,247,255))
                    image_arr[x*scale + int(scale/2) -int(thickness/2): x*scale + int(scale/2) +int(thickness/2), y*scale : y*scale + scale] = np.array((234,63,247,255))     
        return image_arr

    def get_heatmap_color(q, maxm, minm, avg):
        if maxm == 0 and minm == 0 and avg == 0:
            return np.array((255,255,255))
        else:
            if q >= avg:
                if maxm == avg:
                    return np.array((0,0,255))
                else:
                    ratio = (q - avg) / (maxm - avg)
                    # print(q, ratio)
                    b = int(255 * ratio)
                    r = 0
                    g = int(255 * (1 - ratio))
            else:
                if minm == avg:
                    return np.array((0,255,0))
                else:
                    ratio = (q - minm) / (avg - minm)
                    # print(q, ratio)
                    b = int(255 * ratio)
                    r = 255
                    g = int(255 * ratio)

            # if values are less than minm value, then same color as minm
            if r < 0:
                r = 0
            if g < 0:
                g = 0
            if b < 0:
                b = 0

            # if values are greater than maxm value, then same color as maxm
            if r > 255:
                r = 255
            if g > 255:
                g = 255
            if b > 255:
                b = 255
            return np.array((r,g,b))
    
    def get_max_min(q_table):
        if len(q_table) == 0:
            qmax, qmin, qavg = 0.0, 0.0, 0.0
        else:
            states = list(q_table.keys())
            qmax, qmin = -np.inf, np.inf
            # qmax, qmin = 0.0, 0.0
            avg = 0
            qavg = 0
            for s in states:
                # max_temp = np.max(q_table[s])
                # min_temp = np.min(q_table[s])
                max_temp = np.max(list(q_table[s].values()))
                min_temp = np.min(list(q_table[s].values()))
                if max_temp > qmax: qmax = max_temp
                if min_temp < qmin: qmin = min_temp
                for v in q_table[s]: avg += v
            if len(states) > 0:
                qavg = avg/(len(states)*len(q_table[states[-1]]))
        return qmax, qmin, qavg
    
    def get_action_loc(temp):
        temp = temp.replace("(","")
        temp = temp.replace(")","")
        temp = temp.split(", ")
        temp[0] = temp[0].replace('"','')
        temp[0] = temp[0].replace("'",'')
        temp[1] = temp[1].replace('"','')
        temp[1] = temp[1].replace("'",'')
        xi = int(temp[0].split(",")[0])
        xj = int(temp[1].split(",")[0])
        return xi, xj
    
    def get_color_for_abstract_state(abstract_state, q_table, qmax, qmin, qavg, abstraction_colors):
        if abstract_state not in q_table:
            if abstract_state in abstraction_colors:
                color = abstraction_colors[abstract_state]
            else:
                max_q_val = min(0.0,qmin)
                color = Results.get_heatmap_color(max_q_val, qmax, qmin, qavg)
        else:
            # max_q_val = np.array(q_table[abstract_state]).max()
            max_q_val = 0.0
            if len(q_table[abstract_state]) > 0:
                max_q_val = np.array(list(q_table[abstract_state].values())).max()
            color = Results.get_heatmap_color(max_q_val, qmax, qmin, qavg)
            abstraction_colors[abstract_state] = color
        return color, abstraction_colors
    
    def get_best_action(abstract_state, q_table, best_actions):
        if q_table is None:
            best_action = -1
        else:
            if abstract_state not in q_table:
                if abstract_state in best_actions:
                    best_action = best_actions[abstract_state]
                else:
                    best_action = -1
            else:
                best_action = np.array(q_table[abstract_state]).argmax()
                best_actions[abstract_state] = best_action
        return best_action, best_actions

    def get_abstraction_heatmap(abstraction_array, q_table, qmax, qmin, qavg, abstraction_colors, best_actions, label=True):
        image_array = np.full((abstraction_array.shape[0],abstraction_array.shape[1],3), 255)
        # colored_abstactions = {}
        # qmax, qmin, qavg = Results.get_max_min (q_table)
        loc_to_action = dict()

        for x in range(abstraction_array.shape[0]):
            for y in range(abstraction_array.shape[1]):
                if abstraction_array[x,y] not in q_table: 
                    best_action = -1
                else:
                    # best_action = q_table[abstraction_array[x,y]].argmax()
                    best_action = max(q_table[abstraction_array[x,y]], key=lambda x: q_table[abstraction_array[x,y]][x])

                image_array[x,y], abstraction_colors = Results.get_color_for_abstract_state(abstraction_array[x,y], q_table, qmax, qmin, qavg, abstraction_colors)

                if label:
                    best_actions[abstraction_array[x,y]] = best_action

                if abstraction_array[x, y]!='':
                    xi, xj = Results.get_action_loc(abstraction_array[x, y])
                    if label:
                        loc_to_action[(xi,xj)] = best_actions[abstraction_array[x, y]]
        im = Image.fromarray(image_array.astype(np.uint8))
        return im, abstraction_colors, best_actions, loc_to_action

    def add_heatmap_bar(image, q_table, file_name, loc_to_action, scale):
        basepath = os.getcwd()+"/src/"
        font = ImageFont.load_default() 

        num_neg = 0
        num_pos = 0
        num_0 = 0

        # qmax, qmin, avg = Results.get_max_min(q_table)
        qmax, qmin, avg = 255, 0.0, 127.5
        total = num_0 + num_neg + num_pos
        percent_neg = 0.25
        percent_pos = 0.75
        percent_0 = 0.0
        im_arr = np.asarray(image)
        width = im_arr.shape[0]
        height = im_arr.shape[1]
        new_im = np.full((width+60, height+200, 3), 255)

        # heatmap gradient
        for x in range(width):
            for y in range(height, height+50):
                if y >= height and y < height+11:
                    new_im[x+30,y] = (0,0,0)
                elif x < width * percent_neg:#fill with negative gradient
                    num_pix = width - (width-width*percent_neg)
                    new_im[x+30,y] = (255, x*(255/int(width*percent_neg)), x*(255/int(width*percent_neg)))
                    # new_im[x+30,y] = (255, 0+x*(255/int(width*percent_neg)), 0+x*(255/int(width*percent_neg)))
                    # new_im[x+30,y] = (0+x*(255/int(width*percent_neg)), 255, 0+x*(255/int(width*percent_neg)))
                elif x > width * (percent_0+percent_neg):
                    num_pix = width - (width-width*percent_pos)
                    new_im[x+30,y] = (0, 255 - ((x-(width*(1-percent_pos)))* (255/num_pix)), ((x-(width*(1-percent_pos)))* (255/num_pix)))
                    # new_im[x+30,y] = (255 - ((x-(width*(1-percent_pos)))* (255/num_pix)), 255 - ((x-(width*(1-percent_pos)))* (255/num_pix)), 255)
                    #new_im[x,y] = (255- (x-width*(percent_pos))*(255/(width*percent_pos)),255 - (x-(width*percent_pos))*(255/(percent_pos)), 255)

        # black borders
        new_im[30:width+30,height+50: height+53] = (0,0,0)
        new_im[20:30, 0:height+53] = (0,0,0)
        new_im[width+30:width+40, 0:height+53] = (0,0,0)

        new_im[30:33,height+11:height+50] = (255,0,0)#red bar for minimum
        new_im[width+30-3 : width+30, height+11 : height+50] = (255,0,0) #red bar for max
        min_dist = 10000
        for x in range(30,width+30):
            if  min_dist > int(255-new_im[x, height+25, 0]+ 255-new_im[x, height+25, 1] + 255-new_im[x, height+25, 2]) and min_dist >= 0:
                mid_x = x
                min_dist = int(255-new_im[x, height+25, 0]+ 255-new_im[x, height+25, 1] + 255-new_im[x, height+25, 2])
        new_im[mid_x-1:mid_x+1, height+11:height+50] = (255,0,0)
        im = Image.fromarray(new_im.astype(np.uint8))
        im = ImageOps.flip(im)
        draw = ImageDraw.Draw(im)
        draw.text((height+55 ,30-15), str(int(qmax)), (255,0,0), font = font)#max
        # draw.text((height+55 , width-mid_x+15+30), '0', (255,0,0), font = font)#middle
        draw.text((height+55 , width-mid_x+15+30), str(int(avg)), (255,0,0), font = font)#middle
        draw.text((height+55 , 30+width-15), str(int(qmin)), (255,0,0), font = font)#min

        action_to_name = {-1:"-",0:"N",1:"S",2:"W",3:"E",4:"p",5:"d"}
        draw = ImageDraw.Draw(image)
        for (x,y),action in loc_to_action.items():
            action_name = action_to_name[action]
            draw.text((y*scale, x*scale), str(action_name), (240,110,24,255), font = font)

        im.paste(image, (0,30), 0)
        # im.show('Results.png')
        im.save(file_name)
            
    def get_qtable_heatmap(map_arr, abstraction_array, scale, q_table, qmax, qmin, qavg, file_name, init, goal, p_locs, abstraction_colors, best_actions, label=True):
        abstraction_image, abstraction_colors, best_actions, loc_to_action = Results.get_abstraction_heatmap(abstraction_array, q_table, qmax, qmin, qavg, abstraction_colors, best_actions, label)
        image = Image.new("RGBA", (abstraction_array.shape[0]*scale, abstraction_array.shape[1]*scale), (255,255,255,0))
        image_arr = np.asarray(image)
        symbols = Image.fromarray(Results.mark_destination(image_arr, map_arr, scale, 7, init, goal, p_locs, label).astype(np.uint8))
        overlay = Results.get_faded_overlay(Results.clear_resize(abstraction_image, scale), symbols, 255)
        Results.add_heatmap_bar(overlay, q_table, file_name, loc_to_action, scale)
        return abstraction_colors, best_actions

    def get_option_heatmap(map_arr, abstraction_array, scale, q_table, file_name, init, goal, p_locs, label=True):
        image_array = np.full((abstraction_array.shape[0],abstraction_array.shape[1],3), 255)
        for x in range(abstraction_array.shape[0]):
            for y in range(abstraction_array.shape[1]):
                if abstraction_array[x,y] not in q_table: 
                    image_array[x,y] = np.array((255, 255, 255))
                else:
                    image_array[x,y] = q_table[abstraction_array[x,y]]
        abstraction_image = Image.fromarray(image_array.astype(np.uint8))

        image = Image.new("RGBA", (abstraction_array.shape[0]*scale, abstraction_array.shape[1]*scale), (255,255,255,0))
        image_arr = np.asarray(image)
        symbols = Image.fromarray(Results.mark_destination(image_arr, map_arr, scale, 7, init, goal, p_locs, label).astype(np.uint8))
        overlay = Results.get_faded_overlay(Results.clear_resize(abstraction_image, scale), symbols, 255)
        Results.add_heatmap_bar(overlay, q_table, file_name, {}, scale)

    def clear_resize(image, scale):
        old_image = np.asarray(image.convert("RGBA"))
        image_array = np.full((old_image.shape[0] * scale,old_image.shape[1] * scale, 4), 255)
        for x in range(old_image.shape[0]):
            for y in range(old_image.shape[1]):
                color = old_image[x, y]
                image_array[x*scale : x*scale + scale, y* scale : y*scale + scale] = color
        return Image.fromarray(image_array.astype(np.uint8))

    def get_random_color():
        import random
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        return np.array((r,g,b))

    def get_faded_overlay(background_im, foreground_im, opacity):
        white_back = Image.new("RGBA" , background_im.size, "WHITE")
        new_back = np.asanyarray(background_im.convert("RGBA"))
        new_back = new_back.copy()
        new_back[:, :, 3] = opacity
        new_back_image = Image.fromarray(new_back.astype(np.uint8))
        transparent = foreground_im
        new_back_image.paste(transparent, (0,0), transparent)
        white_back.paste(new_back_image, (0,0), new_back_image)
        return white_back