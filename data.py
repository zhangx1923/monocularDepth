import copy
import pandas as pd
import os
from tools import Conf

'''
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''

#give picture ID, return corresponding label
class GenerateData:
    def __init__(self, f_path = "test img"):
        conf = Conf()
        self.csv_data = conf.getContent("csv_data")
        self.label_path = f_path + "/"
        # #希望从txt文件中得到哪些值
        # self.label_list = ["type", "truncated", "occluded", "xmin", "ymin","xmax","ymax", \
        #     "angle", "height","width","length","x","y","z","rotation_y"]

        #将txt文件中的英文type转化为int
        self.__type_to_int_kitti = {'Car':0, 'Van':1, 'Truck':2,
                     'Pedestrian':3, 'Person_sitting':4, 'Cyclist':5, 'Tram':6,
                     'Misc':7}

        ############标签应采用coco的meaning，目前只检测和kitti和coco重合的几类
    # car (coco car)  cocotype 3 kitti 0
    # Pedestrian (coco person) coco 1 kitti 3
    # Cyclist (coco bicyle) coco 2 kitti 5
    # truck (coco truck)  coco 8 kitti 2
        #self.__type_to_int_coco  = {'Car':3, 'Pedestrian':1, 'Cyclist': 2, 'Truck': 8}
        self.__type_to_int_coco  = {'Car':3, 'Pedestrian':1, 'Cyclist': 2}

    #parse txt file, and generate csv file from these txt files
    def generateCSV(self):
        df = pd.DataFrame(columns=['filename', 'class', 'truncated', 'occluded', 'observation angle', \
                           'xmin', 'ymin', 'xmax', 'ymax', 'height', 'width', 'length', \
                           'rot_y','xloc', 'yloc', 'zloc'])
        #csv文件中的行号
        count = 0
        for file_path_ls in os.listdir(self.label_path):
            with open(os.path.join(self.label_path,file_path_ls), 'r') as f: 
                data_content = list(l.split("\n")[0].split(" ") for l in f.readlines() if "DontCare" not in l)
            #generate csv file
            for dc in data_content:
                #过滤掉coco没有的标签
                if dc[0] not in self.__type_to_int_coco.keys():
                    continue

                df.at[count, 'filename'] = file_path_ls.split(".")[0] + ".png"

                df.at[count, 'class'] = self.__type_to_int_coco[dc[0]]
                df.at[count, 'truncated'] = dc[1]
                df.at[count, 'occluded'] = dc[2]
                df.at[count, 'observation angle'] = dc[3]

                # bbox coordinates
                df.at[count, 'xmin'] = dc[4]
                df.at[count, 'ymin'] = dc[5]
                df.at[count, 'xmax'] = dc[6]
                df.at[count, 'ymax'] = dc[7]

                # 3D object dimensions
                df.at[count, 'height'] = dc[8]
                df.at[count, 'width'] = dc[9]
                df.at[count, 'length'] = dc[10]

                # 3D object location
                df.at[count, 'xloc'] = dc[11]
                df.at[count, 'yloc'] = dc[12]
                df.at[count, 'zloc'] = dc[13]

                # rotation around y-axis in camera coordinates
                df.at[count, 'rot_y'] = dc[14]
                count += 1
        df.to_csv(self.csv_data, index=False)

    def get_type_int(self):
        return self.__type_to_int_coco


# if __name__ == "__main__":
#     gs = GetLabel()
#     gs.generateCSV([0,1,2,3,4])

        
        