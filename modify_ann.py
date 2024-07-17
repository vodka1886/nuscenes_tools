
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

import numpy as np

from argparse import ArgumentParser

from pyquaternion import Quaternion  
import numpy as np  

from nuscenes_tools.nuscenes_math import *
from nuscenes_tools.nuscenes_type import *
from nuscenes_tools.nuscenes_explorer import NuscenesExplorer
from nuscenes_tools.nuscenes_visualizer import *


click_position_g = None


def quaternion_ansys(w,x,y,z):
    # 给定的四元数  
    # w, x, y, z = 0.940133570865004, 0.0, 0.0, -0.3408062043634423  
    q = Quaternion(w=w, x=x, y=y, z=z)  
    
    # 转换为欧拉角 (pitch, yaw, roll)  
    yaw,pitch,roll = q.yaw_pitch_roll  # 注意：这里使用的是'zyx'顺序，你可以根据需要选择'xyz'或其他   
    print(f"Pitch: {pitch} degrees, Yaw: {yaw} degrees, Roll: {roll} degrees")  
    q2 = eula_to_quaternion(* q.yaw_pitch_roll)
    print(q2.elements)
    q3 = eula_to_quaternion(* q2.yaw_pitch_roll)
    print(q3.elements)
    # 计算旋转矩阵  
    rotation_matrix = q.rotation_matrix  
    print("Rotation Matrix:")  
    print(rotation_matrix)  
    
    # 计算逆矩阵（旋转矩阵的逆就是其转置，因为旋转矩阵是正交矩阵）  
    inverse_rotation_matrix = np.linalg.inv(rotation_matrix)  
    print("Inverse Rotation Matrix:")  
    print(inverse_rotation_matrix)  
    
    # 或者直接使用转置作为逆（对于旋转矩阵）  
    transpose_rotation_matrix = rotation_matrix.T  
    print("Transpose (Inverse) Rotation Matrix:")  
    print(transpose_rotation_matrix)  


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('root_path', help='Root path')
    parser.add_argument(
        '--version', default='v1.0-mini', help='version of datasets')
    parser.add_argument(
        '--cam-type',
        type=str,
        default='CAM_FRONT',
        help='choose camera type to inference')
   
    call_args = vars(parser.parse_args())

    # call_args['inputs'] = dict(
    #     points=call_args.pop('pcd'),
    #     img=call_args.pop('img'),
    #     infos=call_args.pop('infos'))
    # call_args.pop('cam_type')

    # if call_args['no_save_vis'] and call_args['no_save_pred']:
    #     call_args['out_dir'] = ''


    return call_args

if __name__=='__main__':
    call_args = parse_args()
    sample_token = '3e8750f331d7499e9b5123e9eb70f2e2'
    ann_token = 'b79ed739e23c47acb269731412eeeac1'
    nusc_ep = NuscenesExplorer(version=call_args['version'], root_path=call_args['root_path'])
    q = nusc_ep.get_ann_qu(ann_token)
    loc = nusc_ep.get_ann_loc(ann_token)
    while True:  
        user_input = input("请输入 'a' 或 'd'（或输入 'q' 退出）: ").lower()  
        add_ele_type = None
        if user_input == 'w':  
            print("w")
            loc = adjust_ann_loc(loc,q,0.1,0)
            nusc_ep.set_ann_loc(ann_token,loc)
            print(loc) 
        elif user_input == 's':  
            print("s")
            loc = adjust_ann_loc(loc,q,-0.1,0)
            nusc_ep.set_ann_loc(ann_token,loc)
            print(loc) 
        elif user_input == 'a':  
            print("a")
            loc = adjust_ann_loc(loc,q,0,0.1)
            nusc_ep.set_ann_loc(ann_token,loc)
            print(loc) 
        elif user_input == 'd':  
            print("d")
            loc = adjust_ann_loc(loc,q,0,-0.1)
            nusc_ep.set_ann_loc(ann_token,loc)
            print(loc) 
        elif user_input == 'q':  
            print("q")
            q = adjust_ann_yaw(q,10)
            nusc_ep.set_ann_qu(ann_token,q)
            print(q.elements) 
        elif user_input == 'e':  
            print("e")
            q = adjust_ann_yaw(q,-10)
            nusc_ep.set_ann_qu(ann_token,q)
            print(q.elements) 
        elif user_input == 'b':  
            print("b")
            vis_nuscenes_sample(nusc_ep,sample_token,'bev')
        elif user_input == 'p':  
            print("p")
            vis_nuscenes_sample(nusc_ep,sample_token,'img')
        elif user_input == 'm':  
            print("m")
            vis_nuscenes_sample(nusc_ep,sample_token,'man')
        elif user_input == 't':  
            print("t: add target")
            add_ele_type = input("请输入 'car' 或 'person': ").lower()  
            # add info to 
            if add_ele_type in size_map_g:
                click_position_use = get_pixel_from_image(nusc_ep,sample_token)
                print(click_position_use)
                global_pt,global_qu = image_pt_to_global_pt(nusc_ep,sample_token,click_position_use)
                print(global_pt,global_qu.elements)
                added_target = nusc_ep.generate_target(global_pt,global_qu,add_ele_type)
                nusc_ep.add_ann(sample_token,added_target)
                ann_token = added_target.token
                add_ele = None
        elif user_input == 'y':  
            print("y: add person")
            add_ele_type = "person"
        elif user_input == 'z':  
            print("退出程序。")  
            break  
        else:  
            print("无效的输入，请输入 'w:forward'、's:backword' 、'a:left' 、'd:right' 、'q:turn left' 、'e:turn right'  、'b:show bev'  、'p:play image' 或 'z:quit'。")
        
    
    print("end")