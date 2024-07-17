from argparse import ArgumentParser
import os

from nuscenes.nuscenes import NuScenes

from nuscenes_tools.nuscenes_math import *
from nuscenes_tools.nuscenes_type import *
from nuscenes_tools.nuscenes_explorer import *
from nuscenes_tools.nuscenes_visualizer import *

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('root_path', help='Root path')
    parser.add_argument(
        '--version', default='v1.0-mini', help='version of datasets')
    parser.add_argument(
        '--dst_version',
        type=str,
        default='v1.0-demo',
        help='dst version')
   
    call_args = vars(parser.parse_args())
    return call_args


if __name__=='__main__':
    call_args = parse_args()
    
    nusc_ep = NuscenesExplorer(version=call_args['version'], root_path=call_args['root_path'])
    nusc_dst = NuscenesExplorer(version=call_args['dst_version'], root_path=call_args['root_path'])
    # initial tokens
    scene_token = nusc_ep.list_scenes()[0]
    sample_token = nusc_ep.list_samples(scene_token)[0]
    ann_token = nusc_ep.list_anns(sample_token)[0]
    mode_map = ['img','bev','cv']
    mode_idx = 0
    
    
    
    while True:  
        help_info = "请输入 's:select scene and sample' 'a:ann modify' 'p:play' 'c:copy'（或输入 'q' 退出）: "
        print("Current: Scene: {} Sample: {} Ann: {}".format(scene_token,sample_token,ann_token))
        user_input = input(help_info).lower()  
        if user_input == 's':  
            print("s: select scene and sample")
            scene_token,sample_token = vis_nuscenes(nusc_ep)
        elif user_input == 'a':  
            print("a: select ann and edit")
            ann_token = vis_ann_on_image(nusc_ep,sample_token)
        
        elif user_input == 'p':  
            print("p: play")
            vis_nuscenes_sample(nusc_ep,sample_token,mode_map[mode_idx])
            # vis_nuscenes_scene(nusc_ep,scene_token,'man')
        elif user_input == 'm':  
            print("m: change mode")
            mode_idx = (mode_idx + 1)%len(mode_map)
        elif user_input == 'c':  
            print("c: copy")
            if not nusc_dst.nusc:
                nusc_dst.create_empty_dataset(nusc_ep.nusc)
                nusc_dst.load()
            nusc_dst.copy_sample_form_dataset(nusc_ep,sample_token)
        elif user_input == 'q':  
            print("退出程序。")  
            break  
        else:  
            print("无效的输入: {} ! {}".format(user_input,help_info))
        
    print("end")