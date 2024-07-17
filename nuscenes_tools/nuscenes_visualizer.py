from nuscenes.nuscenes import NuScenes
import cv2 
# from PIL import Image
# import numpy as np
# import open3d as o3d
import os
import matplotlib.pyplot as plt
import datetime
from nuscenes_tools.nuscenes_math import global_pt_to_image
from nuscenes_tools.nuscenes_explorer import NuscenesExplorer

# import keyboard

import threading
# show_g = False
# show_sample_token_g = ""
def vis_nuscenes(nusc_ep:NuscenesExplorer):
    scenes = nusc_ep.list_scenes()
    mode = "man"
    help_info = "请输入 'p: pre n: next 或 'q: quit' "
    scene_index = 0
    sample_index = 0
    samples = nusc_ep.list_samples(scenes[scene_index])
    while True: 
        vis_nuscenes_sample(nusc_ep,samples[sample_index],mode)
        user_input = input(help_info).lower()  
        if user_input == 'o': 
            print("o: pre scene")
            scene_index = (scene_index + 1) % len(scenes) 
            samples = nusc_ep.list_samples(scenes[scene_index])
            sample_index = 0
        elif user_input == 'p': 
            print("p: next scene")
            scene_index = (scene_index - 1 + len(scenes)) % len(scenes)  
            samples = nusc_ep.list_samples(scenes[scene_index])
            sample_index = 0
        elif user_input == 'k': 
            print("k: pre sample")
            sample_index = (sample_index + 1) % len(samples)  
        elif user_input == 'l': 
            print("l: next sample")
            sample_index = (sample_index - 1 + len(samples)) % len(samples)  
        elif user_input == 'q':  # 如果按下'q'，退出  
            cv2.destroyAllWindows()
            break  
        else:
            print("无效的输入 {} ! {}".format(user_input,help_info))
    return scenes[scene_index],samples[sample_index]


def vis_nuscenes_scene(nusc_ep:NuscenesExplorer,scene_token:str=None,mode:str="img"):
    
    samples = nusc_ep.list_samples(scene_token)
    current_index = 0
    help_info = "请输入 'p: pre n: next 或 'q: quit' "
    # 显示或保存图像  
    while True: 
        vis_nuscenes_sample(nusc_ep,samples[current_index],mode)
        user_input = input(help_info).lower()  
        if user_input == 'p': 
            print("p: pre")
            current_index = (current_index + 1) % len(samples) 
            # show_sample_token_g = samples[current_index]
        elif user_input == 'n': 
            print("n: next")
            current_index = (current_index - 1 + len(samples)) % len(samples)  
            # show_sample_token_g = samples[current_index] 
        elif user_input == 'q':  # 如果按下'q'，退出  
            # show_g = False
            break  
        else:
            print("无效的输入 {} ! {}".format(user_input,help_info))

    return samples[current_index]

def draw_ann_on_image(nusc_ep:NuscenesExplorer,sample_token:str,ann_token:str,img):
    sample = nusc_ep.nusc.get('sample', sample_token)
    ann = nusc_ep.nusc.get('sample_annotation', ann_token)
    glo_loc = ann['translation']
    glo_rot = ann['rotation']
    # 读取所有sample_ann
    cam_token = sample['data']['CAM_FRONT']
    cam_intrinsic,cam_translation,cam_rotation,ego_translation,ego_rotation = nusc_ep.get_cam_calis(cam_token)
    pix_loc = global_pt_to_image(glo_loc,glo_rot,ego_translation,ego_rotation,cam_translation,cam_rotation,cam_intrinsic)

    # print(pix_loc)
    test_pt = (int(pix_loc[0][0]),int(pix_loc[1][0]))
    cv2.circle(img, test_pt, 8, (255, 0, 0), 2)  
    return img

    
def vis_ann_on_image(nusc_ep:NuscenesExplorer,sample_token:str=None,mode:str="img"):
    anns = nusc_ep.list_anns(sample_token)
    mode = "man"
    help_info = "请输入 'p: pre n: next 或 'q: quit' "
    ann_index = 0
    unshown_list = []
    while True: 
        ori_frame,frame = vis_nuscenes_sample(nusc_ep,sample_token,mode,False,unshown_list)
        frame_focus = draw_ann_on_image(nusc_ep,sample_token,anns[ann_index],frame)
        cv2.imshow("test",frame_focus)
        cv2.waitKey(1)
        user_input = input(help_info).lower()  
        if user_input == 'o': 
            print("o: pre ann")
            ann_index = (ann_index + 1) % len(anns) 
        elif user_input == 'p': 
            print("p: next ann")
            ann_index = (ann_index - 1 + len(anns)) % len(anns)  
        elif user_input == 'u':  
           unshown_list.append(anns[ann_index]) 
        elif user_input == 'a':  
           unshown_list.remove(anns[ann_index]) 
        elif user_input == 's':   
            current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            ori_file = os.path.join("./outputs","{}_{}_ori.jpg".format(sample_token,current_time))
            result_file = os.path.join("./outputs","{}_{}.jpg".format(sample_token,current_time))
            cv2.imwrite(ori_file,ori_frame)
            cv2.imwrite(result_file,frame)
            break  
        elif user_input == 'q':  # 如果按下'q'，退出  
            cv2.destroyAllWindows()
            break  
        else:
            print("无效的输入 {} ! {}".format(user_input,help_info))
    
    
    
def vis_nuscenes_sample(nusc_ep:NuscenesExplorer,sample_token:str,mode:str,is_show=True,unshown_list:list=[]):
    # 初始化NuScenes对象
    nusc = nusc_ep.nusc

    # 选择一个场景并获取其token
    sample = nusc.get('sample', sample_token)
    
    if mode == 'img':
        sensor = 'CAM_FRONT'
        cam_front_data = nusc.get('sample_data', sample['data'][sensor])
        nusc.render_sample_data(cam_front_data['token'])
        return
    
    if mode == 'bev':
        sensor = 'LIDAR_TOP'
        lidar_top_data = nusc.get('sample_data', sample['data'][sensor])
        nusc.render_sample_data(lidar_top_data['token'])
        return
    
    # 读取图片
    # 获取CAM_FRONT的图像数据和内参  
    cam_front_token = sample['data']['CAM_FRONT']  
    cam_front_data = nusc.get('sample_data', cam_front_token)  
    # # 读取图像文件  
    image_path = cam_front_data['filename']  
    image_ori = cv2.imread(os.path.join(nusc.dataroot,image_path)) 
    image = image_ori.copy()
    # image = Image.open(os.path.join(nusc.dataroot,image_path)).convert('BGR')  
    # image_np = np.array(image)  # 转换为NumPy数组以便使用OpenCV  
    
    # 读取所有sample_ann
    cam_token = sample['data']['CAM_FRONT']
    cam_intrinsic,cam_translation,cam_rotation,ego_translation,ego_rotation = nusc_ep.get_cam_calis(cam_token)
    
    # anns = list(map(get_ann,sample['anns']))
    category_map = ["1fa93b757fc74fb197cdd60001ad8abf",
                        "fd69059b62a3469fbaef25340c0eab7f",
                        "dfd26f200ade4d24b540184e16050022",
                        "fc95c87b806f48f8a1faea2dcc2222a4",
                        "6021b5187b924d64be64a702e5570edf",
                        ]
    
    anns = nusc_ep.list_anns(sample_token)
    selected_anns = []
    for ann_token in anns:
        if ann_token in unshown_list:
            continue
        ann = nusc_ep.nusc.get('sample_annotation', ann_token)
        ins = nusc_ep.nusc.get('instance', ann['instance_token'])
        if ins['category_token'] in category_map:
            selected_anns.append(ann_token)
    # Get annotations and params from DB.
    data_path, boxes, camera_intrinsic = nusc_ep.nusc.get_sample_data(cam_front_token,
                                                                           box_vis_level=1,selected_anntokens=selected_anns)
    front_color = (0, 255 , 0)
    rear_color = front_color
    side_color = front_color
    
    for box in boxes:
        box.render_cv2(image, view=camera_intrinsic, normalize=True,colors=(front_color, rear_color, side_color))

    # for ann_token in sample['anns']:
    #     ann = nusc.get('sample_annotation', ann_token)
    #     glo_loc = ann['translation']
    #     glo_rot = ann['rotation']
        
    #     pix_loc = global_pt_to_image(glo_loc,glo_rot,ego_translation,ego_rotation,cam_translation,cam_rotation,cam_intrinsic)

    #     # print(pix_loc)
    #     test_pt = (int(pix_loc[0][0]),int(pix_loc[1][0]))
    #     cv2.circle(image, test_pt, 2, (0, 255, 0), 2)  
    if is_show:
        cv2.imshow("{}".format(sample_token),image)
        cv2.waitKey(1)
        print("Show over!")
    return image_ori,image

    
    
def get_pixel_from_image(nusc_ep:NuscenesExplorer,sample_token):
    global click_position_g
    click_position_g = None
    # 初始化 NuScenes 数据集
    nusc = nusc_ep.nusc
    # 获取一个样本
    sample = nusc.get('sample', sample_token)

    # 获取摄像头数据
    cam_token = sample['data']['CAM_FRONT']
    cam_data = nusc.get('sample_data', cam_token)
    # 读取图像
    image_path = nusc.get_sample_data_path(cam_token)
    image = plt.imread(image_path)

    # 定义鼠标点击事件的回调函数
    def onclick(event):
        global click_position_g
        if event.xdata is not None and event.ydata is not None:
            click_position_g = (int(event.xdata), int(event.ydata))
            print(f"点击位置的像素坐标: {click_position_g}")
            plt.close()  # 关闭图像

    # 显示图像并捕获鼠标点击事件
    fig, ax = plt.subplots()
    ax.imshow(image)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # 阻塞，直到点击事件发生
    plt.show()
    click_position_use = click_position_g
    click_position_g = None
    return click_position_use


 # sample = nusc.get('sample', sample_token) 
    # lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

    # 加载点云数据
    # pc = LidarPointCloud.from_file(os.path.join(root_path,lidar_data['filename']))
  
    
    # my_sample = nusc.sample[0]
    # nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP')

    

    # sensor = 'LIDAR_TOP'
    # radar_front_data = nusc.get('sample_data', my_sample['data'][sensor])
    # print(radar_front_data)
    # nusc.render_sample_data(radar_front_data['token'])

    # # 将点云转换为numpy数组  
    # points = pc.points[:3, :].T  # 取出x, y, z坐标  
      
    # # 使用open3d创建点云对象  
    # pcd = o3d.geometry.PointCloud()  
    # pcd.points = o3d.utility.Vector3dVector(points)  
      
    # # 可视化  
    # o3d.visualization.draw_geometries([pcd])  
    
    # # 获取相机视图的点云
    # points = view_points(pointcloud.points, np.array(sensor_tokens['TRANSFORMATION_MATRIX']), normalize=True)

    # # 创建Open3D点云对象
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)

    # # 可视化点云
    # o3d.visualization.draw_geometries([pcd])
