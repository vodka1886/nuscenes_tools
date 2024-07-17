import json
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion  
import os
import os.path as osp
import shutil
import numpy as np  

from nuscenes_tools.nuscenes_type import *

class NuscenesExplorer:
    def __init__(self,root_path:str,version:str):
        self.root_path = root_path
        self.version = version
        self.table_names = ['category', 'attribute', 'visibility', 'instance', 'sensor', 'calibrated_sensor',
                            'ego_pose', 'log', 'scene', 'sample', 'sample_data', 'sample_annotation', 'map']
        
        self.load()
        
    def load(self):
        if not os.path.exists(self.root_path):
            return
        if not os.path.exists(os.path.join(self.root_path,self.version)):
            self.nusc = None
            return
        self.nusc = NuScenes(version=self.version, dataroot=self.root_path, verbose=True)
        
    def get_ann_loc(self,sample_ann_token:str):
        ann_data = self.nusc.get('sample_annotation', sample_ann_token)
        r = ann_data['translation']
        return r

    def get_ann_qu(self,sample_ann_token:str):
        # 初始化NuScenes对象
        ann_data = self.nusc.get('sample_annotation', sample_ann_token)
        r = ann_data['rotation']
        return Quaternion(r[0], r[1], r[2], r[3]) 

    def set_ann_qu(self,sample_ann_token:str,q:Quaternion):
        # 初始化NuScenes对象
        ann_file = os.path.join(self.nusc.dataroot,self.nusc.version,'sample_annotation.json')
        
        # 读取JSON文件  
        with open(ann_file, 'r', encoding='utf-8') as f:  
            data = json.load(f) 
        
        # 搜索token值为3333的元素  
        # 假设我们要在一个列表（比如名为'items'）中搜索这个元素  
        # 注意：这里的'items'应该根据你的实际JSON结构来替换  
        modified = False  
        for item in data:  # 假设items是包含多个元素的列表  
            if item.get('token') == sample_ann_token:  
                # 修改name子元素  
                print("find!")
                print('type: ',type(item['rotation']))
                print('type: ',type(q.elements))
                item['rotation'] = q.elements.tolist()
                modified = True
                break  # 如果只需要修改第一个匹配的元素，可以使用break  
        
        # 如果找到了并修改了元素，则写回文件  
        if modified:  
            with open(ann_file, 'w', encoding='utf-8') as f:  
                json.dump(data, f, ensure_ascii=False, indent=4)  # 确保中文等字符能正确写入，并美化输出  
            print('File updated successfully.')  
        else:  
            print('No item with token found.')
        # reload dataset
        self.load()

    def set_ann_loc(self,sample_ann_token:str,loc):
        # 初始化NuScenes对象
        ann_file = os.path.join(self.nusc.dataroot,self.nusc.version,'sample_annotation.json')
        
        # 读取JSON文件  
        with open(ann_file, 'r', encoding='utf-8') as f:  
            data = json.load(f) 
        
        # 搜索token值为3333的元素  
        # 假设我们要在一个列表（比如名为'items'）中搜索这个元素  
        # 注意：这里的'items'应该根据你的实际JSON结构来替换  
        modified = False  
        for item in data:  # 假设items是包含多个元素的列表  
            if item.get('token') == sample_ann_token:  
                # 修改name子元素  
                print("find!")
                print('type: ',type(item['translation']))
                print('type: ',type(loc))
                item['translation'] = loc.tolist()
                modified = True
                break  # 如果只需要修改第一个匹配的元素，可以使用break 
        # 如果找到了并修改了元素，则写回文件  
        if modified:  
            with open(ann_file, 'w', encoding='utf-8') as f:  
                json.dump(data, f, ensure_ascii=False, indent=4)  # 确保中文等字符能正确写入，并美化输出  
            print('File updated successfully.')  
        else:  
            print('No item with token found.')
        # reload dataset
        self.load()

    def add_info_to_dataset(self,key:str,info):
        file = os.path.join(self.nusc.dataroot,self.nusc.version,key + '.json')
        # 读取原始 JSON 文件
        with open(file, 'r') as f:
            file_data = json.load(f)
        info_json = info.to_json()
        # check token
        is_used = False
        for ele in file_data:
            if ele['token'] == info_json['token']:
                is_used = True
                break
        if is_used:
            return
        file_data.append(info_json)
        
        
        with open(file, 'w', encoding='utf-8') as f:  
            json.dump(file_data, f, ensure_ascii=False, indent=4)  # 确保中文等字符能正确写入，并美化输出  
            print('Dataset file updated successfully.')  
    
    def add_ann(self,target:AnnotationData):
        # add ann
        ann_file = os.path.join(self.nusc.dataroot,self.nusc.version,'sample_annotation.json')
        # 读取原始 JSON 文件
        with open(ann_file, 'r') as f:
            ann_data = json.load(f)
        ann_json = target.to_json()
        ann_data.append(ann_json)
        
        with open(ann_file, 'w', encoding='utf-8') as f:  
            json.dump(ann_data, f, ensure_ascii=False, indent=4)  # 确保中文等字符能正确写入，并美化输出  
            print('Annotation file updated successfully.')  
        
        # add instance
        ins_file = os.path.join(self.nusc.dataroot,self.nusc.version,'instance.json')
        with open(ins_file, 'r') as f:
            ins_data = json.load(f)
        ins_json = target.instance.to_json()
        ins_data.append(ins_json)
        
        with open(ins_file, 'w', encoding='utf-8') as f:  
            json.dump(ins_data, f, ensure_ascii=False, indent=4)  # 确保中文等字符能正确写入，并美化输出  
            print('Instance file updated successfully.')  
        # reload dataset
        self.load()
        
    @staticmethod   
    def generate_target(translation:np.ndarray,rotation:Quaternion,c_type:str = "car"):
        tar = AnnotationData()
        tar.set_rotation(rotation)
        tar.set_translation(translation)
        tar.set_classtype(c_type)
        return tar

    def get_cam_calis(self,cam_token):
        cam_data = self.nusc.get('sample_data', cam_token)
        cam_calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        ego_calib = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        Ci = np.array(cam_calib['camera_intrinsic'])
        Ct = np.array(cam_calib['translation'])
        Cr = Quaternion(cam_calib['rotation'])
        Et = np.array(ego_calib['translation'])
        Er = Quaternion(ego_calib['rotation'])
        return Ci,Ct,Cr,Et,Er
    
    def list_scenes(self) -> list:
        """ Lists all scenes with some meta data. """

        def ann_count(record):
            count = 0
            sample = self.nusc.get('sample', record['first_sample_token'])
            while not sample['next'] == "":
                count += len(sample['anns'])
                sample = self.nusc.get('sample', sample['next'])
            return count

        recs = [(self.nusc.get('sample', record['first_sample_token'])['timestamp'], record) for record in
                self.nusc.scene]
        rets = [record['token'] for record in
                self.nusc.scene]
        idx = 0
        for start_time, record in recs:
            start_time = self.nusc.get('sample', record['first_sample_token'])['timestamp'] / 1000000
            length_time = self.nusc.get('sample', record['last_sample_token'])['timestamp'] / 1000000 - start_time
            location = self.nusc.get('log', record['log_token'])['location']
            desc = record['name'] + ', ' + record['description']
            if len(desc) > 55:
                desc = desc[:51] + "..."
            if len(location) > 18:
                location = location[:18]

            print('{} : {:16} {} {:4.0f}s, {}, #anns:{}'.format(idx,
                desc,record['token'],
                length_time, location, ann_count(record)))
            idx = idx + 1
        return rets
            
    def list_samples(self,scene_token:str=None) -> list:
        """ Lists all scenes with some meta data. """
        if not scene_token:
            scene_token = self.nusc.scene[0]['token']
        
        recs = []
        for sample in self.nusc.sample:
            # ann = self.nusc.get('sample_annotation', ann_token)
            if sample['scene_token'] == scene_token:
                recs.append(sample['token'])
             
        idx = 0
        for sample_token in recs:
            print('{} :  {}'.format(idx,
                sample_token))
            idx = idx + 1
        return recs
    
    def list_anns(self,sample_token:str=None) -> list:
        """ Lists all scenes with some meta data. """
        if not sample_token:
            scene_token = self.nusc.scene[0]['token']
            sample_token = self.nusc.get('scene', scene_token)['first_sample_token']
        # ann_tokens =self.nusc.get('sample', sample_token)['anns']
        recs = []
        
        for ann in self.nusc.sample_annotation:
            # ann = self.nusc.get('sample_annotation', ann_token)
            if ann['sample_token'] == sample_token:
                recs.append(ann['token'])
             
        idx = 0
        for ann_token in recs:
            print('{} :  {}'.format(idx,
                ann_token))
            idx = idx + 1
        return recs
    
    def list_sample_data(self,sample_token:str=None):
        if not sample_token:
            scene_token = self.nusc.scene[0]['token']
            sample_token = self.nusc.get('scene', scene_token)['first_sample_token']
        # find first sample_data
        # first_sample_data_token = None
        ret = []
        for sample_data in self.nusc.sample_data:
            if sample_data['sample_token'] == sample_token:
                ret.append(sample_data['token'])
                # first_sample_data_token = sample_data['token']
                # break
        
        # find pre
        # current_sample_data_token = first_sample_data_token
        # while current_sample_data_token != "":
        #     current_sample_data = self.nusc.get('sample_data', current_sample_data_token)
        #     if current_sample_data['sample_token'] != sample_token:
        #         break
        #     ret.append(current_sample_data['token'])
        #     current_sample_data_token = current_sample_data["prev"]
        # # find next
        # current_sample_data_token = self.nusc.get('sample_data', first_sample_data_token)['next']
        # while current_sample_data_token != "":
        #     current_sample_data = self.nusc.get('sample_data', current_sample_data_token)
        #     if current_sample_data['sample_token'] != sample_token:
        #         break
        #     ret.append(current_sample_data['token'])
        #     current_sample_data_token = current_sample_data["next"]
        return ret
            
    def create_empty_dataset(self,nusc:NuScenes):
        version_path = os.path.join(self.root_path,self.version)
        os.makedirs(version_path, exist_ok=True)
        copy_names = ['category', 'attribute', 'visibility', 'sensor', 'calibrated_sensor',
                             'log', 'map']
        data=[]
        for name in self.table_names:
            if name in copy_names:
                src_file = os.path.join(nusc.dataroot,nusc.version,name+".json")
                dst_file = os.path.join(version_path,name+".json")
                if os.path.exists(src_file):
                    shutil.copy(src_file, dst_file)
            else:
                file = os.path.join(version_path,name+".json")
                with open(file, 'w', encoding='utf-8') as f:  
                    json.dump(data, f, ensure_ascii=False, indent=4)  # 确保中文等字符能正确写入，并美化输出  

    def copy_sample_form_dataset(self,nusc_ep,sample_token:str):
        # read sample info
        sample = nusc_ep.nusc.get('sample', sample_token)
    
        # copy scene
        if sample['scene_token'] not in self.list_scenes():
            scene = nusc_ep.nusc.get('scene', sample['scene_token'])
            scene['first_sample_token'] = sample['token']
            scene['last_sample_token'] = sample['token']
            sample['prev'] = ""
            sample['next'] = ""
            sceneData = SceneData.from_json(scene)
            self.add_info_to_dataset("scene",sceneData)
        else:
            scene = self.nusc.get('scene', sample['scene_token'])
            last_sample = self.nusc.get('sample', scene['last_sample_token'])
            last_sample['next'] = sample['token']
            sample['prev'] = last_sample['token']
            scene['last_sample_token'] = sample['token']
            sceneData = SceneData.from_json(scene)
            self.add_info_to_dataset("scene",sceneData)
        # copy sample   
        sampleData = SampleData.from_json(sample)
        self.add_info_to_dataset("sample",sampleData)
            
        
        # copy sample_data
        sample_data_tokens = nusc_ep.list_sample_data(sample_token)
        for sample_data_token in sample_data_tokens:
            # add sample_data
            sample_data = nusc_ep.nusc.get('sample_data', sample_data_token)
            sampleDataData = SampleDataData.from_json(sample_data)
            self.add_info_to_dataset("sample_data",sampleDataData)
            
            # add ego_pose
            ego_pose = nusc_ep.nusc.get('ego_pose', sample_data['ego_pose_token'])
            egoPoseData = EgoPoseData.from_json(ego_pose)
            self.add_info_to_dataset("ego_pose",egoPoseData)
        
        # copy sampe_ann
        category_map = ["1fa93b757fc74fb197cdd60001ad8abf",
                        "fd69059b62a3469fbaef25340c0eab7f",
                        "dfd26f200ade4d24b540184e16050022",
                        "fc95c87b806f48f8a1faea2dcc2222a4",
                        "6021b5187b924d64be64a702e5570edf",
                        ]
        ann_tokens = nusc_ep.list_anns(sample_token)
        for ann_token in ann_tokens:
            # check instance type
            ann = nusc_ep.nusc.get('sample_annotation', ann_token)
            instance = nusc_ep.nusc.get('instance', ann['instance_token'])
            if instance['category_token'] not in category_map:
                continue
            # add sample_data
            annData = AnnotationData.from_json(ann)
            self.add_info_to_dataset("sample_annotation",annData)
            
            # add instance 
            instanceData = InstanceData.from_json(instance)
            self.add_info_to_dataset("instance",instanceData)
        self.load()
            

