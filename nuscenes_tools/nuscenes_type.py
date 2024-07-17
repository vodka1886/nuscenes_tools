from pyquaternion import Quaternion  
from nuscenes_tools.nuscenes_math import generate_random_key
import numpy as np  

size_map_g = {'car':[1.8, 4.0, 1.2],'truck':[2.0, 6, 2.5],'person':[0.3, 0.3, 1.8],'bicycle':[0.3, 1.5, 0.8]}

class EgoPoseData:
    def __init__(self,
                 token = None,
                 timestamp = 0,
                 rotation = [],
                 translation = [],
                ):
        if not token:
            token = generate_random_key()
        self.token = token
        self.timestamp = timestamp
        self.rotation = rotation
        self.translation = translation
        
    @classmethod
    def from_json(cls, data):
        return cls(
            data['token'],
            data['timestamp'],
            data['rotation'],
            data['translation'])
        
    def to_json(self):
        data = {
            'token': self.token,
            'timestamp': self.timestamp,
            'rotation': self.rotation,
            'translation': self.translation}
        return data

class SceneData:
    def __init__(self,
                 token = None,
                 log_token = "",
                 nbr_samples = 0,
                 first_sample_token = "",
                 last_sample_token = "",
                 name = "",
                 description = "",
                ):
        if not token:
            token = generate_random_key()
        self.token = token
        self.log_token = log_token
        self.nbr_samples = nbr_samples
        self.first_sample_token = first_sample_token
        self.last_sample_token = last_sample_token
        self.name = name
        self.description = description
        
    @classmethod
    def from_json(cls, data):
        return cls(
            data['token'],
            data['log_token'],
            data['nbr_samples'],
            data['first_sample_token'],
            data['last_sample_token'],
            data['name'],
            data['description'])
        
    def to_json(self):
        data = {
            'token': self.token,
            'log_token': self.log_token,
            'nbr_samples': self.nbr_samples,
            'first_sample_token': self.first_sample_token,
            'last_sample_token': self.last_sample_token,
            'name': self.name,
            'description': self.description}
        return data

class SampleData:
    def __init__(self,
                 token = None,
                 timestamp = 0,
                 prev = "",
                 next = "",
                 scene_token = ""
                ):
        if not token:
            token = generate_random_key()
        self.token = token
        self.timestamp = timestamp
        self.prev = prev
        self.next = next
        self.scene_token = scene_token   
        
    @classmethod
    def from_json(cls, data):
        return cls(
            data['token'],
            data['timestamp'],
            data['prev'],
            data['next'],
            data['scene_token'])
        
    def to_json(self):
        data = {
            'token': self.token,
            'timestamp': self.timestamp,
            'prev': self.prev,
            'next': self.next,
            'scene_token': self.scene_token}
        return data

class SampleDataData:
    def __init__(self,
                 token = None,
                 sample_token = "",
                 ego_pose_token = "",
                 calibrated_sensor_token = "",
                 timestamp = 0,
                 fileformat = "",
                 is_key_frame = "",
                 height = 0,
                 width = 0,
                 filename = "",
                 prev = "",
                 next = ""
                ):
        if not token:
            token = generate_random_key()
        self.token = token
        self.sample_token = sample_token
        self.ego_pose_token = ego_pose_token
        self.calibrated_sensor_token = calibrated_sensor_token
        self.timestamp = timestamp   
        self.fileformat = fileformat
        self.is_key_frame = is_key_frame
        self.height = height
        self.width = width  
        self.timestamp = timestamp
        self.filename = filename
        self.prev = prev
        self.next = next  
        
    @classmethod
    def from_json(cls, data):
        return cls(
            data['token'],
            data['sample_token'],
            data['ego_pose_token'],
            data['calibrated_sensor_token'],
            data['timestamp'],
            data['fileformat'],
            data['is_key_frame'],
            data['height'],
            data['width'],
            data['filename'],
            data['prev'],
            data['next'])
        
    def to_json(self):
        data = {
            'token': self.token,
            'sample_token': self.sample_token,
            'ego_pose_token': self.ego_pose_token,
            'calibrated_sensor_token': self.calibrated_sensor_token,
            'timestamp': self.timestamp,
            'fileformat': self.fileformat,
            'is_key_frame': self.is_key_frame,
            'height': self.height,
            'width': self.width,
            'filename': self.filename,
            'prev': self.prev,
            'next': self.next}
        return data

class InstanceData:
    def __init__(self,
                 token = None,
                 category_token = "",
                 nbr_annotations = 1,
                 first_annotation_token = "",
                 last_annotation_token = ""
                ):
        if not token:
            token = generate_random_key()
        self.token = token
        self.category_token = category_token
        self.nbr_annotations = nbr_annotations
        self.first_annotation_token = first_annotation_token
        self.last_annotation_token = last_annotation_token   
        
    @classmethod
    def from_json(cls, data):
        return cls(
            data['token'],
            data['category_token'],
            data['nbr_annotations'],
            data['first_annotation_token'],
            data['last_annotation_token'])
        
    def to_json(self):
        data = {
            'token': self.token,
            'category_token': self.category_token,
            'nbr_annotations': self.nbr_annotations,
            'first_annotation_token': self.first_annotation_token,
            'last_annotation_token': self.last_annotation_token}
        return data
    
    def set_category(self,c_type:str):
        if c_type == 'car':
            self.category_token = 'fd69059b62a3469fbaef25340c0eab7f'
        if c_type == 'truck':
            self.category_token = '6021b5187b924d64be64a702e5570edf'
        if c_type == 'person':
            self.category_token = '1fa93b757fc74fb197cdd60001ad8abf' 
        if c_type == 'bicycle':
            self.category_token = 'fc95c87b806f48f8a1faea2dcc2222a4' 

class AnnotationData:
    def __init__(self,
                 token = None, 
                 sample_token = "", 
                 instance_token = "", 
                 visibility_token = "1", 
                 attribute_tokens = "",
                 translation = [], 
                 size = [], 
                 rotation = [], 
                 prev = "", 
                 next_ = "", 
                 num_lidar_pts = 0, 
                 num_radar_pts = 0
                 ):
        if not token:
            token = generate_random_key()
        self.token = token
        self.sample_token = sample_token
        self.instance_token = instance_token
        self.visibility_token = visibility_token
        self.attribute_tokens = attribute_tokens
        self.translation = translation
        self.size = size
        self.rotation = rotation
        self.prev = prev
        self.next = next_
        self.num_lidar_pts = num_lidar_pts
        self.num_radar_pts = num_radar_pts
        self.instance = None
            
    @classmethod
    def from_json(cls, data):
        return cls(
            data['token'],
            data['sample_token'],
            data['instance_token'],
            data['visibility_token'],
            data['attribute_tokens'],
            data['translation'],
            data['size'],
            data['rotation'],
            data['prev'],
            data['next'],
            data['num_lidar_pts'],
            data['num_radar_pts']
        )

    def to_json(self):
        data = {
            'token': self.token,
            'sample_token': self.sample_token,
            'instance_token': self.instance_token,
            'visibility_token': self.visibility_token,
            'attribute_tokens': self.attribute_tokens,
            'translation': self.translation,
            'size': self.size,
            'rotation': self.rotation,
            'prev': self.prev,
            'next': self.next,
            'num_lidar_pts': self.num_lidar_pts,
            'num_radar_pts': self.num_radar_pts
        }
        return data
    def set_translation(self,translation:np.ndarray):
        self.translation = translation.tolist()
    def set_rotation(self,rotation:Quaternion):
        self.rotation = rotation.elements.tolist()
    def set_classtype(self,c_type:str):
        if self.instance == None:
            self.instance = InstanceData()
            self.instance_token = self.instance.token
        self.instance.set_category(c_type)
        self.size = self.get_size_by_category(c_type)
    def get_size_by_category(self,c_type:str):
        global size_map_g
        defalt_size = [2, 4, 1.2]
        if c_type in size_map_g:
            defalt_size = size_map_g[c_type]
        return defalt_size
        
