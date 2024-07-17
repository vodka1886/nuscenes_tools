import secrets

import numpy as np  

from pyquaternion import Quaternion  
from scipy.spatial.transform import Rotation


def generate_random_key(length=32):
    alphabet = '0123456789abcdef'
    key = ''.join(secrets.choice(alphabet) for _ in range(length))
    return key

def remove_element_by_token(my_list, key, val):
    # 遍历列表找到目标元素的索引
    for index, element in enumerate(my_list):
        if element.get(key) == val:
            # 删除目标元素
            del my_list[index]
            return True  # 返回 True 表示删除成功
    return False  # 返回 False 表示没有找到目标元素

def eula_to_quaternion(y,p,r):
    # 欧拉角（以弧度为单位）  
    yaw = y  # 假设yaw为90度  
    pitch = p  # 假设pitch为30度  
    roll = r  # 假设roll为45度  
    
    # 计算四元数的各个分量  
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)  
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)  
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)  
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)  
    return Quaternion(qw,qx,qy,qz)
def translate(center, x: np.ndarray) -> None:
    """
    Applies a translation.
    :param x: <np.float: 3, 1>. Translation in x, y, z direction.
    """
    center_out = center + x
    return center_out

def rotate(center,orientation, quaternion: Quaternion) -> None:
    """
    Rotates box.
    :param quaternion: Rotation to apply.
    """
    center_out = np.dot(quaternion.rotation_matrix, center)
    orientation_out = quaternion * orientation
    return center_out,orientation_out

def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
    all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points
    
def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def adjust_ann_yaw(q,dYaw):
    yaw,pitch,roll = q.yaw_pitch_roll
    return eula_to_quaternion(np.deg2rad(np.rad2deg(yaw) + dYaw), pitch, roll)

def adjust_ann_loc(loc,q,forward_distance, lateral_distance):
    """
    计算车辆在给定前进距离和横向移动距离后的三维坐标。

    参数：
    position -- 当前坐标，格式为 [x, y, z]
    quaternion -- 表示方向的四元数，格式为 [w, x, y, z]
    forward_distance -- 前进的距离
    lateral_distance -- 横向移动的距离

    返回值：
    新的坐标，格式为 [x, y, z]
    """
    # 将四元数转换为旋转对象
    quaternion_wxyz = q.elements.tolist()
    quaternion_xyzw = quaternion_wxyz[1:] + [quaternion_wxyz[0]]
    rotation = Rotation.from_quat(quaternion_xyzw)
    
    euler_angles = rotation.as_euler('xyz', degrees=True) 
    
    # 获取前向方向向量和横向方向向量
    forward_vector = rotation.apply([1, 0, 0])  # 将局部x轴方向向量转换到全局坐标系
    lateral_vector = rotation.apply([0, 1, 0])  # 将局部y轴方向向量转换到全局坐标系
    
    # 计算新的位置
    new_position = loc + forward_distance * forward_vector + lateral_distance * lateral_vector
    
    return new_position

def global_pt_to_image(glo_loc,glo_rot: Quaternion,ego_translation: np.ndarray,ego_rotation: Quaternion,cam_translation: np.ndarray,cam_rotation: Quaternion,cam_intrinsic: np.ndarray):
    ego_loc = translate(glo_loc,-ego_translation)
    ego_loc,ego_rot = rotate(ego_loc,glo_rot,ego_rotation.inverse)

    #  Move box to sensor coord system.
    cam_loc = translate(ego_loc,-cam_translation)
    cam_loc,cam_rot = rotate(cam_loc,ego_rot,cam_rotation.inverse)
    
    # show in image
    pix_loc = view_points(cam_loc[:, np.newaxis] , cam_intrinsic, normalize=True)[:2, :]
    return pix_loc

def image_pt_to_global_pt(pix_loc,pix_rot: Quaternion,ego_translation: np.ndarray,ego_rotation: Quaternion,cam_translation: np.ndarray,cam_rotation: Quaternion,cam_intrinsic: np.ndarray):
    # nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    # cam_extrinsic = np.eye(4)
    # cam_extrinsic[:3, :3] = quaternion_to_rotation_matrix(cam_rotation)  # 将旋转部分放入变换矩阵的旋转子矩阵
    # cam_extrinsic[:3, 3] = cam_translation  # 将平移部分放入变换矩阵的平移向量
    
    # 计算像素坐标对应的相机坐标系下的3D坐标
    # image_width = cam_intrinsic[0, 2] * 2
    # image_height = cam_intrinsic[1, 2] * 2
    pixel_coord = np.array([pix_loc[0], pix_loc[1], 1])
    base_qu = eula_to_quaternion(0,0,0)
    # normalized_coord = pixel_coord / np.array([image_width, image_height, 1])
    # camera_coord = view_points(pixel_coord,np.linalg.inv(cam_intrinsic),False)
    
    # 将归一化平面坐标转换为相机坐标系下的坐标
    camera_coord = np.dot(np.linalg.inv(cam_intrinsic), pixel_coord)
    
    # Move camera coord system to ego vehicle coord system.
    ego_coord,ego_qu = rotate(camera_coord,base_qu,cam_rotation)
    ego_coord = translate(ego_coord,cam_translation)
    
    #  Move ego vehicle coord system to global coord system.
    global_coord,global_qu = rotate(ego_coord,ego_qu,ego_rotation)
    global_coord = translate(global_coord,ego_translation)

    return global_coord,global_qu   # 返回位置坐标（去除齐次坐标） 

def find_idx_in_list(eles:list,tar):
    ret = -1
    idx = 0
    for ele in eles:
        if ele == tar:
            ret = idx
            break
        idx = idx + 1
    return ret