import numpy as np
import cv2

def normalize_data(img, face_center, camera_matrix, gaze_target_3d=None):
    """
    标准数据归一化
    """
    # 数据合法性检查
    face_center = np.array(face_center, dtype=np.float32).reshape(-1)
    
    # 检查维度：必须是 3D 坐标 (x, y, z)
    if face_center.shape[0] != 3:
        raise ValueError(f"Face center dimension error: expected 3, got {face_center.shape[0]}")
        
    # 检查数值：不能是 [0, 0, 0]
    norm = np.linalg.norm(face_center)
    if norm < 1e-6:
        raise ValueError("Face center is zero vector (invalid 3D coordinate)")

    # 定义归一化后的虚拟相机参数
    focal_length = 960 
    normalized_image_size = (224, 224) 
    
    norm_camera_matrix = np.array([
        [focal_length, 0, normalized_image_size[0] / 2],
        [0, focal_length, normalized_image_size[1] / 2],
        [0, 0, 1]
    ])

    # 构建旋转矩阵 R
    forward = face_center / norm
    
    # 假设原相机坐标系: x右, y下, z前
    up = np.array([0, -1, 0], dtype=np.float32)
    
    # 计算新的 x 轴
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        # 退化情况处理
        forward = forward + np.array([0.001, 0.001, 0.001])
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
    
    right = right / right_norm
    
    # 计算新的 y 轴
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)
    
    R = np.vstack([right, down, forward])

    # 计算透视变换矩阵 W
    W = np.dot(np.dot(norm_camera_matrix, R), np.linalg.inv(camera_matrix))

    # 执行图像 Warp
    warped_img = cv2.warpPerspective(img, W, normalized_image_size)

    # 归一化 Gaze 向量
    normalized_gaze = None
    if gaze_target_3d is not None:
        gaze_target_3d = np.array(gaze_target_3d, dtype=np.float32).reshape(-1)
        if gaze_target_3d.shape[0] == 3:
            # 归一化注视向量 = R * 原始注视向量
            normalized_gaze = np.dot(R, gaze_target_3d)
            normalized_gaze = normalized_gaze / (np.linalg.norm(normalized_gaze) + 1e-6)
    
    return warped_img, normalized_gaze

def vector_to_pitchyaw(vector):
    """将归一化后的向量转换为 pitch, yaw (角度制)"""
    x, y, z = vector
    y = np.clip(y, -1.0, 1.0)
    pitch = np.arcsin(-y)
    yaw = np.arctan2(-x, -z)
    return np.degrees(np.array([yaw, pitch]))