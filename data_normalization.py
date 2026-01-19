import numpy as np
import cv2

def normalize_data(img, face_center, camera_matrix, gaze_target_3d=None):
    face_center = np.array(face_center, dtype=np.float32).reshape(-1)
    
    if face_center.shape[0] != 3:
        raise ValueError(f"Face center dimension error: expected 3, got {face_center.shape[0]}")
        
    norm = np.linalg.norm(face_center)
    if norm < 1e-6:
        raise ValueError("Face center is zero vector (invalid 3D coordinate)")

    focal_length = 960 
    normalized_image_size = (224, 224) 
    
    norm_camera_matrix = np.array([
        [focal_length, 0, normalized_image_size[0] / 2],
        [0, focal_length, normalized_image_size[1] / 2],
        [0, 0, 1]
    ])

    forward = face_center / norm
    
    up = np.array([0, -1, 0], dtype=np.float32)
    
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        forward = forward + np.array([0.001, 0.001, 0.001])
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
    
    right = right / right_norm
    
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)
    
    R = np.vstack([right, down, forward])

    W = np.dot(np.dot(norm_camera_matrix, R), np.linalg.inv(camera_matrix))

    warped_img = cv2.warpPerspective(img, W, normalized_image_size)

    normalized_gaze = None
    if gaze_target_3d is not None:
        gaze_target_3d = np.array(gaze_target_3d, dtype=np.float32).reshape(-1)
        if gaze_target_3d.shape[0] == 3:
            normalized_gaze = np.dot(R, gaze_target_3d)
            normalized_gaze = normalized_gaze / (np.linalg.norm(normalized_gaze) + 1e-6)
    
    return warped_img, normalized_gaze

def vector_to_pitchyaw(vector):
    x, y, z = vector
    y = np.clip(y, -1.0, 1.0)
    pitch = np.arcsin(-y)
    yaw = np.arctan2(-x, -z)
    return np.degrees(np.array([yaw, pitch]))
