## visualize skeleton of two agents


import numpy as np
import cv2
import os 
from os.path import isfile, join
from os import listdir
from moviepy.editor import VideoFileClip, AudioFileClip


SMPLX_JOINT_NAMES = [
    'pelvis', #0
    'left_hip', 
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle', # 7
    'right_ankle', # 8
    'spine3', 
    'left_foot', # 10
    'right_foot', # 11
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow', 
    'right_elbow',
    'left_wrist', #20
    'right_wrist', #21
    'jaw', #22
    'left_eye_smplhf', #23
    'right_eye_smplhf', #24
    'left_index1', #25
    'left_index2', #26
    'left_index3', #27
    'left_middle1', #28
    'left_middle2', #29
    'left_middle3', #30
    'left_pinky1', #31
    'left_pinky2', #32
    'left_pinky3', #33 
    'left_ring1', #34
    'left_ring2',# 35
    'left_ring3', #36
    'left_thumb1', #37
    'left_thumb2', #38
    'left_thumb3', #39
    'right_index1', #40
    'right_index2', 
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3'
]



# Define edges to connect joints
edges = [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 4],
    [2, 5],
    [3, 6],
    [4, 7],
    [5, 8],
    [6, 9],
    [7, 10],
    [8, 11],
    [9, 12],
    [9, 13],
    [9, 14],
    [12, 15],
    [13, 16],
    [14, 17],
    [16, 18],
    [17, 19],
    [18, 20],
    [19, 21],
    [20, 25], [25, 26], [26, 27],
    [20, 28], [28, 29], [29, 30],
    [20, 31], [20, 32], [20, 33],
    [20, 34], [20, 35], [20, 36],
    [20, 37], [20, 38], [20, 39],

    [21, 40], [40, 41], [41, 42],
    [21, 43], [21, 44], [21, 45],
    [21, 46], [46, 47], [47, 48],
    [21, 49], [49, 50], [50, 51],
    [21, 52], [52, 53], [53, 54]
]

# Create figure and scene
def visualize(joints, evaldir, dance_names, zoom=100, **kwargs):
    save_folder = os.path.join(evaldir, 'videos')
    os.makedirs(save_folder, exist_ok=True)

    # 确定 joints_num
    # 假设 joints[0] 是 (motion_data, label_data) 或 (motion_data)
    sample_motion_data = joints[0][0] if isinstance(joints[0], tuple) else joints[0]
    
    # 兼容 (frames, joints*3) 和 (batch, frames, joints*3) 两种维度
    if sample_motion_data.ndim == 3: # (batch, frames, joints*3)
        joints_num = sample_motion_data.shape[-1] // 3
    elif sample_motion_data.ndim == 2: # (frames, joints*3)
        joints_num = sample_motion_data.shape[-1] // 3
    else:
        raise ValueError(f"Unexpected joints dimension: {sample_motion_data.ndim}. Expected 2 or 3.")

    colors = [(255, 80, 80)]
    output_videos = []

    for ii in range(len(joints)):
        # 确保正确提取动作数据
        this_joints = joints[ii][0].reshape(-1, joints_num, 3) if isinstance(joints[ii], tuple) else joints[ii].reshape(-1, joints_num, 3)
        
        video_name = os.path.join(save_folder, f'{dance_names[ii]}.mp4')
        output_videos.append(video_name)

        frame_rate = 30.0
        frame_size = (960, 540)

        frames = []

        # Generate frames
        for i in range(len(this_joints)):
            frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

            for j in range(len(edges)):
                if edges[j][0] < joints_num and edges[j][1] < joints_num:
                    pt1 = tuple((this_joints[i][edges[j][0]] * zoom).astype(int))
                    pt2 = tuple((this_joints[i][edges[j][1]] * zoom).astype(int))
                    
                    # 调整偏移量以使骨骼居中显示
                    # 假设你的坐标系：X轴是左右，Y轴是前后，Z轴是上下
                    # 屏幕坐标系：X是左右，Y是上下（向下为正）
                    # 因此，pt1[0] 对应屏幕X，-pt1[2] 对应屏幕Y (因为Z轴向上为正，屏幕Y向下为正)
                    # config.width // 2 和 config.height // 2 用于居中
                    cv2.line(frame, 
                             [pt1[0] + frame_size[0] // 2, -pt1[2] + frame_size[1] // 2], 
                             [pt2[0] + frame_size[0] // 2, -pt2[2] + frame_size[1] // 2], 
                             colors[0], thickness=2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
            frames.append(frame)

        print(f"[{ii + 1}/{len(joints)}] Frames generated for {video_name}")

        # Create a video clip from the frames
        clip = ImageSequenceClip(frames, fps=frame_rate)
        clip = clip.set_audio(None)

        # Write once with optional audio
        clip.write_videofile(
            video_name,
            codec='libx264',
            audio_codec='aac',
            # 建议使用绝对路径来防止临时文件问题
            temp_audiofile=os.path.abspath(os.path.join(save_folder, 'temp-audio.m4a')),
            remove_temp=True,
            # 可以添加logger='bar'来查看更详细的 ffmpeg 输出，以帮助调试
            # logger='bar'
        )
        
    return output_videos

def determine_up_axis(joints, joints_num):
    # SMPL-X 关节索引
    head_idx = 15  # 'head'
    left_foot_idx = 10  # 'left_foot'
    right_foot_idx = 11  # 'right_foot'
    
    # 取第一组关节数据（形状为 (帧数, 关节数 * 3)）
    this_joints = joints[0][0].reshape(-1, joints_num, 3)  # 重塑为 (帧数, 关节数, 3)
    
    # 提取头部和脚部坐标
    head_pos = this_joints[:, head_idx, :]  # 形状为 (帧数, 3)
    left_foot_pos = this_joints[:, left_foot_idx, :]  # 形状为 (帧数, 3)
    right_foot_pos = this_joints[:, right_foot_idx, :]  # 形状为 (帧数, 3)
    
    # 计算脚部平均位置
    foot_pos = (left_foot_pos + right_foot_pos) / 2  # 形状为 (帧数, 3)
    
    # 计算头部和脚部的平均坐标差异
    diff = np.mean(head_pos - foot_pos, axis=0)  # 形状为 (3,)，表示 x, y, z 的平均差
    
    # 找到绝对值最大的轴
    max_axis = np.argmax(np.abs(diff))
    up_axis = ['x', 'y', 'z'][max_axis]
    
    # 判断方向（正或负）
    direction = 'positive' if diff[max_axis] > 0 else 'negative'
    
    print(f"Determined up axis: {up_axis} ({direction})")
    return up_axis, direction

def visualize2(joints, joints_leader, config, evaldir, dance_names, zoom=100):
    '''
    video 左上角原点，人物z为上方向（与video纵轴方向相反，所以需要-z）
     (0, 0)
        ------------->
        |
        |
        |
        |
        |
        v
    '''
    save_folder = os.path.join(evaldir, 'videos')
    os.makedirs(save_folder, exist_ok=True)
    
    joints_num = joints[0][0].shape[-1] // 3
    colors = [(80, 80, 255)]  # follower color
    colors_l = [(255, 80, 80)]  # leader color
    output_videos = []
    
    # 判断上方向
    up_axis, direction = determine_up_axis(joints, joints_num)
    
    print(len(joints), len(joints_leader), flush=True)
    len_joints = min(len(joints), len(joints_leader))

    for ii in range(len_joints):
        this_joints = joints[ii][0]
        this_leader_joints = joints_leader[ii][0]
        this_leader_joints = this_leader_joints.reshape(len(this_leader_joints), joints_num, 3)
        this_joints = this_joints.reshape(len(this_joints), joints_num, 3)

        video_name = os.path.join(save_folder, f'{dance_names[ii]}.mp4')
        output_videos.append(video_name)

        frame_rate = 30.0
        frame_size = (config.width, config.height)

        frames = []

        # Generate frames
        for i in range(len(this_joints)):
            frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

            # --- 添加 XYZ 坐标轴 ---
            origin = (480, 270)
            # axis_length = 50
            # x_end = (origin[0] + axis_length, origin[1])
            # cv2.line(frame, origin, x_end, (0, 0, 255), 2)
            # cv2.putText(frame, 'X', x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # y_end = (origin[0], origin[1] - axis_length)
            # cv2.line(frame, origin, y_end, (0, 255, 0), 2)
            # cv2.putText(frame, 'Y', y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # z_end = (origin[0], origin[1] + axis_length)
            # cv2.line(frame, origin, z_end, (255, 0, 0), 2)
            # cv2.putText(frame, 'Z', z_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            y_offset, z_offset = origin
            # --- 绘制关节连线 ---
            for j in range(len(edges)):
                if edges[j][0] < joints_num and edges[j][1] < joints_num:
                    pt1 = tuple((this_joints[i][edges[j][0]] * zoom).astype(int))
                    pt2 = tuple((this_joints[i][edges[j][1]] * zoom).astype(int))
                    cv2.line(frame, [pt1[1] + y_offset, -pt1[2] + z_offset], [pt2[1] + y_offset, -pt2[2] + z_offset], colors[0], thickness=2)

            for j in range(len(edges)):
                if edges[j][0] < joints_num and edges[j][1] < joints_num:
                    pt1 = tuple((this_leader_joints[i][edges[j][0]] * zoom).astype(int))
                    pt2 = tuple((this_leader_joints[i][edges[j][1]] * zoom).astype(int))
                    cv2.line(frame, [pt1[1] + y_offset, -pt1[2] + z_offset], [pt2[1] + y_offset, -pt2[2] + z_offset], colors_l[0], thickness=2)

            # Convert BGR to RGB (OpenCV uses BGR by default)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        print(f"[{ii + 1}/{len(joints)}] Frames generated for {video_name}")

        # Create a video clip from the frames
        clip = ImageSequenceClip(frames, fps=frame_rate)
        clip = clip.set_audio(None)

        # Write once with optional audio
        clip.write_videofile(
            video_name,
            codec='libx264',
            audio_codec='aac',
            # 建议使用绝对路径来防止临时文件问题
            temp_audiofile=os.path.abspath(os.path.join(save_folder, 'temp-audio.m4a')),
            remove_temp=True,
            # 可以添加logger='bar'来查看更详细的 ffmpeg 输出，以帮助调试
            # logger='bar'
        )
        
    return output_videos


# def visualize3d(joints, joints_leader, config, evaldir, dance_names, quants=None):
#     save_folder = os.path.join(evaldir, 'videos')
#     os.makedirs(save_folder, exist_ok=True)
    
#     joints_num = joints[0][0].shape[-1] // 3
#     distance=1
    
#     # Define colors
#     colors = [(80, 80, 255)]  # Blue for follower
#     colors_l = [(255, 80, 80)]  # Red for leader
#     output_videos = []
    
#     print(len(joints), len(joints_leader), flush=True)
#     len_joints = min(len(joints), len(joints_leader))

#     def draw_joints_with_perspective(frame, joints_data, edges, color, frame_size, radius=4, distance=15):
#         """
#         Draw joints and ground plane on a frame with perspective projection.
#         """
#         # Define perspective projection parameters to match plot_3d_motion
#         fov = 45  # Field of view in degrees
#         elev = np.radians(120)  # Elevation angle
#         azim = np.radians(-10)  # Azimuth angle

#         # Create rotation matrices for elevation and azimuth
#         R_elev = np.array([
#             [1, 0, 0],
#             [0, np.cos(elev), -np.sin(elev)],
#             [0, np.sin(elev), np.cos(elev)]
#         ])
#         R_azim = np.array([
#             [np.cos(azim), -np.sin(azim), 0],
#             [np.sin(azim), np.cos(azim), 0],
#             [0, 0, 1]
#         ])
#         R = R_elev @ R_azim  # Combined rotation matrix

#         # Perspective projection matrix
#         aspect = frame_size[0] / frame_size[1]
#         near = 1
#         far = 100
#         scale = np.tan(np.radians(fov) / 2) * near
#         P = np.array([
#             [near / (scale * aspect), 0, 0, 0],
#             [0, near / scale, 0, 0],
#             [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
#             [0, 0, -1, 0]
#         ])

#         # Center and scale coordinates
#         center = np.array([0, radius / 4, radius / 4])

#         # Draw ground plane (XZ plane at y=0)
#         minx, maxx, minz, maxz = -3, 3, -3, 3  # Same as plot_3d_motion
#         miny = 0  # Ground plane at y=0
#         verts_3d = [
#             np.array([minx, miny, minz], dtype=np.float64),
#             np.array([minx, miny, maxz], dtype=np.float64),
#             np.array([maxx, miny, maxz], dtype=np.float64),
#             np.array([maxx, miny, minz], dtype=np.float64)
#         ]
#         verts_2d = []
#         for vert in verts_3d:
#             vert -= center
#             vert *= 2 / radius
#             vert_rot = R @ vert
#             vert_4d = np.append(vert_rot, 1)
#             vert_proj = P @ vert_4d
#             vert_2d = vert_proj[:2] / vert_proj[3]
#             vert_2d = (vert_2d * np.array([frame_size[0]/2, -frame_size[1]/2]) * distance / 15 + 
#                        np.array([frame_size[0]/2, frame_size[1]/2])).astype(int)
#             verts_2d.append([vert_2d[0], vert_2d[1]])

#         # Draw semi-transparent ground plane
#         overlay = frame.copy()
#         cv2.fillPoly(overlay, [np.array(verts_2d)], (128, 128, 128))  # Gray color
#         cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)  # 50% transparency

#         # Draw joints
#         for j in range(len(edges)):
#             if edges[j][0] < joints_num and edges[j][1] < joints_num:
#                 pt1_3d = joints_data[edges[j][0]].copy()
#                 pt2_3d = joints_data[edges[j][1]].copy()

#                 # Normalize coordinates
#                 pt1_3d -= center
#                 pt2_3d -= center
#                 pt1_3d *= 2 / radius
#                 pt2_3d *= 2 / radius

#                 # Apply rotation
#                 pt1_3d = R @ pt1_3d
#                 pt2_3d = R @ pt2_3d

#                 # Convert to homogeneous coordinates
#                 pt1_4d = np.append(pt1_3d, 1)
#                 pt2_4d = np.append(pt2_3d, 1)

#                 # Apply perspective projection
#                 pt1_proj = P @ pt1_4d
#                 pt2_proj = P @ pt2_4d
#                 pt1_2d = pt1_proj[:2] / pt1_proj[3]
#                 pt2_2d = pt2_proj[:2] / pt2_proj[3]

#                 # Map to frame coordinates with distance scaling
#                 pt1_2d = (pt1_2d * np.array([frame_size[0]/2, -frame_size[1]/2]) * distance / 15 + 
#                           np.array([frame_size[0]/2, frame_size[1]/2])).astype(int)
#                 pt2_2d = (pt2_2d * np.array([frame_size[0]/2, -frame_size[1]/2]) * distance / 15 + 
#                           np.array([frame_size[0]/2, frame_size[1]/2])).astype(int)

#                 # Draw line
#                 cv2.line(frame, (pt1_2d[0], pt1_2d[1]), (pt2_2d[0], pt2_2d[1]), color, thickness=2)

#     for ii in range(len_joints):
#         this_joints = joints[ii][0].reshape(-1, joints_num, 3)
#         this_leader_joints = joints_leader[ii][0].reshape(-1, joints_num, 3)
        
#         # Normalize joints (similar to plot_3d_motion)
#         MINS = np.minimum(this_joints.min(axis=0).min(axis=0), this_leader_joints.min(axis=0).min(axis=0))
#         height_offset = MINS[1]
#         this_joints[:, :, 1] -= height_offset
#         this_leader_joints[:, :, 1] -= height_offset
        
#         video_name = os.path.join(save_folder, dance_names[ii] + '.mp4')
#         output_videos.append(video_name)
        
#         frame_rate = 30.0
#         frame_size = (config.width, config.height)

#         # Initialize video writer
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(video_name, fourcc, frame_rate, frame_size)

#         # Generate frames
#         for i in range(len(this_joints)):
#             # Create blank frame
#             frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            
#             # Add XYZ coordinate axes
#             origin = (480, 270)
#             axis_length = 50
#             # X-axis (red)
#             x_end = (origin[0] + axis_length, origin[1])
#             cv2.line(frame, origin, x_end, (0, 0, 255), 2)
#             cv2.putText(frame, 'X', x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
#             # Y-axis (green)
#             y_end = (origin[0], origin[1] - axis_length)
#             cv2.line(frame, origin, y_end, (0, 255, 0), 2)
#             cv2.putText(frame, 'Y', y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#             # Z-axis (blue)
#             z_end = (origin[0], origin[1] + axis_length)
#             cv2.line(frame, origin, z_end, (255, 0, 0), 2)
#             cv2.putText(frame, 'Z', z_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

#             # Draw joints and ground plane with perspective projection
#             draw_joints_with_perspective(frame, this_joints[i], edges, colors[0], frame_size, distance=distance)
#             draw_joints_with_perspective(frame, this_leader_joints[i], edges, colors_l[0], frame_size, distance=distance)
            
#             out.write(frame)
            
#         print(f"[{ii + 1}/{len(joints)}] Video {video_name} saved!")
#         out.release()

#         # Add music if available
#         music_source = config.music_source if hasattr(config, 'music_source') else "../ReactDance/data_lazy/music/mp3/test/"
#         if '2023' not in dance_names[ii]:
#             if '_' in dance_names[ii]:
#                 mp3 = dance_names[ii].split('_')[0] + '_' + dance_names[ii].split('_')[1] + '_' + dance_names[ii].split('_')[2]
#                 music_name = os.path.join(music_source, mp3 + '.mp3')
#             else:
#                 music_name = os.path.join(music_source, dance_names[ii] + '.mp3')
#         else:
#             folder = dance_names[ii].split('_')[0]
#             mp3 = dance_names[ii].split('_')[1]
#             music_name = os.path.join(music_source, folder, mp3[:-2] + '0' + mp3[-2:] + '.mp3')
        
#         print(music_name, flush=True)
#         if os.path.exists(music_name):
#             video = VideoFileClip(video_name)
#             audio = AudioFileClip(music_name)
#             video_with_music = video.set_audio(audio)
#             video_with_music.write_videofile(video_name[:-4] + '_music.mp4', codec='libx264', audio_codec='aac', 
#                                             temp_audiofile=os.path.join(save_folder, 'temp-audio.m4a'), remove_temp=True)
#             os.remove(video_name)
    
#     print(f"All single videos saved!")
#     return output_videos

from moviepy.editor import ImageSequenceClip, AudioFileClip
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize3d(joints, joints_leader, config, evaldir, dance_names, quants=None):
    save_folder = os.path.join(evaldir, 'videos')
    os.makedirs(save_folder, exist_ok=True)

    joints_num = joints[0][0].shape[-1] // 3
    kinematic_tree = edges  # 确保 edges 已定义

    colors = ['blue'] * joints_num
    colors_l = ['red'] * joints_num
    output_videos = []

    def plot_xz_plane(ax, minx, maxx, miny, minz, maxz):
        verts = [[minx, miny, minz], [minx, miny, maxz], [maxx, miny, maxz], [maxx, miny, minz]]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    for ii in range(len(joints)):
        this_joints = joints[ii][0].reshape(-1, joints_num, 3)
        this_leader_joints = joints_leader[ii][0].reshape(-1, joints_num, 3)

        # Normalize and center
        MINS = np.minimum(this_joints.min(axis=0).min(axis=0), this_leader_joints.min(axis=0).min(axis=0))
        MAXS = np.maximum(this_joints.max(axis=0).max(axis=0), this_leader_joints.max(axis=0).max(axis=0))
        center = (MINS + MAXS) / 2
        scale = np.max(MAXS - MINS) * 1.5
        this_joints = (this_joints - center) / scale
        this_leader_joints = (this_leader_joints - center) / scale

        height_offset = MINS[1]
        this_joints[:, :, 1] -= height_offset
        this_leader_joints[:, :, 1] -= height_offset

        frame_number = len(this_joints)
        video_name = os.path.join(save_folder, f'{dance_names[ii]}.mp4')
        output_videos.append(video_name)

        radius = 1.5
        frame_rate = 30
        frame_size = (config.width, config.height)

        # Prepare figure once
        fig = plt.figure(figsize=(frame_size[0]/100, frame_size[1]/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        frames = []

        for i in range(frame_number):
            ax.clear()
            ax.set_xlim([-radius, radius])
            ax.set_ylim([-radius, radius])
            ax.set_zlim([0, radius * 2])
            ax.set_xlabel('X', color='r', fontsize=10)
            ax.set_ylabel('Z', color='b', fontsize=10)
            ax.set_zlabel('Y', color='g', fontsize=10)
            ax.view_init(elev=120, azim=-45)
            ax.set_box_aspect([1, 1, 1])
            ax.grid(False)
            plot_xz_plane(ax, -radius, radius, 0, -radius, radius)

            # Plot follower
            for chain, color in zip(kinematic_tree, colors):
                ax.plot(this_joints[i, chain, 0], this_joints[i, chain, 2], 
                        this_joints[i, chain, 1], linewidth=2, color=color)
            
            # Plot leader
            for chain, color in zip(kinematic_tree, colors_l):
                ax.plot(this_leader_joints[i, chain, 0], this_leader_joints[i, chain, 2], 
                        this_leader_joints[i, chain, 1], linewidth=2, color=color)

            # Render to image
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(img)

        plt.close(fig)

        print(f"[{ii + 1}/{len(joints)}] Frames generated for {video_name}")

        # Create a video clip from the frames
        clip = ImageSequenceClip(frames, fps=frame_rate)
        clip = clip.set_audio(None)

        # Write once with optional audio
        clip.write_videofile(
            video_name,
            codec='libx264',
            audio_codec='aac',
            # 建议使用绝对路径来防止临时文件问题
            temp_audiofile=os.path.abspath(os.path.join(save_folder, 'temp-audio.m4a')),
            remove_temp=True,
            # 可以添加logger='bar'来查看更详细的 ffmpeg 输出，以帮助调试
            # logger='bar'
        )
        
    return output_videos

def save_rot(smplxs, smplxs_leader, transl, transl_leader, config, evaldir, dance_names, epoch_tested):
    """
        input: 
            smplxs: list of [1xTx55*9]
            smplxs_leader: list of [1xTx55x9] 
    """
    for smplxf, smplxl in zip(smplxs, smplxs_leader):
        smplxf, smplxl = smplxf[0], smplxl[0]

    
def visualize_smplx(joints, joints_leader, config, evaldir, dance_names, epoch_tested, quants):
    pass


def mirror_visualize(joints, config, gtdir, gt_group_dir, dance_names, epoch_tested, quants):
    if len(dance_names) % 2:
        raise ValueError("dance data is not paired")
    
    videos = visualize(joints, config, gtdir, dance_names, epoch_tested, quants)
    assert len(videos) % 4 == 0
    for i in range(0, len(videos), 4):
        _concat_videos_horizontally(videos[i: i + 4], f'{gt_group_dir}')
        print(f"[{i//4 + 1}/{len(videos)//4}] Group {dance_names[i]} saved!")

def mirror_visualize2(joints_follower, joints_leader, config, gtdir, gt_group_dir, dance_names, epoch_tested):
    if len(dance_names) % 2:
        raise ValueError("dance data is not paired")
    
    videos = visualize2(joints_follower, joints_leader, config, gtdir, dance_names, epoch_tested, quants)
    assert len(videos) % 4 == 0
    for i in range(0, len(videos), 4):
        _concat_videos_horizontally(videos[i: i + 4], f'{gt_group_dir}')
        print(f"[{i//4 + 1}/{len(videos)//4}] Group {dance_names[i]} saved!")

def _add_title(frame, text, position, font_scale=1, color=(0, 255, 0), thickness=2):
    """ 在视频帧上添加文本和分割线 """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

def _concat_videos_horizontally(video_paths, output_dir, titles=None):
    dance_name = video_paths[0].split('/')[-1].split('.')[0] + '.mp4'
    output_path = os.path.join(output_dir, 'videos', dance_name)
    
    os.makedirs(os.path.join(output_dir, 'videos'), exist_ok=True)
    '''
    先列后行
    0   2
    1   3
    '''
    # video size check
    if video_paths is None:
        raise ValueError("视频列表为空")
    else:
        for cap in video_paths:
            if not os.path.exists(cap):
                raise ValueError(f"视频文件 {cap} 不存在")
           
    cap_width = 0
    cap_height = 0
    cap_min_fps = 600
    cap_object_list = []
    last_frames = [None] * len(video_paths)
    # 获取视频的宽度、高度和帧率
    for cap in video_paths:
        cap = cv2.VideoCapture(cap)
        cap_object_list.append(cap)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_min_fps = min(cap.get(cv2.CAP_PROP_FPS), cap_min_fps)
        
        if cap_width == 0:
            cap_width = width
            cap_height = height
        else:
            if height != cap_height:
                raise ValueError("视频的高度必须一致")
            
        # cap.release()
        
    # 视频的规格没有统一，停止处理
    if len(video_paths) != len(cap_object_list):
        for cap in cap_object_list:
            cap.release()
            
    num_rows = 2
    num_columns = (len(video_paths)+ num_rows - 1) // num_rows 
    
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap_min_fps, (cap_width * num_columns, cap_height * num_rows))
    title_list = [video_path.split('/')[-1] for video_path in video_paths]

    while True:
        frames = []
        all_done = True
        for i, cap in enumerate(cap_object_list):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                all_done = False
                last_frames[i] = frame  # Update last frame for this video
            else:
                frames.append(last_frames[i])  # Use the last frame if video is done
        
        if all_done:
            break
        
        # # Concatenate frames horizontally
        # combined_frame = np.hstack(frames)
        
        # Create an empty frame to place videos on
        combined_frame = np.zeros((num_rows * cap_height, num_columns * cap_width, 3), dtype=np.uint8)
        
        # Place each frame in the correct position
        for idx, frame in enumerate(frames):
            row = idx % num_rows
            col = idx // num_rows
            x_offset = col * (cap_width)
            y_offset = row * (cap_height)
            combined_frame[y_offset:y_offset + cap_height, x_offset:x_offset + cap_width] = frame
        
        # Add titles and split lines
        line_thickness = 2
        line_color = (255, 255, 255)  # White line
        
        if titles is not None:
            assert len(titles) == len(video_paths)
            title_list = titles
        else:
            title_list = [os.path.basename(path) for path in video_paths]
        
        # Add lines
        for i in range(len(video_paths) - 1):
            line_position = (i + 1) * cap_width  # Adding 10 pixels between videos
            cv2.line(combined_frame, (line_position, 0), (line_position, cap_height * num_rows), line_color, line_thickness)
        cv2.line(combined_frame, (0, cap_height), (cap_width * num_columns, cap_height), line_color, line_thickness)
        
        # Add titles at the bottom of each video
        for idx, title in enumerate(title_list):
            row = idx % num_rows + 1
            col = idx // num_rows
            _add_title(combined_frame, title, (col * cap_width + cap_width // 4, row * cap_height - 40), font_scale=0.75)
        
        # Write the frame to the output video
        out.write(combined_frame)
    
    # Release resources
    for cap in cap_object_list:
        cap.release()
    out.release()
    
    # if find music
    music_source = "../ReactDance/data_lazy/music/mp3/test/"
    if '2023' not in dance_name:
        if '_' in dance_name:
            mp3 = dance_name.split('_')[0] + '_' + dance_name.split('_')[1] + '_' +  dance_name.split('_')[2]
            music_name = os.path.join(music_source, mp3 + '.mp3')
        else:
            music_name = os.path.join(music_source, dance_name + '.mp3')
    else:
        folder = dance_name.split('_')[0]
        mp3 = dance_name.split('_')[1]
        music_name = os.path.join(music_source, folder, mp3[:-2] + '0' + mp3[-2:] + '.mp3')
    
    # print(music_name, flush=True)
    if os.path.exists(music_name):
        video = VideoFileClip(output_path)
        audio = AudioFileClip(music_name)
        # bg_music = audio.set_duration(video.duration)
        video_with_music = video.set_audio(audio)
        video_with_music.write_videofile(output_path[:-4] + '_a.mp4', codec='libx264', audio_codec='aac', temp_audiofile=output_path[:-4] + 'temp-audio.m4a', remove_temp=True)
        os.remove(output_path)
            
    print(f"合并视频保存到 {output_path}")

if __name__ == '__main__':
    output_path = 'data/mirror/motion/Ballet_009_001_00-Lumba_005_001_00.mp4'
    video_paths = ['experiments/motion_vqvae/eval/videos/ep0500/Ballet_009_001_00.npy.mp4',
                   'experiments/motion_vqvae/eval/videos/ep0500/Lumba_005_001_00.npy.mp4',
                   'experiments/motion_vqvae/eval/videos/ep0500/Waltz_005_002_01.npy.mp4',
                   'experiments/motion_vqvae/eval/videos/ep0500/Tango_005_023_00.npy.mp4',
                   'experiments/motion_vqvae/eval/videos/ep0500/Samba_005_002_00.npy.mp4'
                   ]
    _concat_videos_horizontally(video_paths, output_path)