## 导入第三方库
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys


## 读取关节数据
def read_skeleton(file):
    with open(file, 'r') as f:  # 打开file(.skeleton)文件
        skeleton_sequence = {}  # 初始化skeleton_sequence
        skeleton_sequence['numFrame'] = int(f.readline())  # 读取.skeleton文件第一行，即帧数
        skeleton_sequence['frameInfo'] = []

        for t in range(skeleton_sequence['numFrame']):  # 遍历每一帧
            frame_info = {}  # 初始化frame_info
            frame_info['numBody'] = int(f.readline())  # 再次调用.readline函数，读取.skeleton文件的下一行，即body数
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):  # 遍历每一个body
                body_info = {}  # 初始化body_info
                body_info_key = [  # key: 数字表示的意义，即对应的key
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)  # 字典类型; key: value(float类型)
                    for k, v in zip(body_info_key, f.readline().split())  # 读取下一行数据，根据key打包数据，遍历返回key, value
                }

                body_info['numJoint'] = int(f.readline())  # 读取下一行数据，即关节数
                body_info['jointInfo'] = []

                for v in range(body_info['numJoint']):  # 遍历25个关节的数据
                    joint_info_key = [  # Key: 数字表示的意义，即对应的key
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)  # 字典类型; key: value(float类型)
                        for k, v in zip(joint_info_key, f.readline().split())  # 读取下一行数据，根据key打包数据，遍历返回key, value
                    }
                    body_info['jointInfo'].append(joint_info)  # 保存关节数据

                frame_info['bodyInfo'].append(body_info)  # 保存body数据
            skeleton_sequence['frameInfo'].append(frame_info)  # 保存当前帧的数据
    return skeleton_sequence


## 读取关节的x，y，z三个坐标
def read_xyz(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)  # 调用read_skeleton()函数读取.skeleton文件的数据

    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))  # 初始化数据； 3 × 帧数 × 25 × max_body
    for n, f in enumerate(seq_info['frameInfo']):  # 遍历每一帧的数据
        for m, b in enumerate(f['bodyInfo']):  # 遍历每一个body的数据
            for j, v in enumerate(b['jointInfo']):  # 遍历每一个关节的数据
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]  # 保存 x,y,z三个坐标的数据
                else:
                    pass
    return data


## 2D展示
def Print2D(num_frame, point, arms, rightHand, leftHand, legs, body):
    # 求坐标最大值
    xmax = np.max(point[0, :, :, :])
    xmin = np.min(point[0, :, :, :])
    ymax = np.max(point[1, :, :, :])
    ymin = np.min(point[1, :, :, :])
    zmax = np.max(point[2, :, :, :])
    zmin = np.min(point[2, :, :, :])

    n = 0  # 从第n帧开始展示
    m = num_frame  # 到第m帧结束，n<m<row
    plt.figure()
    plt.ion()
    for i in range(n, m):
        plt.cla()  # # Clear axis, 即清除当前图形中的当前活动轴, 其他轴不受影响

        # 画出两个body所有关节
        plt.scatter(point[0, i, :, :], point[1, i, :, :], c='green', s=2)  # c: 颜色;  s: 大小

        # 连接第一个body的关节，形成骨骼
        plt.plot(point[0, i, arms, 0], point[1, i, arms, 0], c='green', lw=2.0)
        plt.plot(point[0, i, rightHand, 0], point[1, i, rightHand, 0], c='green', lw=2.0)  # c: 颜色;  lw: 线条宽度
        plt.plot(point[0, i, leftHand, 0], point[1, i, leftHand, 0], c='green', lw=2.0)
        plt.plot(point[0, i, legs, 0], point[1, i, legs, 0], c='green', lw=2.0)
        plt.plot(point[0, i, body, 0], point[1, i, body, 0], c='green', lw=2.0)

        # 连接第二个body的关节，形成骨骼
        plt.plot(point[0, i, arms, 1], point[1, i, arms, 1], c='green', lw=2.0)
        plt.plot(point[0, i, rightHand, 1], point[1, i, rightHand, 1], c='green', lw=2.0)
        plt.plot(point[0, i, leftHand, 1], point[1, i, leftHand, 1], c='green', lw=2.0)
        plt.plot(point[0, i, legs, 1], point[1, i, legs, 1], c='green', lw=2.0)
        plt.plot(point[0, i, body, 1], point[1, i, body, 1], c='green', lw=2.0)

        plt.text(xmax, ymax + 0.2, 'frame: {}/{}'.format(i, num_frame - 1))  # 文字说明
        plt.xlim(xmin - 0.5, xmax + 0.5)  # x坐标范围
        plt.ylim(ymin - 0.3, ymax + 0.3)  # y坐标范围
        plt.pause(0.001)  # 停顿延时

    plt.ioff()
    plt.show()


## 3D展示
def Print3D(num_frame, point, arms, rightHand, leftHand, legs, body):
    # 求坐标最大值
    xmax = np.max(point[0, :, :, :])
    xmin = np.min(point[0, :, :, :])
    ymax = np.max(point[1, :, :, :])
    ymin = np.min(point[1, :, :, :])
    zmax = np.max(point[2, :, :, :])
    zmin = np.min(point[2, :, :, :])

    n = 0  # 从第n帧开始展示
    m = num_frame  # 到第m帧结束，n<m<row
    plt.figure()
    plt.ion()
    for i in range(n, m):
        plt.cla()  # Clear axis, 即清除当前图形中的当前活动轴, 其他轴不受影响

        plot3D = plt.subplot(projection='3d')
        plot3D.view_init(120, -90)  # 改变视角

        Expan_Multiple = 1.4  # 坐标扩大倍数，绘图较美观

        # 画出两个body所有关节
        plot3D.scatter(point[0, i, :, :] * Expan_Multiple, point[1, i, :, :] * Expan_Multiple, point[2, i, :, :],
                       c='red', s=40.0)  # c: 颜色;  s: 大小

        # 连接第一个body的关节，形成骨骼
        plot3D.plot(point[0, i, arms, 0] * Expan_Multiple, point[1, i, arms, 0] * Expan_Multiple, point[2, i, arms, 0],
                    c='green', lw=2.0)
        plot3D.plot(point[0, i, rightHand, 0] * Expan_Multiple, point[1, i, rightHand, 0] * Expan_Multiple,
                    point[2, i, rightHand, 0], c='green', lw=2.0)  # c: 颜色;  lw: 线条宽度
        plot3D.plot(point[0, i, leftHand, 0] * Expan_Multiple, point[1, i, leftHand, 0] * Expan_Multiple,
                    point[2, i, leftHand, 0], c='green', lw=2.0)
        plot3D.plot(point[0, i, legs, 0] * Expan_Multiple, point[1, i, legs, 0] * Expan_Multiple, point[2, i, legs, 0],
                    c='green', lw=2.0)
        plot3D.plot(point[0, i, body, 0] * Expan_Multiple, point[1, i, body, 0] * Expan_Multiple, point[2, i, body, 0],
                    c='green', lw=2.0)

        # 连接第二个body的关节，形成骨骼
        plot3D.plot(point[0, i, arms, 1] * Expan_Multiple, point[1, i, arms, 1] * Expan_Multiple, point[2, i, arms, 1],
                    c='green', lw=2.0)
        plot3D.plot(point[0, i, rightHand, 1] * Expan_Multiple, point[1, i, rightHand, 1] * Expan_Multiple,
                    point[2, i, rightHand, 1], c='green', lw=2.0)
        plot3D.plot(point[0, i, leftHand, 1] * Expan_Multiple, point[1, i, leftHand, 1] * Expan_Multiple,
                    point[2, i, leftHand, 1], c='green', lw=2.0)
        plot3D.plot(point[0, i, legs, 1] * Expan_Multiple, point[1, i, legs, 1] * Expan_Multiple, point[2, i, legs, 1],
                    c='green', lw=2.0)
        plot3D.plot(point[0, i, body, 1] * Expan_Multiple, point[1, i, body, 1] * Expan_Multiple, point[2, i, body, 1],
                    c='green', lw=2.0)

        plot3D.text(xmax - 0.3, ymax + 1.1, zmax + 0.3, 'frame: {}/{}'.format(i, num_frame - 1))  # 文字说明
        plot3D.set_xlim3d(xmin - 0.5, xmax + 0.5)  # x坐标范围
        plot3D.set_ylim3d(ymin - 0.3, ymax + 0.3)  # y坐标范围
        plot3D.set_zlim3d(zmin - 0.3, zmax + 0.3)  # z坐标范围
        plt.pause(0.001)  # 停顿延时

    plt.ioff()
    plt.show()


## main函数
def main():
    sys.path.extend(['D:/ZC/skeletondate NTU120/'])  # 扩展路径
    # data_path = 'D:/ZC/skeletondate NTU120/S010C001P019R001A010.skeleton'  # 拍手skeleton文件名
    data_path = 'D:/ZC/skeletondate NTU120/S001C001P002R002A001.skeleton' # 握手skeleton文件名，two body
    point = read_xyz(data_path)  # 读取 x,y,z三个坐标
    print('Read Data Done!')  # 数据读取完毕

    num_frame = point.shape[1]  # 帧数
    print(point.shape)  # 坐标数(3) × 帧数 × 关节数(25) × max_body(2)

    # 相邻关节标号
    arms = [23, 11, 10, 9, 8, 20, 4, 5, 6, 7, 21]  # 23 <-> 11 <-> 10 ...
    rightHand = [11, 24]  # 11 <-> 24
    leftHand = [7, 22]  # 7 <-> 22
    legs = [19, 18, 17, 16, 0, 12, 13, 14, 15]  # 19 <-> 18 <-> 17 ...
    body = [3, 2, 20, 1, 0]  # 3 <-> 2 <-> 20 ...

    Print2D(num_frame, point, arms, rightHand, leftHand, legs, body)  # 2D可视化
    # Print3D(num_frame, point, arms, rightHand, leftHand, legs, body) # 3D可视化


main()