#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import os
from std_msgs.msg import Int32, Float32
from geometry_msgs.msg import Point32, PoseStamped
from sensor_msgs.msg import Imu
from beginner_tutorials.msg import TrajectoryArray, TrajectoryPoint, TrackingArray, TrackingPoint  # 실제 메시지 파일 경로를 맞게 수정
from visualization_msgs.msg import Marker, MarkerArray
import tf
import time
import csv
from scipy.spatial import distance, KDTree
from math import pi

# 폴더 내의 ego_tracking_data_*.csv 파일 개수를 확인하고 index를 설정
data_folder = '/home/taewook/catkin_ws/src/beginner_tutorials/data/tracking'
existing_files = [f for f in os.listdir(data_folder) if f.startswith('ego_tracking_data_') and f.endswith('.csv')]
index = len(existing_files) + 1

class DataTrackingLogger:
    def __init__(self):
        # CSV 파일 생성
        self.file_name = f'/home/taewook/catkin_ws/src/beginner_tutorials/data/tracking/ego_tracking_data_{index}.csv'
        with open(self.file_name, 'w', newline='') as csvfile:
            self.csvwriter = csv.writer(csvfile)
            #self.csvwriter.writerow(['time', 'x', 'y', 'vx', 'vy', 'x_lane1', 'x_lane2', 'x_lane3'])  # 헤더 작성
        
        self.path = '/home/taewook/catkin_ws/src/beginner_tutorials/path/'
        self.lane_files = ['lane1.txt', 'lane2.txt', 'lane3.txt']

        # 시간 관리 변수
        self.last_logged_time = None
        self.dt = 0.1  # [s], 기록할 시간 간격
        
        # ROS 구독자 설정
        self.subscriber = rospy.Subscriber('/ego_tracking', TrackingPoint, self.vehicle_callback)
        
        self.time = None
        self.x = None
        self.y = None
        self.vx = None
        self.vy = None   
        self.yaw = None

        # 차선 데이터 (x값)
        self.lane_data = [None, None, None]

        # 각 차선 데이터 로드
        self.lane_data_points = [self.load_lane_data(f) for f in self.lane_files]  # 경로와 파일명을 합쳐서 파일 로드


    def vehicle_callback(self, msg):
        #rospy.loginfo(f"Received tracking message: time={msg.time}, x={msg.x}, y={msg.y}")
        # TrackingPoint 메시지에서 필요한 정보 추출
        self.x = msg.x
        self.y = msg.y
        self.vx = msg.vx
        self.vy = msg.vy
        self.yaw = msg.yaw * np.pi / 180 # [rad]
        self.time = msg.time

    def load_lane_data(self, file_name):
        # 텍스트 파일에서 lane 데이터를 읽어옴
        lane_points = []
        full_path = self.path + file_name  # 경로와 파일명 결합
        with open(full_path, 'r') as file:
            reader = csv.reader(file, delimiter=' ')
            for row in reader:
                x, y, z = map(float, row)
                lane_points.append([x, y, z])
        return np.array(lane_points)  # NumPy 배열로 변환하여 반환

    
    def find_closest_point(self, lane, ego_x, ego_y):
        # KDTree를 사용해 가장 가까운 점을 빠르게 찾음
        tree = KDTree(lane[:, :2])  # (x, y) 좌표로 KDTree 생성
        dist, closest_index = tree.query([ego_x, ego_y])  # 가장 가까운 점 검색
        return lane[closest_index]  # 가장 가까운 점의 좌표 반환 (x, y, z)

    def run(self):
        # self.time이 None이면 실행하지 않고 대기
        if self.time is None:
            rospy.logwarn("Waiting for the first message to arrive...")
            return

        # ROS 메시지에서 시간 정보를 msg.time에서 가져옴 (초 단위)
        current_time = self.time.secs + self.time.nsecs * 1e-9

        # 0.1초 단위로 저장
        if self.last_logged_time is None or current_time - self.last_logged_time >= self.dt:  # 0.1초
            with open(self.file_name, 'a', newline='') as csvfile:
                self.csvwriter = csv.writer(csvfile)

                # 각 차선에서 가장 가까운 좌표 찾기
                for i, lane in enumerate(self.lane_data_points):
                    closest_point = self.find_closest_point(lane, self.x, self.y)
                    self.lane_data[i] = closest_point[0]  # x 좌표만 저장

                # CSV에 기록
                self.csvwriter.writerow([current_time, self.x, self.y, np.sqrt(self.vx**2 + self.vy**2), self.yaw, self.lane_data[0], self.lane_data[1], self.lane_data[2]])

            self.last_logged_time = current_time

if __name__ == '__main__':
    rospy.init_node('ego_tracking_logger', anonymous=True)
    logger = DataTrackingLogger()  # 클래스의 인스턴스를 생성

    # 첫 메시지가 도착할 때까지 대기
    rospy.loginfo("Waiting for the first message from /ego_tracking...")
    while not rospy.is_shutdown() and logger.time is None:
        rospy.sleep(0.01)  # 10ms 대기 후 다시 확인

    rospy.loginfo("First message received. Starting to log data.")

    try:
        while not rospy.is_shutdown():
            logger.run()  # 가능한 한 빠르게 logger.run() 실행
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down logger node")
