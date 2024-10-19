#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import time
import rospy
import rospkg
from math import cos,sin,pi ,sqrt,pow,atan2
from std_msgs.msg import Int32, Float32
from geometry_msgs.msg import Point,PoseWithCovarianceStamped
from nav_msgs.msg import Odometry,Path
from morai_msgs.msg import CtrlCmd, EgoVehicleStatus
import numpy as np
import tf
from tf.transformations import euler_from_quaternion,quaternion_from_euler

# advanced_purepursuit 은 차량의 차량의 종 횡 방향 제어 예제입니다.
# Purpusuit 알고리즘의 Look Ahead Distance 값을 속도에 비례하여 가변 값으로 만들어 횡 방향 주행 성능을 올립니다.
# 횡방향 제어 입력은 주행할 Local Path (지역경로) 와 차량의 상태 정보 Odometry 를 받아 차량을 제어 합니다.
# 종방향 제어 입력은 목표 속도를 지정 한뒤 목표 속도에 도달하기 위한 Throttle control 을 합니다.
# 종방향 제어 입력은 longlCmdType 1(Throttle control) 이용합니다.

# 노드 실행 순서 
# 1. subscriber, publisher 선언
# 2. 속도 비례 Look Ahead Distance 값 설정
# 3. 좌표 변환 행렬 생성
# 4. Steering 각도 계산
# 5. PID 제어 생성
# 6. 도로의 곡률 계산
# 7. 곡률 기반 속도 계획
# 8. 제어입력 메세지 Publish

SAFE = 102
DECEL = 101
AEB = 100

class pure_pursuit :
    def __init__(self):
        rospy.init_node('pure_pursuit', anonymous=True)

        #TODO: (1) subscriber, publisher 선언
        #rospy.Subscriber("/global_path", Path, self.global_path_callback)
        
        #rospy.Subscriber("/lattice_path", Path, self.path_callback)
        rospy.Subscriber("/local_path",Path,self.path_callback)

        rospy.Subscriber("/min_collision_time", Float32, self.min_collision_time_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/Ego_topic",EgoVehicleStatus, self.status_callback)
        rospy.Subscriber("/distance_to_ego",Float32,self.distance_to_ego_callback)

        self.ctrl_cmd_pub = rospy.Publisher('ctrl_cmd',CtrlCmd, queue_size=1)

        self.ctrl_cmd_msg = CtrlCmd()
        self.ctrl_cmd_msg.longlCmdType = 2

        self.is_path = False
        self.is_odom = False 
        self.is_status = False
        #self.is_global_path = False
        self.is_TTC = False
        self.is_distance_to_ego = False

        self.is_look_forward_point = False

        self.forward_point = Point()
        self.current_postion = Point()

        self.vehicle_length = 2.84 #차체 길이 (임시 지정)
        self.lfd = 15
        if self.vehicle_length is None or self.lfd is None:
            print("you need to change values at line 57~58 ,  self.vegicle_length , lfd")
            exit()
        self.min_lfd = 10
        self.max_lfd = 30
        self.lfd_gain =  1.2 #0.78
        self.target_velocity = 30 # [km/h] Target 속도

        self.TTC = 999
        self.distance_to_ego = 999999
        self.max_deceleration = -8
        self.TTC_warning = 2.7
        self.TTC_AEB = 100

        self.system_status = SAFE

        self.AEB_counter = 0

        self.pid = pidControl()
        # self.vel_planning = velocityPlanning(self.target_velocity/3.6, 0.15)
        
        # while True:
        #     if self.is_global_path == True:
        #         self.velocity_list = self.vel_planning.curvedBaseVelocity(self.global_path, 50)
        #         break
        #     else:
        #         rospy.loginfo('Waiting global path data')

        rate = rospy.Rate(20) # 30hz
        while not rospy.is_shutdown():
            #rospy.loginfo(self.is_TTC)
            if self.is_path == True and self.is_odom == True and self.is_status == True:
                prev_time = time.time()

                # 스티어링 제어
                #self.current_waypoint = self.get_current_waypoint(self.status_msg,self.global_path)
                #self.target_velocity = self.velocity_list[self.current_waypoint]*3.6
                

                steering = self.calc_pure_pursuit()
                if self.is_look_forward_point :
                    self.ctrl_cmd_msg.steering = steering
                else : 
                    rospy.loginfo("no found forward point")
                    self.ctrl_cmd_msg.steering = 0.0
                
                # 상태 머신
                # if not self.is_TTC: # 상대 차량이 사라졌을 때 TTC 초기화
                #     self.TTC = 999

                self.TTC_AEB = min(self.status_msg.velocity.x / (abs(self.max_deceleration)-1), 1.5)  # velocity는 상대속도로 바꿔야 함
                self.system_status = self.state_machine(self.system_status, self.status_msg.velocity.x, self.TTC, self.TTC_warning, self.TTC_AEB)

                # 상태머신에 따른 속도 제어
                if self.system_status == SAFE:
                    self.ctrl_cmd_msg.longlCmdType = 2
                    self.ctrl_cmd_msg.velocity = self.target_velocity # km/h로 들어감
                
                elif self.system_status == DECEL:
                    self.ctrl_cmd_msg.longlCmdType = 3
                    self.ctrl_cmd_msg.acceleration = -4
                    #acc_cmd = self.adaptive_cruise_control(self.status_msg.velocity.x, self.target_velocity, 1.0)  # 1초 간격 유지
                    #self.ctrl_cmd_msg.acceleration = acc_cmd
                    
                
                elif self.system_status == AEB:
                    self.ctrl_cmd_msg.longlCmdType = 3
                    self.ctrl_cmd_msg.acceleration = self.max_deceleration

                #output = self.pid.pid(self.target_velocity,self.status_msg.velocity.x*3.6)

                # if output > 0.0:
                #     self.ctrl_cmd_msg.accel = output
                #     self.ctrl_cmd_msg.brake = 0.0
                # else:
                #     self.ctrl_cmd_msg.accel = 0.0
                #     self.ctrl_cmd_msg.brake = -output

                #TODO: (8) 제어입력 메세지 Publish
                print("Type = {}".format(self.ctrl_cmd_msg.longlCmdType))
                print("current status: {}".format(self.system_status))
                rospy.loginfo("TTC: {}".format(self.TTC))
                #print("TTC: {}".format(self.TTC))
                print("Distance: {}".format(self.distance_to_ego))
                #print("steering: {}".format(steering))
                
                self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)
                
            rate.sleep()

    def adaptive_cruise_control(self, ego_velocity, target_velocity, time_gap):
        """ACC 알고리즘을 통해 가속도를 계산"""
        # 시간 간격을 고려한 속도 차이
        distance_error = time_gap * target_velocity - ego_velocity
        # PID 제어기를 이용해 가속도를 계산
        acc_cmd = self.pid.pid(target_velocity, ego_velocity)
        return acc_cmd
    
    def path_callback(self,msg):
        self.is_path=True
        self.path=msg  

    def odom_callback(self,msg):
        self.is_odom=True
        odom_quaternion=(msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w)
        _,_,self.vehicle_yaw=euler_from_quaternion(odom_quaternion)
        self.current_postion.x=msg.pose.pose.position.x
        self.current_postion.y=msg.pose.pose.position.y

    def status_callback(self,msg): ## Vehicle Status Subscriber 
        self.is_status=True
        self.status_msg=msg
    
    def distance_to_ego_callback(self, msg):
        if msg.data:
            self.is_distance_to_ego = True
            self.distance_to_ego = msg.data  # msg.data로 접근
        else:
            self.distance_to_ego = 999999
            self.is_distance_to_ego = False
        
    # def global_path_callback(self,msg):
    #     self.global_path = msg
    #     self.is_global_path = True

    def min_collision_time_callback(self, msg):
    # 콜백 값이 정상적으로 들어왔을 때만 값 업데이트
        self.TTC = msg.data
        self.is_TTC = True
        # if msg.data:  # msg에 값이 있을 경우
        #     self.TTC = msg.data
        #     self.is_TTC = True
        # else:
        #     # 값이 없으면 기본값 999로 설정
        #     self.TTC = 999
        #     self.is_TTC = False
    
    # def get_current_waypoint(self,ego_status,global_path):
    #     min_dist = float('inf')        
    #     currnet_waypoint = -1
    #     for i,pose in enumerate(global_path.poses):
    #         dx = ego_status.position.x - pose.pose.position.x
    #         dy = ego_status.position.y - pose.pose.position.y

    #         dist = sqrt(pow(dx,2)+pow(dy,2))
    #         if min_dist > dist :
    #             min_dist = dist
    #             currnet_waypoint = i
    #     return currnet_waypoint

    def calc_pure_pursuit(self,):

        #TODO: (2) 속도 비례 Look Ahead Distance 값 설정
        self.lfd = (self.status_msg.velocity.x) * self.lfd_gain
        
        if self.lfd < self.min_lfd : 
            self.lfd=self.min_lfd
        elif self.lfd > self.max_lfd :
            self.lfd=self.max_lfd
        rospy.loginfo(self.lfd)
        
        vehicle_position=self.current_postion
        self.is_look_forward_point= False

        translation = [vehicle_position.x, vehicle_position.y]

        #TODO: (3) 좌표 변환 행렬 생성
        trans_matrix = np.array([
                [cos(self.vehicle_yaw), -sin(self.vehicle_yaw),translation[0]],
                [sin(self.vehicle_yaw),cos(self.vehicle_yaw),translation[1]],
                [0                    ,0                    ,1            ]])

        det_trans_matrix = np.linalg.inv(trans_matrix)

        for num,i in enumerate(self.path.poses) :
            path_point=i.pose.position

            global_path_point = [path_point.x,path_point.y,1]
            local_path_point = det_trans_matrix.dot(global_path_point)    

            if local_path_point[0]>0 :
                dis = sqrt(pow(local_path_point[0],2)+pow(local_path_point[1],2))
                if dis >= self.lfd :
                    self.forward_point = path_point
                    self.is_look_forward_point = True
                    break
        
        #TODO: (4) Steering 각도 계산
        theta = atan2(local_path_point[1],local_path_point[0])
        steering = atan2((2*self.vehicle_length*sin(theta)),self.lfd)

        return steering

    def state_machine(self, current_status, ego_vx, TTC, TTC_warning, TTC_AEB):
        if TTC < TTC_AEB and (current_status == SAFE or DECEL):
            return AEB
        
        elif TTC < TTC_warning and (current_status == SAFE):
            return DECEL
        
        elif TTC > TTC_warning*1.5 and self.distance_to_ego > 15 and current_status == DECEL: # TTC_warning보다 기준이 커야함, 전방에 이륜차가 천천히 간다면 그 속도에 맞춰야 하기 때문에,
            return SAFE

        elif TTC > TTC_warning and ego_vx < 1 and self.distance_to_ego > 6 and current_status == AEB:
            if self.AEB_counter >= 30*1: # 30Hz *1 초
                self.AEB_counter = 0
                return SAFE
            else:
                self.AEB_counter+=1
                return AEB
                
        else:
            return current_status 

class pidControl:
    def __init__(self):
        self.p_gain = 0.3
        self.i_gain = 0.0001
        self.d_gain = 0.1
        self.prev_error = 0
        self.i_control = 0
        self.controlTime = 0.05

    def pid(self,target_vel, current_vel):
        error = target_vel - current_vel

        #TODO: (5) PID 제어 생성
        p_control = self.p_gain * error
        self.i_control += self.i_gain * error * self.controlTime
        d_control = self.d_gain * (error-self.prev_error) / self.controlTime

        output = p_control + self.i_control + d_control
        self.prev_error = error

        return output

# class velocityPlanning:
#     def __init__ (self,car_max_speed, road_friciton):
#         self.car_max_speed = car_max_speed
#         self.road_friction = road_friciton

#     def curvedBaseVelocity(self, gloabl_path, point_num):
#         out_vel_plan = []

#         for i in range(0,point_num):
#             out_vel_plan.append(self.car_max_speed)

#         for i in range(point_num, len(gloabl_path.poses) - point_num):
#             x_list = []
#             y_list = []
#             for box in range(-point_num, point_num):
#                 x = gloabl_path.poses[i+box].pose.position.x
#                 y = gloabl_path.poses[i+box].pose.position.y
#                 x_list.append([-2*x, -2*y ,1])
#                 y_list.append((-x*x) - (y*y))

#             #TODO: (6) 도로의 곡률 계산
#             x_matrix = np.array(x_list)
#             y_matrix = np.array(y_list)
#             x_trans = x_matrix.T

#             a_matrix = np.linalg.inv(x_trans.dot(x_matrix)).dot(x_trans).dot(y_matrix)
#             a = a_matrix[0]
#             b = a_matrix[1]
#             c = a_matrix[2]
#             r = sqrt(a*a+b*b-c)

#             #TODO: (7) 곡률 기반 속도 계획
#             v_max = sqrt(r*9.8*self.road_friction)

#             if v_max > self.car_max_speed:
#                 v_max = self.car_max_speed
#             out_vel_plan.append(v_max)

#         for i in range(len(gloabl_path.poses) - point_num, len(gloabl_path.poses)-10):
#             out_vel_plan.append(30)

#         for i in range(len(gloabl_path.poses) - 10, len(gloabl_path.poses)):
#             out_vel_plan.append(0)

#         return out_vel_plan

if __name__ == '__main__':
    try:
        test_track=pure_pursuit()
    except rospy.ROSInterruptException:
        pass
