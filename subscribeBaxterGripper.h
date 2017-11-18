#pragma once
#include <RosCom/roscom.h>
#include <ros/ros.h>
#include "baxter_core_msgs/EndEffectorState.h"

///subscribe to baxter's left gripper
/// get force data
struct SubscribeBaxterGripper{
  Access<double> access;
  ros::NodeHandle *nh;
  ros::Subscriber sub;

  SubscribeBaxterGripper():
    access(NULL, "baxter_left_gripper"), nh(NULL) {
    nh = new ros::NodeHandle;
    cout << "subscribing to /robot/end_effector/left_gripper/state into baxter_left_gripper access" << endl;
    sub = nh->subscribe("/robot/end_effector/left_gripper/state", 100, &SubscribeBaxterGripper::callback, this);
  }

  void callback(const baxter_core_msgs::EndEffectorState state){
    access.writeAccess();
    access() = state.force;
    access.deAccess();
  }

  ~SubscribeBaxterGripper(){
    if(nh) delete nh;
  }
};

