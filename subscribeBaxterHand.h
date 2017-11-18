#pragma once
#include <RosCom/roscom.h>
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>

// Using our templates
#if 0
struct SubscribeBaxterHand{
  Access<cv::Mat> baxter_left_camera;
  SubscriberConv<sensor_msgs::Image, byteA, &conv_image2byteA> subImage;

  SubscribeBaxterHand()
    : baxter_left_camera(NULL, "baxter_left_camera"),
      subImage("/cameras/left_hand_camera/image", baxter_left_camera){
  }
  ~SubscribeBaxterHand(){
  }
};

#else
struct SubscribeBaxterHand{
  Access<cv::Mat> access;
  ros::NodeHandle *nh;
  ros::Subscriber sub;

  SubscribeBaxterHand():
    access(NULL, "baxter_left_camera"), nh(NULL) {
    nh = new ros::NodeHandle;
    cout << "subscribing to /cameras/left_hand_camera/image into baxter_left_camera access" << endl;
    sub = nh->subscribe("/cameras/left_hand_camera/image", 1, &SubscribeBaxterHand::callback, this);
  }

  void callback(const sensor_msgs::Image& msg){
    access.writeAccess();
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    access() = cv_ptr->image;
    access.deAccess();
  }
  ~SubscribeBaxterHand(){
    if(nh) delete nh;
  }
};

#endif
