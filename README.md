# Baxter-Grasping
This is code for Practical Robotics course hold by Uni Stuttgart
Solve robotic grasping of single object in 2D plane
Apply reinforcement learning, supervised learning, computer vision

Use SVN to learn the grasping point by shape-context feature, standford grasping data
Tabular Q learning, LSPI to update the policy
Camera calibration, ARUCO marker, SIFT


Compilation:

1.make sure opencv 3.2 and opencv_contrib is installed, we use ARUCO marker,SIFT 

2.set test grasping condition to ‘if(1) ...’(first test in main) in main function in main.cpp
	
3.type 'make' in command line to compile


Run: (THIS PART REQUIRED ROS AND CODEBASE PROVIDED BY UNI STUTTGART)


1.modify MT.cfg, set 'useRos = 1'

2.type 'source ../../../bin/baxterlansetup.sh' or 'source ../../../bin/baxterwlansetup.sh' to connect to Baxter

3.type 'source ../../../bin/baxterStart.sh'(or type 'rosrun baxter_tool tuck.py -u' if baxter already start)

4.after Baxter start, type './x.exe' to run 

