BASE = ../../..

PLIB = 1
ROS = 1
PCL = 1
OPENCV = 1
DEPEND = Core RosCom Control Gui Hardware_gamepad Kin Motion Geo Roopi Perception PCL

LIBS += -lcv_bridge
OBJS = main.o stereo.o sc_dis.o basictask.o vision.o shape.o kmean.o
include $(BASE)/build/generic.mk
