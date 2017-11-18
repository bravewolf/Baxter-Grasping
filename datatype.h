#ifndef DATATYPE
#define DATATYPE
#include <Roopi/roopi.h>
#include "opencv2/core.hpp"

///grid index
struct Index{
    int x;
    int y;
    int z;
};

///id,position,x axis,y axis,z axis of marker in world frame
struct MarkerData{
    int id;
    mlr::Vector pos;
    mlr::Vector x;
    mlr::Vector y;
    mlr::Vector z;
    cv::Point2f center;
};



#endif // DATATYPE

