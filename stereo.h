#ifndef STEREO
#define STEREO
#include "opencv2/core/utility.hpp"
using namespace cv;

void shapeContextDescriptor(std::vector<Point> contour);

namespace grab_stereo {

///compute disparity map from imageL and imageR
Mat disparityMap(Mat imgL,Mat imgR,bool right=false);

///compute depth map
Mat stereoMatching(Mat disparity,Mat camMat,Mat distCoeffs,Size size,Mat R,Mat T);

}


#endif // STEREOMATCHING

