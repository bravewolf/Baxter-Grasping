#include "stereo.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/stereo.hpp"
#include "opencv2/shape.hpp"
#include "opencv2/ximgproc.hpp"
#include "sc_dis.hpp"
#include <iostream>
using namespace cv;
using namespace std;
using namespace cv::stereo;

namespace grab_stereo {

/**
 * @brief disparityMap, generate disparity map
 * @param imgLeft
 * @param imgRight
 * @param right,flag to use right matcher
 * @return
 */
Mat disparityMap(Mat imgLeft,Mat imgRight,bool right){
    cv::cvtColor(imgLeft, imgLeft, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgRight, imgRight, cv::COLOR_BGR2GRAY);

    imgLeft.convertTo(imgLeft,CV_8UC1);
    imgRight.convertTo(imgRight,CV_8UC1);

    if( imgLeft.empty() || imgRight.empty() )
    { std::cout<< " --(!) Error reading images " << std::endl; }
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,64,11);
    sgbm->setP1(8*3*11*11);
    sgbm->setP2(32*3*11*11);

    //-- 3. Calculate the disparity image
    Mat dist;
    if(!right)
        sgbm->compute(imgLeft, imgRight, dist );
    else
        cv::ximgproc::createRightMatcher(sgbm)->compute(imgRight,imgLeft,dist);

    //-- Check its extreme values
//    double minVal; double maxVal;
//    minMaxLoc( dist/16, &minVal, &maxVal );
//    printf("Min disp: %f Max value: %f \n", minVal, maxVal);
    return dist/16;
}

/**
 * @brief stereoMatching, generate depth map from disparity map
 * @param disparity
 * @param camMat, camera matrix
 * @param distCoeffs, distortion coeffcient
 * @param size
 * @param R
 * @param T
 * @return
 */
Mat stereoMatching(Mat disparity,Mat camMat,Mat distCoeffs,Size size,Mat R,Mat T){
    R.convertTo(R,CV_64F);
    T.convertTo(T,CV_64F);
    Mat R1,R2,P1,P2,Q,_3dImage;
    stereoRectify(camMat,distCoeffs,camMat,distCoeffs,size,R,T,R1,R2,P1,P2,Q);
    reprojectImageTo3D(disparity,_3dImage,Q,true);
    return _3dImage;
}

}
