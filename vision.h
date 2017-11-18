#ifndef VISION
#define VISION
#include <Roopi/roopi.h>
#include "opencv2/core.hpp"
#include "datatype.h"
#include <string>
using namespace std;
using namespace cv;
namespace grab_vision{

///convert cv mat to mlr vector
mlr::Vector mat2vector(cv::Mat mat);

///record video from webcam
void recordVideo(string filename);

///cut a rect from image
Mat cutImage(Mat image,int x,int y,int width,int height,int maxWidth, int maxHeight);

///cur a rotated rect from image
/// given rotated rect
Mat cutRotatedRect(Mat image, RotatedRect rect);

///return convex hull of work space
vector<Point> detectWorkspace(cv::Mat cameraFrame);

///use background substraction to detect the biggest object in work space
vector<Point> objDetect(cv::Mat cameraFrame);

///similar to objDetect, but return all objects in worksapce
vector<vector<Point>> objDetect2(cv::Mat cameraFrame);

///map object center in camera image to grid index (0-201,0-201)
Index objPos2grid2(Mat cameraFrame,vector<Point> obj);

///draw contour on image
void drawContour(Mat image,vector<Point> contour);

///return pixel position of feature patch
vector<Point2f> trackFeature(cv::Mat cameraFrame,cv::Mat queryImg,vector<Point> workspace);

///return pixel position of label on object
cv::Point2i trackLabel(cv::Mat cameraFrame);

///return edge points
/// filtered by worksapce
vector<Point> getEdges(Mat image,vector<Point> workspace=vector<Point>(0),double p1=200, double p2=300);

/////return id,position,x,y,z of each marker as a list
vector<MarkerData> detectMarker(Roopi& R,Mat cameraFrame);

///average old marker position/orientation
MarkerData updateMarkerDate(MarkerData oldMarkerDate,MarkerData newMarkerData, double alpha,bool ignoreId);
}

#endif // VISION

