#ifndef KNN
#define KNN
#include "opencv2/core.hpp"
#include "random"

///this file contain the function used to
///split contour of objects by kmean clustering
namespace grab_kmean{
///find index of max
int maxIndex(std::vector<double> distance);

///find index of min
int minIndex(std::vector<double> distance);

///compute center of points
cv::Point2f meanOfCluster(std::vector<cv::Point> cluster);

///compute eculidean distance between two points
double computeDistance(cv::Point a, cv::Point b);

///assign points to cluster centers
std::vector<std::vector<cv::Point>> assign2center(std::vector<cv::Point> centers,std::vector<cv::Point> points);

///randomly generate centers
std::vector<cv::Point> randomInit(std::vector<cv::Point> points,int n);

///split contour into clusters by kmean
std::vector<std::vector<cv::Point>> splitContour(std::vector<cv::Point> contours,int maxIt,int num = 1);
}

#endif // KNN

