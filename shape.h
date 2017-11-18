#ifndef SHAPE
#define SHAPE
#include "opencv2/core.hpp"
using namespace std;
///this file contain the function used to compute shape context
/// and predict grasping point based on object edges
namespace grab_shape {

///sigmoid function
double sigmoid(double discriminant);

///implementation of SC from opencv
cv::Mat shapeContext(std::vector<cv::Point> contour, double r_inner=0.125,double r_outer=2.,int n_r=5,int n_theta = 4);

///our implementation of SC
cv::Mat shapeContext2(std::vector<cv::Point> contour, double r_inner=0.125,double r_outer=2.,int n_r=5,int n_theta = 4);

///uniformly sample edge points
vector<cv::Point> sampleContour( vector<cv::Point> contoursQuery, int n=300 );

///find index of query point in vector
vector<int> findIndices(cv::Point query,vector<cv::Point> points);

///predict grasping point given edge points
vector<cv::Point> predictGraspingPoint(vector<cv::Point> edges,cv::Mat image,double threshold=.5);

}
#endif // SHAPE

