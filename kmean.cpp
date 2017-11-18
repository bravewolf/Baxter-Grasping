#include "kmean.h"
#include "opencv2/core.hpp"
using namespace cv;
using namespace std;
namespace grab_kmean{

///return the index of max value in vector
int maxIndex(vector<double> distance){
    int index = 0;
    double value = distance.at(0);
    for(int i =0;i<distance.size();i++){
        if(distance.at(i)>value){
            index = i;
            value = distance.at(i);
        }
    }
    return index;
}

///return the index of min value in vector
int minIndex(vector<double> distance){
    int index = 0;
    double value = distance.at(0);
    for(int i =0;i<distance.size();i++){
        if(distance.at(i)<value){
            index = i;
            value = distance.at(i);
        }
    }
    return index;
}

///compute center of cluster
Point2f meanOfCluster(vector<Point> cluster){

    Point sum = Point(0,0);//ini
    for(int i =0;i<cluster.size();i++){
        sum += cluster.at(i);
    }
    Point2f newCenter = Point2f((double)sum.x*1./cluster.size(),(double)sum.y*1./cluster.size());
    return newCenter;

}

///return distance between two points
double computeDistance(Point a, Point b){
    Point2f c = a-b;

    return sqrt((double)c.ddot(c));
}

///assign each point to closest center
vector<vector<Point>> assign2center(vector<Point> centers,vector<Point> points){
    vector<vector<Point>> clusters(centers.size());
    //assign to center
    for (vector<Point>::iterator it = points.begin() ; it != points.end(); ++it){
        //compute distance to each center
        vector<double> distanceList;
        for(vector<Point>::iterator it2 = centers.begin() ; it2 != centers.end(); ++it2){
            distanceList.push_back(computeDistance(*it,*it2));
        }
        //assign
        int i = minIndex(distanceList);
        clusters.at(i).push_back(*it);
    }
    return clusters;
}

/**
 * @brief randomInit, randomly choose initial center from contour
 * @param points, original contours
 * @param n, number of initial center
 * @return
 */
vector<Point> randomInit(vector<Point> points,int n){
    vector<Point> init;
    int min = 0;
    int max = points.size()-1;
    int c = max/(n+1);
    for(int i=0;i<n;i++){
        int output = min + (rand() % static_cast<int>(max - min+1));
        init.push_back(points.at(output));
    }
    return init;
}

/**
 * @brief splitContour
 * @param contours
 * @param maxIt, max iteration
 * @param num, number of clusters
 * @return
 */
vector<vector<Point>> splitContour(vector<Point> contours,int maxIt,int num){
    vector<Point> newCenters = randomInit(contours,num);
    vector<vector<Point>> clusters;
    //iterate
    for(int i =0;i<maxIt;i++){
        clusters = assign2center(newCenters,contours);
        //compute new centers
        newCenters.clear();//empty vector
        for(vector<vector<Point>>::iterator it = clusters.begin() ; it != clusters.end(); ++it){
            vector<Point> cluster = *it;
            if(cluster.size()>0){
                Point newCenter = meanOfCluster(*it);
                newCenters.push_back(newCenter);
            }else{

            }
        }

    }

    return clusters;
}

}
