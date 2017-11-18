#include "vision.h"
#include "shape.h"
#include "opencv2/shape.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/core.hpp"
#include "sc_dis.hpp"
#include "fstream"
#include <iostream>
using namespace cv;
using namespace std;
namespace grab_shape {

double sigmoid(double discriminant){
    return 1./(1.+exp(-discriminant));
}


//shape context descriptor of opencv, not used
Mat shapeContext(std::vector<Point> contour, double r_inner,double r_outer,int n_r,int n_theta){
    // Checking //
    InputArray contour1 = InputArray(contour);
    Mat sset1=contour1.getMat();
    Mat set1;
    if (set1.type() != CV_32F)
        sset1.convertTo(set1, CV_32F);
    else
        sset1.copyTo(set1);

    CV_Assert((set1.channels()==2) && (set1.cols>0));

    // Force vectors column-based
    if (set1.dims > 1)
        set1 = set1.reshape(2, 1);

    SCD set1SCE(4,5,0.125,2.,true);
    Mat set1SCD;
    set1SCE.extractSCD(set1,set1SCD);
    return set1SCD;

}

/**
 * @brief shapeContext2
 * @param contour
 * @param r_inner
 * @param r_outer
 * @param n_r
 * @param n_theta
 * @return
 */
Mat shapeContext2(std::vector<Point> contour, double r_inner,double r_outer,int n_r,int n_theta){

    InputArray contour1 = InputArray(contour);
    Mat sset1=contour1.getMat();
    Mat set1;
    if (set1.type() != CV_32F)
        sset1.convertTo(set1, CV_32F);
    else
        sset1.copyTo(set1);

    CV_Assert((set1.channels()==2) && (set1.cols>0));

    // Force vectors column-based
    if (set1.dims > 1)
        set1 = set1.reshape(2, 1);

    //compute
    SCD set1SCE(n_theta,n_r,r_inner,r_outer,true);
    cv::Mat disMatrix = cv::Mat::zeros(set1.cols, set1.cols, CV_32F);
    set1SCE.buildNormalizedDistanceMatrix(set1,disMatrix,std::vector<int>(),-1);
    std::vector<double> logspaces;
    set1SCE.logarithmicSpaces(logspaces);
    cv::Mat r_array_q = cv::Mat::zeros(set1.cols, set1.cols, CV_32F);
    for(int m=0;m<n_r;m++){
        for(int i=0;i<set1.cols;i++)
            for(int j=0;j<set1.cols;j++){
                if(disMatrix.at<float>(i,j)<logspaces.at(m))
                    r_array_q.at<float>(i,j)+=1;
            }
    }
    Mat angMat = Mat::zeros(set1.cols,set1.cols,CV_32F);
    for(int i =0;i<contour.size();i++)
        for(int j =0;j<contour.size();j++){
            Point p2 = contour.at(j);
            Point p1 = contour.at(i);
            if(p1==p2){
                angMat.at<float>(i,j)=0;
            }else{
                double angle = atan2((p2.y-p1.y),(p2.x-p1.x));
                if(angle<0) angle += 2*CV_PI;
                angMat.at<float>(i,j)=angle;
            }
        }
    cv::Mat theta_array_q = cv::Mat::zeros(set1.cols, set1.cols, CV_32F);

    for(int i=0;i<set1.cols;i++)
        for(int j=0;j<set1.cols;j++){

            theta_array_q.at<float>(i,j) =1+floor(angMat.at<float>(i,j)/(2*CV_PI/n_theta));
        }

    vector<Mat> descriptors;

    Mat descriptor = cv::Mat::zeros(contour.size(), n_r*n_theta, CV_32F);
    for(int i=0;i<set1.cols;i++){
        Mat sn = Mat::zeros(n_r,n_theta,CV_32F);
        for(int j=0;j<set1.cols;j++){
            if(r_array_q.at<float>(i,j)>0){
                int index1 =(int)(r_array_q.at<float>(i,j)-1);
                int index2 = (int)(theta_array_q.at<float>(i,j)-1);
                sn.at<float>(index1,index2) += 1;
            }

        }

        sn = sn.reshape(0,1);
        descriptors.push_back(sn);
    }
    vconcat(descriptors,descriptor);
    return descriptor;
}

/**
 * @brief sampleContour, uniformly sample edge points
 * @param contoursQuery, original contour
 * @param n,number of samples
 * @return
 */
vector<Point> sampleContour( vector<Point> contoursQuery, int n )
{
    // In case actual number of points is less than n
    int dummy=0;
    for (int add=(int)contoursQuery.size()-1; add<n; add++)
    {
        contoursQuery.push_back(contoursQuery[dummy++]); //adding dummy values
    }

    // Uniformly sampling
    random_shuffle(contoursQuery.begin(), contoursQuery.end());
    vector<Point> cont;
    for (int i=0; i<n; i++)
    {
        cont.push_back(contoursQuery[i]);
    }
    return cont;
}

//find indices
vector<int> findIndices(cv::Point query,vector<cv::Point> points){
    vector<int> indices;
    for(int i=0;i<points.size();i++){
        Point p = points.at(i);
        if(p.x==query.x && p.y == query.y){
            indices.push_back(i);
        }
    }
    return indices;
}

/**
 * @brief predictGraspingPoint
 * @param edges
 * @param image
 * @param threshold, probability threshold
 * @return
 */
vector<cv::Point> predictGraspingPoint(vector<cv::Point> edges,Mat image,double threshold){
    //parameters
    int step = 10;
    int patchSize1 = 10;
    int patchSize2 = 40;
    int patchSize3 = 80;
    vector<int> patchSize;
    patchSize.push_back(patchSize1);
    patchSize.push_back(patchSize2);
    patchSize.push_back(patchSize3);
    int n_r = 5;
    int n_theta =4;
    double inner_r = 0.125;
    double outer_r = 2.;
    int n_sample = 300;
    vector<Point> graspingPoints;
    vector<double> probs;

    //read coefficient
    vector<double> coef;
    std::ifstream file("coef/coef2.txt");
    std::string str;
    while (std::getline(file, str))
    {
        std::stringstream ss(str);
        std::string c;

        while(std::getline(ss, c, ',')) {
            coef.push_back(stof(c));
        }
    }

    Mat coefMat = Mat(coef);
    Mat sampled_edges_img = Mat::zeros(image.size.p[0],image.size.p[1],CV_8UC1);
    vector<Point> sampled_edges = grab_shape::sampleContour(edges);
    Mat shape_context = grab_shape::shapeContext2(sampled_edges,inner_r,outer_r,n_r,n_theta);
    //generate sampled edge image
    for(int i =0;i<sampled_edges.size();i++){
        Point edgePoint = sampled_edges[i];
        sampled_edges_img.at<uchar>(edgePoint.y,edgePoint.x)=255;
    }

    //get bounding box
    Rect box1 = cv::boundingRect(edges);
    RotatedRect box2 = cv::minAreaRect(edges);
    //    cv::rectangle(image,box1,Scalar(0,255,0),1);
    Point2f rect_points[4];
    box2.points(rect_points);
    //    for( int j = 0; j < 4; j++ ){
    //        line( image, rect_points[j], rect_points[(j+1)%4], Scalar(0,255,0), 1);
    //    }
    vector<Point2f> filter;
    filter.push_back(rect_points[0]);
    filter.push_back(rect_points[1]);
    filter.push_back(rect_points[2]);
    filter.push_back(rect_points[3]);

    //generate patch center
    vector<Point> centers;
    int n_x = box1.width/step;
    int n_y = box1.height/step;
    for(int i =0;i<n_x;i++)
        for(int j =0;j<n_y;j++){
            Point center = Point(box1.x+i*step,box1.y+j*step);
            if(cv::pointPolygonTest(filter,Point2f(center.x,center.y),false)>=0)
                centers.push_back(center);
            //                    cv::circle(image,center,2,Scalar(0,0,255));
        }

    //compute descriptor for each patch
    for(int i=0;i<centers.size();i++){
        Point center = centers.at(i);

        vector<Mat> descriptors;
        for(int j=0;j<patchSize.size();j++){
            int pSize = patchSize.at(j);
            Mat patch = grab_vision::cutImage(sampled_edges_img,center.x,center.y,pSize,pSize,image.size.p[1],image.size.p[0]);
            Point compensateStart = Point(max(center.x-pSize/2,0),max(center.y-pSize/2,0));
            vector<Point> inside;
            cv::findNonZero(patch,inside);
            cv::Mat accumulated = cv::Mat::zeros(1,n_r*n_theta,shape_context.type());
            if(inside.size()>0){
                for(int k=0;k<inside.size();k++){
                    Point insidePoint = inside.at(k)+compensateStart;
                    vector<int> indices = grab_shape::findIndices(insidePoint,sampled_edges);
                    //accumulate shape context
                    for(int l=0;l<indices.size();l++){
                        accumulated += shape_context.row(indices.at(l));
                    }
                }
            }else{
                //non edge points
            }
            descriptors.push_back(accumulated);

        }
        Mat final_descriptor;
        hconcat(descriptors,final_descriptor);
        //predict
        double d = 0;
        final_descriptor.convertTo(final_descriptor,coefMat.type());
        d = coefMat.t().dot(final_descriptor);
        double prob = grab_shape::sigmoid(d);
        if(prob>threshold){
            probs.push_back(prob);
            graspingPoints.push_back(center);
        }

    }

    //choose 5 grasping point with highest probability
    vector<Point> finalPoints;
    Point finalPoint;
    for(int j =0;j<5;j++){
        double maxProb=threshold;
        for(int i =0;i<graspingPoints.size();i++){
            if(probs.at(i)>=maxProb){
                //            finalPoints.push_back(graspingPoints.at(i));
                finalPoint = graspingPoints.at(i);
                graspingPoints.erase(graspingPoints.begin()+i);
                maxProb = probs.at(i);
            }
        }
        finalPoints.push_back(finalPoint);
    }
    return finalPoints;
}

}
