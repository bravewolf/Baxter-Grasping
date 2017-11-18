#include <Perception/opencv.h>
#include <opencv2/opencv.hpp>
#include <Core/array.h>
#include <Core/thread.h>
#include "subscribeBaxterHand.h"
#include "subscribeBaxterGripper.h"
#include <Roopi/roopi.h>
#include <Control/taskControl.h>
//#include <opencv2/aruco.hpp>
#include "opencv2/core.hpp"
//#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <list>
#include <iostream>
#include <fstream>
#include <string>
#include "basictask.h"
#include "opencv2/stereo.hpp"
#include "stereo.h"
#include "basictask.h"
#include "vision.h"
#include "datatype.h"
#include "shape.h"
#include "kmean.h"
using namespace cv;
using namespace std;
using namespace cv::stereo;
using namespace basictask;
using namespace grab_vision;
using namespace grab_kmean;

///check whether gripper successfully grip something
bool checkGripper(Roopi& R,double gripCriterion){
    Access<double> force("baxter_left_gripper");
    grip(&R,true);//close gripper
    mlr::wait(4.);//wait 4 seconds
    cout<<"Check gripper force "<<force.get()<<endl;
    if(force.get()<gripCriterion){ //check force sensor
        //open gripper if nothing in gripper
        grip(&R,false);
        return false;
    }else{
        return true;
    }

}

///read policy from csv file
void readPolicy(arr& policy,std::string filename){
    std::ifstream file(filename);
    std::string str;
    while (std::getline(file, str))
    {
        std::stringstream ss(str);
        int x,y,action;
        int index = 0;
        double temp;
        while (ss >> temp)
        {
            switch(index){
            case 0:
                x = (int)temp;
                break;
            case 1:
                y  = (int)temp;
                break;
            case 2:
                action  = (int)temp;
                break;
            }
            if (ss.peek() == ',')
                ss.ignore();
            index++;
        }
        policy(x,y) = action;
    }
}

///choose grasping rectangle from object contour
/// select grasping point as the center of the smallest rect
/// filtering rects by aspect ratio and area
RotatedRect chooseGraspingRect(vector<Point> contour,int maxIt, int num=1){
    //split contour by kmean clustering
    vector<vector<Point>> clusters = splitContour(contour,maxIt,num);

    //choose grasping box
    vector<double> areaList,aspectRatioList;
    vector<RotatedRect> rectList;

    for(vector<vector<Point>>::iterator it2 = clusters.begin() ; it2 != clusters.end(); ++it2){
        vector<Point> cluster = *it2;
        RotatedRect boundRect = minAreaRect( Mat(cluster) );
        Point2f rect_points[4];
        boundRect.points( rect_points );
        Size2f size =  boundRect.size;
        float area = size.area();
        float aspectRatio;
        if(size.height>=size.width)
            aspectRatio = size.height/size.width;
        else
            aspectRatio = size.width/size.height;
        //filter by aspect ratio
        if(aspectRatio<4.){
            rectList.push_back(boundRect);
            areaList.push_back(area);
            aspectRatioList.push_back(aspectRatio);
        }
    }

    //choose rectangle with minimal area
    if(areaList.size()>0){
        int min = minIndex(areaList);
        return rectList.at(min);
    }else{
        return minAreaRect( Mat(contour) );
    }

}

///use stereo matching to compute depth map
/// average depth over grasping box(deprecated)
float getDepthEstimation(Roopi& Ro, mlr::Vector current,RotatedRect graspingBox){
    //camera parameters
    Mat cameraMatrix, distCoeffs;
    float data[] = {   4.0364801365017615e+02, 0., 6.3146928830350316e+02, 0.,
                       4.0620637674934119e+02, 3.6213656430315433e+02, 0., 0., 1.  };
    float data2[] = { 2.5817367285864371e-02, -6.4939093957227453e-02,
                      -1.6997464300167172e-04, -8.7200818996337727e-05,
                      1.8531601637796945e-02 };
    cameraMatrix=cv::Mat(3,3,CV_32F,data);
    distCoeffs=cv::Mat(5,1,CV_32F,data2);
    Size size = Size(1280,800);

    //extrinic parameters
    Mat R = Mat::eye(3,3,CV_32F);
    float t[] = {0.02,0,0};
    float t2[] = {0.02,0,0};
    Mat T = Mat(3,1,CV_32F,t);
    Mat T2 = Mat(3,1,CV_32F,t2);


    mlr::Vector x = {1,0,0};
    mlr::Vector z = {0,0,1};

    Access<cv::Mat> images("baxter_left_camera");
    //move to initial pos
    pos_alignTask(Ro,current,x,-z,"endeffL","world");
    mlr::wait(2.);
    Mat center = images.get()();


    mlr::Vector next = addRelativeTranslation(Ro,"endeffL",mlr::Vector(0,-0.02,0));
    mlr::Vector next2 = addRelativeTranslation(Ro,"endeffL",mlr::Vector(0,0.04,0));
    //move to left position
    pos_alignTask(Ro,next,x,-z,"endeffL","world");//move to next pos
    mlr::wait(2.);
    Mat imgR = images.get()();

    //move to right position
    pos_alignTask(Ro,next2,x,-z,"endeffL","world");//move to next2 pos
    mlr::wait(2.);
    Mat imgL = images.get()();

    Mat dist,dist2,_3dImage,_3dImage2;
    dist = grab_stereo::disparityMap(center,imgR);//left matcher
    dist2 = grab_stereo::disparityMap(imgL,center,true);//right matcher
    dist2 = -(dist2+1);
    Mat merged = (dist+dist2)/2;//merge two disparity map

    _3dImage = grab_stereo::stereoMatching(dist,cameraMatrix,distCoeffs,size,R,T2);
    _3dImage2 = grab_stereo::stereoMatching(dist2,cameraMatrix,distCoeffs,size,R,T);

    //display
    Mat imgDisparity8U,imgDisparity8U2 = Mat( center.rows, center.cols, CV_8UC1 );//for display
    double minVal; double maxVal;
    minMaxLoc( dist, &minVal, &maxVal );
    printf("Min disp: %f Max value: %f \n", minVal, maxVal);
    dist.convertTo( imgDisparity8U, CV_8UC1, 255/(maxVal - minVal));

    minMaxLoc( merged, &minVal, &maxVal );
    printf("Min disp: %f Max value: %f \n", minVal, maxVal);
    merged.convertTo( imgDisparity8U2, CV_8UC1, 255/(maxVal - minVal));

    //


    Point2f rect_points[4];
    graspingBox.points( rect_points );

    //for display
    for( int j = 0; j < 4; j++ ){
        cv::line( center, rect_points[j], rect_points[(j+1)%4], Scalar(255,255,255), 2, 2 );
        cv::line( imgDisparity8U, rect_points[j], rect_points[(j+1)%4], Scalar(255,255,255), 2, 2 );
        cv::line( imgDisparity8U2, rect_points[j], rect_points[(j+1)%4], Scalar(255,255,255), 2, 2 );
    }

    //average depth imformation
    Mat depthMap[3],depthMap2[3];
    split(_3dImage,depthMap);
    split(_3dImage2,depthMap2);
    Mat depth,x_m,y_m,depth2;
    float sum,sum2,sum3=0;
    int counter,counter2=0;
    depth = cutRotatedRect(depthMap[2],graspingBox);
    depth2 = cutRotatedRect(depthMap2[2],graspingBox);

    for(int i=0;i<depth.rows;i++){
        for(int j =0;j<depth.cols;j++){
            float d = depth.at<float>(Point(i,j));
            if(d<0.3 && d>0.0){
                //                cout<<"d "<<d<<endl;
                sum += d;
                counter +=1;
            }

            d = depth2.at<float>(Point(i,j));
            if(d<0.3 && d>0.1){
                //                cout<<"d "<<d<<endl;
                sum2 += d;
                counter2 +=1;
            }

        }
    }
    float averageDepth = sum/counter;
    float averageDepth2 = sum2/counter2;//average depth
    cout<<"counter "<<sum<<" "<<counter<<endl;
    cout<<"counter "<<sum2<<" "<<counter2<<endl;

    cout<<"center "<<depthMap[0].at<float>(graspingBox.center)
       <<endl<<depthMap[1].at<float>(graspingBox.center)
      <<endl<<depthMap[2].at<float>(graspingBox.center)<<endl;

    cout<<"center "<<depthMap2[0].at<float>(graspingBox.center)
       <<endl<<depthMap2[1].at<float>(graspingBox.center)
      <<endl<<depthMap2[2].at<float>(graspingBox.center)<<endl;

    cout<<"average depth: "<<averageDepth<<endl;
    cout<<"average depth: "<<averageDepth2<<endl;
    imshow("center",center);
    imshow("left",imgL);
    imshow("right",imgR);
    imshow("disparity ",imgDisparity8U);
    imshow("disparity2 ",imgDisparity8U2);
    waitKey(0);
    pos_alignTask(Ro,current,x,-z,"endeffL","world");//back to init pos
    return averageDepth2;
}

///record video from cam on left hand, used for calibration
void recordVideoFromRobot(string filename){
    Access<cv::Mat> images("baxter_left_camera");
    SubscribeBaxterHand sbh;
    cv::Mat image; // = cv::imread(argv[1], cv::IMREAD_COLOR);

    ///create output stream
    int frame_width = 1280;
    int frame_height = 800;
    VideoWriter video(filename,CV_FOURCC('M','J','P','G'),10, Size(frame_width,frame_height),true);
    while (1)
    {
        image = images.get()();

        if (image.empty()){
            std::cout << "Failed to load image." << std::endl;
            mlr::wait(1.);
            continue;
        }
        video.write(image);
        imshow( "Cam", image );
        char c = (char)waitKey(33);
        if( c == 27 ) break;

    }

}

///TEST vision part on webcam of laptop
void webCamStart(arr& policy, Roopi& R){
    VideoCapture stream1(0);   //0 is the id of video device.0 if you have only one camera.
    if (!stream1.isOpened()) { //check if video device has been initialised
        cout << "cannot open camera"<<endl;
    }


    MarkerData oldMarkerData = {-1,
                                mlr::Vector(0,0,0),
                                mlr::Vector(0,0,0),
                                mlr::Vector(0,0,0),
                                mlr::Vector(0,0,0)};
    Point2f averageFeatureCenter = Point2f(0.,0.);
    Mat grayPatch = imread("feature_match/box.jpg",0);
    //    Mat grayPatch;
    while(true){
        cv::Mat cameraFrame;//modify here to couple this code to connect real robot camera
        stream1.read(cameraFrame);//modify here to couple this code to connect real robot camera
        cv::Mat originFrame = cameraFrame.clone();//clone origin frame for feature detection

        if(0){
            vector<Point> obj = objDetect(cameraFrame);
            objPos2grid2(cameraFrame,obj);
        }

        //test multi obj detect
        if(0){
            vector<vector<Point>> objs = grab_vision::objDetect2(cameraFrame);
            cv::drawContours(cameraFrame,objs,-1,cv::Scalar(0,255,0),2);
        }

        ///detect marker
        if(1){
            vector<MarkerData> result = detectMarker(R,cameraFrame);
            if(result.size()>0) {
                MarkerData m = result.at(0);
                if(m.id!=-1){
                    oldMarkerData = updateMarkerDate(oldMarkerData,m,.1,true);
                    cout<<"id "<<m.id<<endl
                       <<"pos "<<oldMarkerData.pos<<endl
                      <<"x "<<oldMarkerData.x<<endl
                     <<"y "<<oldMarkerData.y<<endl
                    <<"z "<<oldMarkerData.z<<endl;

                }else{
                    cout<<"No marker detected"<<endl;
                }

            }
        }

        //TEST cut patch from image
        if(0){

            Point2i label =  trackLabel(cameraFrame);
            //            objPos2grid(cameraFrame);

            if(label.x != -1){
                cout<<"label position "<<label<<endl;
                //cut from origin frame
                Mat partOfImage = cutImage(originFrame,label.x,label.y,100,70,cameraFrame.cols,cameraFrame.rows);
                Mat greyPatch;
                cv::cvtColor(partOfImage, greyPatch, cv::COLOR_BGR2GRAY);
                imshow("small",greyPatch);
            }
        }

        //TEST track feature
        if(0){
            Mat queryImg = imread( "feature_match/grasp_point2.png", IMREAD_GRAYSCALE );
            //            imshow("queryImg",queryImg);
            //convert to grey scale
            cv::cvtColor(cameraFrame, cameraFrame, cv::COLOR_BGR2GRAY);
            vector<Point> empty;
            vector<Point2f> corners = trackFeature(cameraFrame,queryImg,empty);
            if(corners.at(0).x!=-1){
                line( cameraFrame, corners.at(0),  corners.at(1), Scalar(0, 255, 0), 4 );
                line( cameraFrame,  corners.at(1) ,  corners.at(2), Scalar( 0, 255, 0), 4 );
                line( cameraFrame, corners.at(2) ,  corners.at(3), Scalar( 0, 255, 0), 4 );
                line( cameraFrame,  corners.at(3) ,  corners.at(0), Scalar( 0, 255, 0), 4 );
                Point2f center =  (corners.at(0)+ corners.at(1)+ corners.at(2)+ corners.at(3))/4.;
                averageFeatureCenter = .95*averageFeatureCenter+.05*center;
                circle(cameraFrame,averageFeatureCenter,20,Scalar(255,0,0),5);
            }
        }

        if(0){
            //detect and draw workspace
            vector<Point> workspace = detectWorkspace(cameraFrame);
            vector<vector<Point>> temp;
            temp.push_back(workspace);
            cv::drawContours(cameraFrame,temp,0,Scalar(0,255,0),2);


            //detect obj
            vector<Point> obj = objDetect(cameraFrame);
            objPos2grid2(cameraFrame,obj);

            //choose grasping box
            if(obj.size()>0){

                //test shape similarity
                shapeContextDescriptor(obj);
                //choose multiple times!!!

                for(int i =0;i<5;i++){

                }
                RotatedRect graspingBox = chooseGraspingRect(obj,40);
                graspingBox.center;
                Point2f rect_points[4];

                //display grasping box angle for wrist aligning
                if(graspingBox.size.width < graspingBox.size.height){
                    cout<<"rect angle"<<graspingBox.angle+180<<endl;

                }else{
                    cout<<"rect angle"<<graspingBox.angle+90<<endl;
                }

                graspingBox.points( rect_points );
                for( int j = 0; j < 4; j++ ){
                    cv::line( cameraFrame, rect_points[j], rect_points[(j+1)%4], Scalar(255,255,255), 2, 2 );
                }

                float a = 1.0;
                Size2f newSize = Size2f(graspingBox.size.width*a,graspingBox.size.height*a);
                RotatedRect enlarged = RotatedRect(graspingBox.center,newSize,graspingBox.angle);

                Mat partOfImage = cutRotatedRect(originFrame,enlarged);
                //                cv::cvtColor(partOfImage, grayPatch, cv::COLOR_BGR2GRAY);
                //                imshow("feature patch",partOfImage);


            }

            if(!grayPatch.empty() && false){
                //                vector<Point> empty;
                trackFeature(originFrame,grayPatch,workspace);
            }


        }

        imshow("Cam",cameraFrame);
        cv::resizeWindow("Cam",540,400);
        if (waitKey(30) >= 0)
            break;
    }
}

///start grasping
///function to fulfill grasping task
void startGrasping(arr& policy, Roopi& R){
    ///get image from left cam
    Access<cv::Mat> images("baxter_left_camera");
    Access<double> force("baxter_left_gripper");
    cv::Mat image,originImage,partOfImage,greyPatch;
    //marker variables
    int currentMarkerId = -1;
    MarkerData oldMarkerData = {-1,
                                mlr::Vector(0,0,0),
                                mlr::Vector(0,0,0),
                                mlr::Vector(0,0,0),
                                mlr::Vector(0,0,0)};
    mlr::Vector oldAveragePos = {0.,0.,0.};
    double difference = 10.;
    double difference2 = 10.;
    double alpha = .1;

    //flags
    bool done = false; //grasping task done
    bool positionAlign = false;//align position with object in 2d
    bool poseAlign = false;//align orientation with object
    bool getFeaturePatch = false;//cut grasping patch
    bool adjustPos = true;//adjust position if grasping fail

    //counters
    int failure = 0;
    int badFeaturePatch = 0;
    int badFeaturePatch2 = 0;
    int noMarker = 0;

    //grasping point position estimation
    Point2f estimation;
    RotatedRect graspingBox;
    vector<Point> graspingPoints;
    Point2f averageFeatureCenter = Point2f(640.,400.);
    Point2f centerOfImage = Point2f(640.,400.);

    //move to start position
    pos_alignTask(R,mlr::Vector(.6,.2,1.2),mlr::Vector(1,0,0),mlr::Vector(0,0,-1),"endeffL","world");

    while (!done)
    {
        image = images.get()();
        if (image.empty()){
            std::cout << "Failed to load image." << std::endl;
            mlr::wait(1.);
            continue;
        }
        originImage = image.clone();//clone origin image for feature detection

        //detect marker and move to marker position/track object by policy
        if(!positionAlign){
            //detect marker
            vector<MarkerData> markerDataList =  detectMarker(R,image);
            cout<<"Num of marker "<<markerDataList.size()<<endl;

            MarkerData marker;
            int markerId = -1;
            //find marker with max id if currentMarker is empty (equals -1)
            if(currentMarkerId == -1){
                int maxId = -1;
                for(int i =0;i<markerDataList.size();i++){
                    if(markerDataList.at(i).id>maxId){
                        maxId = markerDataList.at(i).id;
                        currentMarkerId = maxId;
                        marker = markerDataList.at(i);
                    }

                }

            }else{
                //find current marker if currentMarker is not -1
                for(int i =0;i<markerDataList.size();i++){
                    if(markerDataList.at(i).id == currentMarkerId){
                        markerId = markerDataList.at(i).id;
                        marker = markerDataList.at(i);
                    }
                }
            }
            cout<<"Marker id"<<markerId<<endl;
            cout<<"Current id"<<currentMarkerId<<endl;

            if(markerId == currentMarkerId && markerId != -1){
                //update average position of marker
                oldAveragePos.set(oldMarkerData.pos.x,oldMarkerData.pos.y,oldMarkerData.pos.z);
                oldMarkerData = updateMarkerDate(oldMarkerData,marker,alpha,true);
                difference = (oldAveragePos-oldMarkerData.pos).length();
                difference2 = (getTransformation(R,"endeffL").pos-oldMarkerData.pos).length();

                cout<<"Marker id: "<<markerId<<endl
                   <<"Average position of marker in world: "<<oldMarkerData.pos<<endl
                  <<"Average x axis of marker in world: "<<oldMarkerData.x<<endl
                 <<"Difference of average position: "<<difference<<endl
                <<"Difference between endeff and avePosition: "<<difference2<<endl;

                //check difference between endeff and marker position
                if(difference2<0.02) {
                    positionAlign=true;
                    currentMarkerId = -1;

                }

                //check difference between current marker position and averaged marker position
                //move arm to marker position
                if(difference<0.005){
                    mlr::Vector ref1 = {-1,0.,0.};
                    mlr::Vector v1 = {0,0,1};
                    pos_alignTask(R,oldMarkerData.pos,ref1,v1,"endeffL","world");
                }


            }else{

                //lift arm if no marker counter larger than 100
                noMarker += 1;
                if(noMarker>100){
                    performRelativeAction(R,5,true,0.012);//lift arm if no marker be detected
                    performRelativeAction(R,5,true,0.012);
//                    performRelativeAction(R,5,true,0.012);
//                    performRelativeAction(R,5,true,0.012);
                    noMarker /= 2;
                }

                //if no marker detected
                //move arm to center of object
                //by simple controller/policy
                vector<Point> obj = objDetect(image);
                Index index = objPos2grid2(image,obj);
                if(index.x!=-1){

                    int action = simpleController(index,Index{100,110},5);
                    cout<<"Position align->simple controller: "<<action<<endl;

                    //uncomment to use policy
                    //convert action index since the action index defined in python is different
                    //            int action = policy(index.x,index.y);
                    //            if(action==0) {action = 2;}
                    //            if(action==1) {action = 3;}
                    //            if(action==2) {action = 0;}
                    //            if(action==3) {action = 1;}

                    if(action!=-1){//-1 means centroid of object is roughly align with image center
                        performRelativeAction(R,action,true,0.012);
                    }else{
                        if(getTransformation(R,"endeffL").pos.z>1.03+.2)
                            performRelativeAction(R,4,true,0.012);//lower arm

                        if(!adjustPos){
                            positionAlign = true;
                            currentMarkerId = -1;
                        }
                    }

                }else{
                    cout<<"No object and marker detected"<<endl;
                    //pos_alignTask(R,mlr::Vector(.6,.2,1.2),mlr::Vector(1,0,0),mlr::Vector(0,0,-1),"endeffL","world");
                }
            }
        }

        //move to observation position after done
        //currently only align orientation with object
        if(positionAlign) {
            if(!poseAlign){
                cout<<"Move to obvservation position"<<endl;

                //this part move the endeff to the side of object and point to object
                //not used currently
                //-----------------------------------------
                mlr::Vector ref1 = {1,0.,0.};//x axis of gripper, used for pointing gripper to obj
                //            mlr::Vector ref2 = {0,1.,0.};
                mlr::Vector ref3 = {0,0,-1};//z axis of gripper, used for keeping camera side down
                //            mlr::Vector x = {0,-1,0};

                double height = std::max(1.03,oldMarkerData.pos.z-.3);
                mlr::Vector tempPos = mlr::Vector(oldMarkerData.pos.x,oldMarkerData.pos.y+.2,oldMarkerData.pos.z);
                mlr::Vector obvPos1 = mlr::Vector(oldMarkerData.pos.x,oldMarkerData.pos.y+.2,height);
                mlr::Vector obvPos2 = mlr::Vector(oldMarkerData.pos.x,oldMarkerData.pos.y,oldMarkerData.pos.z);

                mlr::Vector point2marker = oldMarkerData.pos-obvPos1;
                point2marker.z = 0.;
                point2marker.normalize();
                //                posTask(R,obvPos2,"endeffL","world");
                //observation position + align
                //                pos_align_alignTask(R,obvPos1,ref1,point2marker,ref3,mlr::Vector(0,0,-1),"endeffL","world");
                //----------------------------------------

                //align with obj
                vector<Point> obj = objDetect(image);
                if(obj.size()>0){

                    RotatedRect bound = minAreaRect(obj);
                    twistWrist2(R,bound);//align wrist with obj
                    mlr::wait(2.);
                    image = images.get()();//update image
                    originImage = image.clone();

                }

                poseAlign = true;
                getFeaturePatch = false;
            }else{

                vector<Point> obj = objDetect(image);

                //choose grasping point
                if(obj.size()>0){
                    if(!getFeaturePatch){
                        cout<<"Choose grasping point"<<endl;
                        image = images.get()();
                        originImage = image.clone();
                        vector<Point> edges = grab_vision::getEdges(image,obj);//extract edges from image
                        edges.insert(edges.end(),obj.begin(),obj.end());//concatenate edges and contour
                        graspingPoints = grab_shape::predictGraspingPoint(edges,image);//predict grasping point

                        //display grasping point
                        //                        for(int i = 0;i<graspingPoints.size();i++){
                        //                            cv::circle(image,graspingPoints.at(i),2,Scalar(0,255,0),2);
                        //                        }

                        //                       grab_vision::drawContour(image,edges);
                        //                       imshow("edge",originImage);
                        //                        waitKey(0);

                        if(graspingPoints.size()>0){
                            Point2f graspingPoint = Point2f(graspingPoints.at(0).x,graspingPoints.at(0).y);
                            //generate grasping box
                            graspingBox = chooseGraspingRect(obj,5,2);
                            graspingBox.center = Point2f(graspingPoint.x,graspingPoint.y);

                            //first grasping
                            cout<<"First grasp"<<endl;
                            mlr::Vector v1 = {1,0,0};
                            mlr::Vector v2 = {0,0,-1};
                            double lowerArm = -.11;//used to lower arm
                            estimation = (graspingPoint-centerOfImage)/405.*0.2;
                            cout<<"First estimation"<<estimation<<endl;
                            mlr::Vector final_pos = addRelativeTranslation(R,"endeffL",mlr::Vector(0,-estimation.x,-estimation.y-0.04));
                            pos_alignTask(R,final_pos,v1,v2,"endeffL","world");
                            final_pos.z += lowerArm;
                            pos_alignTask(R,final_pos,v1,v2,"endeffL","world");
                            //check gripper
                            if(checkGripper(R,10.)){
                                //to ensure firmly grip
                                grip(&R,false);
                                performRelativeAction(R,4,false);
                                grip(&R,true);
                                mlr::Vector final_pos = {.6,.3,1.5};
                                posTask(R,final_pos,"endeffL","world");
                                if(checkGripper(R,10.)){
                                    cout<<"Grasp something!!"<<endl;
                                    //done = true;

                                    //back to start position and reset params
                                    failure = 0;
                                    grip(&R,false);
                                    averageFeatureCenter = Point2f(640.,400.);//reset feature patch center
                                    mlr::wait(1.);
                                    positionAlign = false;//track object again
                                    adjustPos = false;//track object again
                                    poseAlign = false;//align with object again
                                    getFeaturePatch = false;//cut feature patch again
                                    //back to start position
                                    pos_alignTask(R,mlr::Vector(.6,.2,oldMarkerData.pos.z),mlr::Vector(1,0,0),mlr::Vector(0,0,-1),"endeffL","world");

                                }else{
                                    pos_alignTask(R,oldMarkerData.pos,mlr::Vector(1,0,0),mlr::Vector(0,0,-1),"endeffL","world");
                                }

                            }else{
                                //back to marker position
                                pos_alignTask(R,oldMarkerData.pos,mlr::Vector(1,0,0),mlr::Vector(0,0,-1),"endeffL","world");
                            }

                        }else{
                            //choose grasping point by kmean clustering
                            //if no predicted grasping point
                            graspingBox = chooseGraspingRect(obj,5,2);
                        }

                        //cut a patch around grasping point
                        cout<<"Cut patch"<<endl;
                        float a = 1.3;//scale factor of patch
                        Size2f newSize = Size2f(graspingBox.size.width*a,graspingBox.size.height*a);
                        RotatedRect enlarged = RotatedRect(graspingBox.center,newSize,graspingBox.angle);
                        partOfImage = cutRotatedRect(originImage,enlarged);
                        cv::cvtColor(partOfImage, greyPatch, cv::COLOR_BGR2GRAY);
                        getFeaturePatch = true;
                    }
                }

                //track grasping patch
                if(getFeaturePatch && failure<=5){
                    imshow("Feature patch",partOfImage);//display grasping patch

                    //re-estimate grasping point position
                    cout<<"Estimated grasping point "<<(averageFeatureCenter-centerOfImage)/405.*0.2<<endl;
                    estimation = (averageFeatureCenter-centerOfImage)/405.*0.2;

                    vector<Point> workspace = detectWorkspace(originImage);//detect workspace
                    drawContour(image,workspace);
                    vector<Point2f> corners = trackFeature(originImage,greyPatch,workspace);//tracking grasping patch
                    if(corners.at(0).x!=-1){
                        //filter noisy feature tracking by aspect ratio
                        //compute feature patch aspect ratio
                        RotatedRect box = cv::minAreaRect(corners);
                        float aspectRatio;
                        if(box.size.width>box.size.height)
                            aspectRatio = box.size.width/box.size.height;
                        else
                            aspectRatio = box.size.height/box.size.width;

                        //compute original grasping patch aspect ratio as threshold
                        float aspectRatioThresh;
                        if(graspingBox.size.width>graspingBox.size.height)
                            aspectRatioThresh = graspingBox.size.width/graspingBox.size.height;
                        else
                            aspectRatioThresh = graspingBox.size.height/graspingBox.size.width;

                        //update the center of feature patch if the aspect ratio is within 2xoriginal aspect ratio
                        if(aspectRatio<2*aspectRatioThresh){
                            //update center of averaged grasping patch center
                            Point2f center =  (corners.at(0)+ corners.at(1)+ corners.at(2)+ corners.at(3))/4.;
                            Point2f oldCenter = Point2f(averageFeatureCenter.x,averageFeatureCenter.y);
                            averageFeatureCenter = .7*averageFeatureCenter+.3*center;
                            double diff = (oldCenter-averageFeatureCenter).ddot(oldCenter-averageFeatureCenter);
                            cout<<"Difference between aveFeatureCenter: "<<diff<<endl;

                            //compare difference between old center and updated center in pixels
                            //threshold is 9 pixel^2 -> 3 pixel
                            if(diff<9.){

                                cout<<"Move to grasping point"<<endl;
                                mlr::Vector v1 = {1,0,0};
                                mlr::Vector v2 = {0,0,-1};
                                double lowerArm = -.11;//used to lower arm
                                mlr::Vector final_pos = addRelativeTranslation(R,"endeffL",mlr::Vector(0,-estimation.x,-estimation.y-0.04));
                                pos_alignTask(R,final_pos,v1,v2,"endeffL","world"); //move in 2d plane
                                final_pos.z += lowerArm;
                                pos_alignTask(R,final_pos,v1,v2,"endeffL","world");//lower arm
                                mlr::wait(1.);
                                //try grip
                                grip(&R,true);
                                performRelativeAction(R,4,false,0.012);
                                grip(&R,true);

                                if(checkGripper(R,10.)){//force threshold is 10% of full
                                    //open gripper again and lower gripper to ensure firmly grip
                                    grip(&R,false);
                                    performRelativeAction(R,4,false);
                                    grip(&R,true);
                                    mlr::Vector final_pos = {.6,.3,1.5};
                                    posTask(R,final_pos,"endeffL","world");

                                    //check after lifting the object
                                    if(checkGripper(R,10.)){
                                        //done = true;

                                        //back to start position and reset params
                                        failure = 0;
                                        grip(&R,false);
                                        averageFeatureCenter = Point2f(640.,400.);//reset feature patch center
                                        mlr::wait(1.);
                                        positionAlign = false;//track object again
                                        adjustPos = false;//track object again
                                        poseAlign = false;//align with object again
                                        getFeaturePatch = false;//cut feature patch again
                                        //back to start position
                                        pos_alignTask(R,mlr::Vector(.6,.2,oldMarkerData.pos.z),mlr::Vector(1,0,0),mlr::Vector(0,0,-1),"endeffL","world");

                                    }else{
                                        //if no object in gripper
                                        failure += 1;//increment failure counter
                                        //back to start position and reset flags
                                        averageFeatureCenter = Point2f(640.,400.);//reset feature patch center
                                        pos_alignTask(R,oldMarkerData.pos,mlr::Vector(1,0,0),mlr::Vector(0,0,-1),"endeffL","world");
                                        mlr::wait(1.);
                                        positionAlign = false;//track object again
                                        adjustPos = false;//track object again
                                        poseAlign = false;//align with object again
                                        getFeaturePatch = false;//cut feature patch again
                                    }

                                }else{
                                    //if no object in gripper
                                    failure += 1;//increment failure counter
                                    //back to obv pos and reset flags
                                    averageFeatureCenter = Point2f(640.,400.);
                                    pos_alignTask(R,oldMarkerData.pos,mlr::Vector(1,0,0),mlr::Vector(0,0,-1),"endeffL","world");
                                    mlr::wait(1.);
                                    positionAlign = false;//track object again
                                    adjustPos = false;//track object again
                                    poseAlign = false;//align with object again
                                    getFeaturePatch = false;//cut feature patch again
                                }

                            }

                        }
                    }else{
                        //inrecment bad feature counter is can not track feature
                        badFeaturePatch2 += 1;
                    }
                    //display average grasping patch center
                    cv::circle(image,averageFeatureCenter,3,Scalar(0,0,255),2);

                    //increment bad feature patch counter
                    if(estimation.x==0 && estimation.y==0){
                        badFeaturePatch += 1;//increment if can not track grasping patch
                    }

                    //choose grasping point again
                    if(badFeaturePatch>5 || badFeaturePatch2> 20){
                        badFeaturePatch = 0;
                        badFeaturePatch2 = 0;
                        failure += 1;
                        positionAlign = false;
                        adjustPos = false;
                        poseAlign = false;
                        getFeaturePatch = false;
                    }
                }

                //if baxter can not grasp object after 5 trails, track object center
                if(failure>5){
                    //twist wrist
                    if(obj.size()>0){
                        RotatedRect box = minAreaRect(splitContour(obj,1,1).at(0));
                        twistWrist2(R,box);//align wrist with obj
                        mlr::wait(1.);

                        image = images.get()();
                        originImage = image.clone();
                        obj = objDetect(image);
                    }
                    cout<<"Grasping by center tracking"<<endl;
                    Index index = objPos2grid2(image,obj);
                    if(index.x!=-1){
                        //simple controller
                        int action = simpleController(index,Index{100,130},3);
                        cout<<"Grasping->simple controller: "<<action<<endl;

                        //uncomment to use policy
                        //convert action index since the action index defined in python is different cpp
                        //            int action = policy(index.x,index.y);
                        //            if(action==0) {action = 2;}
                        //            if(action==1) {action = 3;}
                        //            if(action==2) {action = 0;}
                        //            if(action==3) {action = 1;}


                        if(action!=-1){
                            performRelativeAction(R,action,true,0.012);
                        }else{
                            cout<<"Grip&move"<<endl;
                            performRelativeAction(R,4,true,0.02);
                            if(checkGripper(R,10.)){
                                //to ensure firmly grip
                                grip(&R,false);
                                performRelativeAction(R,4,false);
                                //                                performRelativeAction(R,4,false);
                                grip(&R,true);

                                cout<<"Grasp something!!"<<endl;
                                mlr::Vector final_pos = {.6,.3,1.5};
                                posTask(R,final_pos,"endeffL","world");
                                done = true;
                                pos_alignTask(R,mlr::Vector(.5,.2,1.2),mlr::Vector(1,0,0),mlr::Vector(0,0,-1),"endeffL","world");

                            }
                        }
                    }else{
                        cout<<"Grasping->simple controller->no obj center"<<endl;
                    }
                }
            }
        }
        cv::namedWindow("Cam");
        //        cv::resizeWindow("Cam",540,400);// Show our image inside it.
        cv::imshow( "Cam", image );
        mlr::wait(.3);//slow down to ensure the accuracy in movement
        if (waitKey(30) >= 0)
            break;
    }
}

int main(int argc, char** argv){
    mlr::initCmdLine(argc, argv);
    Roopi R(false);
    R.setKinematics("../../../data/baxter_model/baxter.ors", true);
    //R.startTweets();
    R.startTaskController();
    R.startRos();
    R.startBaxter();
    //R.getComBaxter().stopSendingMotionToRobot(true);
    {
        auto home = R.home();
        R.wait(+home);
    }

    {
        //----------TEST grasping------------
        if(1){
            grip(&R,true);
            //subscribe to left gripper state
            SubscribeBaxterGripper sbg;
            //subscribe to left hand camera
            SubscribeBaxterHand sbh;
            //open grib for better camera view
            grip(&R, false);
            arr policy = zeros(201,201);//read policy
            readPolicy(policy,"policy.csv");
            startGrasping(policy,R);
        }
        //------------------------------------------

        //test gripper
        if(0){
            grip(&R,true);
            grip(&R,false);
             grip(&R,true);
        }

        //test canny edge detector2
//        if(0){
//            Mat image = imread("ML_prototype/real_obj_from_baxter/cylinder.png",3);
//            Mat edge;
//            cv::Canny(image,edge,50,300);
//            imshow("edge",edge);
//            if(waitKey(0)>0) destroyAllWindows();
//        }

        //test canny edge detector
        if(0){
            Mat image = imread("shape_match/objs/pen.png",3);
            vector<Point> workspace = detectWorkspace(image);
            vector<Point> edges = grab_vision::getEdges(image,workspace);
            for(int i = 0;i<edges.size();i++){
                cv::circle(image,edges.at(i),2,Scalar(255,0,0));
            }
            imshow("img",image);
            if(waitKey(0)>0) destroyAllWindows();

        }

        //TEST predict grasping point
        if(0){
            Mat image = cv::imread("shape_match/objs/hammer.jpg",3);
            Mat  edges_img;
            Canny( image, edges_img, 100, 200);
            edges_img.convertTo(edges_img,CV_8UC1);

            vector<Point> contour = objDetect(image);
            vector<vector<Point>> temp;
            temp.push_back(contour);
            cv::drawContours(image,temp,-1,Scalar(0,255,0),1);
            vector<Point> edges;
            cv::findNonZero(edges_img,edges);
            vector<Point> graspingPoints = grab_shape::predictGraspingPoint(edges,image,.5);
            for(int i =0;i<graspingPoints.size();i++){
                cv::circle(image,graspingPoints.at(i),2,Scalar(0,0,255),2);
            }
            cv::imshow("grasping point",image);
            cv::imshow("canny edge",edges_img);
            if(cv::waitKey(0)>0)
                destroyAllWindows();


        }

        //TEST shape context descripter and predict grasping point
        //our implementation
        if(0){
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
            //            int n_sample = 300;


            //read coef
            vector<double> coef;
            std::ifstream file("coef.txt");
            std::string str;
            while (std::getline(file, str))
            {
                std::stringstream ss(str);
                std::string c;

                while(std::getline(ss, c, ',')) {
                    //                    std::cout << c << '\n';
                    coef.push_back(stof(c));
                }
            }
            cout<<"sigmoid "<<grab_shape::sigmoid(0.)<<endl;

            cout<<"coef "<<coef.size()<<endl;
            Mat coefMat = Mat(coef);

            Mat image = cv::imread("shape_match/objs/thickpencil0008.png",3);
            //                         Mat image = cv::imread("shape_match/objs/circle2.png",3);
            Mat edges_img;
            Canny( image, edges_img, 200, 300);
            edges_img.convertTo(edges_img,CV_8UC1);

            vector<Point> edges;
            cv::findNonZero(edges_img,edges);

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
            cv::rectangle(image,box1,Scalar(0,255,0),1);
            Point2f rect_points[4];

            box2.points(rect_points);
            for( int j = 0; j < 4; j++ ){
                line( image, rect_points[j], rect_points[(j+1)%4], Scalar(0,255,0), 1);
            }
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
                    //                    Rect patchBound = Rect(compensateStart,Size(pSize,pSize));
                    //                    cv::rectangle(image,patchBound,Scalar(0,0,255));
                    vector<Point> inside;
                    cv::findNonZero(patch,inside);
                    cv::Mat accumulated = cv::Mat::zeros(1,n_r*n_theta,shape_context.type());
                    if(inside.size()>0){
                        for(int k=0;k<inside.size();k++){
                            Point insidePoint = inside.at(k)+compensateStart;
                            //                            cv::circle(image,insidePoint,2,Scalar(0,0,255));
                            //find indices of inside in sampled edges and find shape context
                            //TODO
                            vector<int> indices = grab_shape::findIndices(insidePoint,sampled_edges);
                            //                            cout<<"find indices"<<indices.size()<<endl;
                            //accumulate shape context
                            for(int l=0;l<indices.size();l++){
                                //                                cout<<"sc "<<shape_context.row(indices.at(l))<<endl;
                                accumulated += shape_context.row(indices.at(l));


                            }

                        }
                    }else{
                        //non edge points


                    }
                    //                    cout<<"accumulayed "<<accumulated<<endl;
                    descriptors.push_back(accumulated);

                }
                Mat final_descriptor;
                hconcat(descriptors,final_descriptor);
                //                vector<double> copy;
                //                final_descriptor.col(0).copyTo(copy);
                double d = 0;
                //                 cout<<"final "<<final_descriptor<<endl;
                for(int q=0;q<coef.size();q++){
                    //                     cout<<final_descriptor.at<float>(Point(i,0))<<endl;
                    //                     cout<<copy.at(i)<<endl;
                    //                    d += coef.at(q)*final_descriptor.at<double>(Point(i,0));
                }
                //                 final_descriptor.t()
                //                    cout<<coefMat.size<<endl;
                final_descriptor.convertTo(final_descriptor,coefMat.type());
                d = coefMat.t().dot(final_descriptor);
                //                 cout<<final_descriptor.size<<endl;
                //                cout<<"ddd "<<d<<endl;
                double prob = grab_shape::sigmoid(d);
                if(prob>.5){
                    cv::circle(image,center,2,Scalar(0,0,(int)(255*prob)),2);
                }

            }


            while (1) {
                imshow("edges",edges_img);
                imshow("sampled edges",sampled_edges_img);
                imshow("obj",image);
                if(waitKey(0)>0) break;
            }
            cv::destroyAllWindows();
        }

        //TEST stereo matching
        if(0){

            Mat cameraMatrix, distCoeffs;
            float data[] = {   4.0364801365017615e+02, 0., 6.3146928830350316e+02, 0.,
                               4.0620637674934119e+02, 3.6213656430315433e+02, 0., 0., 1.  };
            float data2[] = { 2.5817367285864371e-02, -6.4939093957227453e-02,
                              -1.6997464300167172e-04, -8.7200818996337727e-05,
                              1.8531601637796945e-02 };
            cameraMatrix=cv::Mat(3,3,CV_32F,data);
            distCoeffs=cv::Mat(5,1,CV_32F,data2);
            Size size = Size(1280,800);
            Mat R = Mat::eye(3,3,CV_32F);
            float t[] = {-0.02,0,0};
            Mat T = Mat(3,1,CV_32F,t);



            //-- 1. Read the images
            Mat imgLeft = imread("stereo/img_series/image-59.png" , 3 );
            Mat imgRight = imread( "stereo/img_series/image-57.png", 3 );
            Mat imgLeft2 = imread( "stereo/img_series/image-61.png", 3 );

            //-- And create the image in which we will save our disparities
            //            Mat imgDisparity16S = Mat( imgLeft.rows, imgLeft.cols, CV_16S );
            Mat imgDisparity8U = Mat( imgLeft.rows, imgLeft.cols, CV_8UC1 );
            Mat imgDisparity8U2 = Mat( imgLeft.rows, imgLeft.cols, CV_8UC1 );

            if( imgLeft.empty() || imgRight.empty() )
            { std::cout<< " --(!) Error reading images " << std::endl; return -1; }



            Mat dist,dist2;
            dist = grab_stereo::disparityMap(imgLeft,imgRight);
            dist2 = grab_stereo::disparityMap(imgLeft2,imgLeft,true);
            dist2 = -(dist2+1);
            //            createRightMatcher(sgbm)->compute(imgRight,imgLeft,dist2);
            double minVal, maxVal;
            //                        imgDisparity16S /= 16;

            minMaxLoc( dist, &minVal, &maxVal );

            printf("Min disp: %f Max value: %f \n", minVal, maxVal);

            //-- 4. Display it as a CV_8UC1 image
            dist.convertTo( imgDisparity8U, CV_8UC1, 255/(maxVal - minVal));

            minMaxLoc( dist2, &minVal, &maxVal );
            dist2.convertTo( imgDisparity8U2, CV_8UC1, 255/(maxVal - minVal));

            cout<<"test "<<dist.at<float>(Point(203,211))<<endl;
            Mat _3dImage;
            _3dImage = grab_stereo::stereoMatching((dist+dist2)/2,cameraMatrix,distCoeffs,size,R,T);

            Mat depthMap[3];
            split(_3dImage,depthMap);
            minMaxLoc( depthMap[2], &minVal, &maxVal );
            Point p = Point(600,400);
            circle(imgLeft,p,5,Scalar(255,255,255),2);
            cout<<"x "<<depthMap[0].at<float>(p)<<endl;
            cout<<"y "<<depthMap[1].at<float>(p)<<endl;
            cout<<"z "<<depthMap[2].at<float>(p)<<endl;
            printf("Min disp: %f Max value: %f \n", minVal, maxVal);
            //            imshow("before",imgDisparity8U);
            //            cv::GaussianBlur(imgDisparity8U, imgDisparity8U, cv::Size(9, 9), 3, 3);
            imshow("imgL",imgLeft);
            imshow( "windowDisparity", imgDisparity8U );
            imshow( "windowDisparity2", imgDisparity8U2 );
            imshow("merge",(imgDisparity8U+imgDisparity8U2)/2);
            waitKey(0);

            destroyAllWindows();


        }

        //TEST basictask functions
        if(0){
            cout<<"current position"<<getTransformation(R,"endeffL").pos<<endl;
            cout<<"test add relative translation"<<
                  addRelativeTranslation(R,"endeffL",mlr::Vector(1,0,0))<<endl;
            cout<<"test get transformation"<<getTransformation(R,"endeffL")<<endl;
            cout<<"test pos diff task"<<endl;
            posTask(R,mlr::Vector(.7,.2,1.4),"endeffL","world");
            cout<<"test align task"<<endl;
            alignTask(R,mlr::Vector(1,0,0),mlr::Vector(1,0,0),"endeffL","world");
            cout<<"test pos+align task"<<endl;
            pos_alignTask(R,mlr::Vector(.2,.4,1.2),mlr::Vector(1,0,0),mlr::Vector(1,0,0),
                          "endeffL","world");
            cout<<"test realtive movement of endeffL"<<endl;
            pos_alignTask(R,mlr::Vector(.5,.2,1.2),mlr::Vector(1,0,0),mlr::Vector(0,0,-1),
                          "endeffL","world");
            while(getTransformation(R,"endeffL").pos.z>1.1) performRelativeAction(R,4,true);

        }

        //TEST stereo matching on robot
        if(0){
            cv::Mat cameraFrame,imgL,imgR;
            //subscribe to left hand camera
            SubscribeBaxterHand sbh;
            Access<cv::Mat> images("baxter_left_camera");
            for(;;){
                cameraFrame = images.get()();
                if (cameraFrame.empty()){
                    std::cout << "Failed to load image." << std::endl;
                    mlr::wait(1.);
                    continue;
                }


                grip(&R, false);
                mlr::Vector target = {.65,.3,1.05};
                mlr::Vector target2 = {.63,.3,1.05};
                mlr::Vector z = {0.,-1.,0.};
                mlr::Vector x = {0.,0.,1.};
                graspFixedPos(R,target,z,x);
                cout<<"obv 1"<<endl;
                mlr::wait(2.);
                for(int i =0;i<20;i++)
                    imgL = images.get()();


                graspFixedPos(R,target2,z,x);
                cout<<"obv 2"<<endl;
                mlr::wait(2.);
                for(int i =0;i<20;i++)
                    imgR = images.get()();

                if(!imgL.empty() && !imgR.empty())
                    break;

                imshow("cam",cameraFrame);
                if(waitKey(0)>0)
                    break;

            }
            Mat cameraMatrix, distCoeffs;
            float data[] = {   4.0364801365017615e+02, 0., 6.3146928830350316e+02, 0.,
                               4.0620637674934119e+02, 3.6213656430315433e+02, 0., 0., 1.  };
            float data2[] = { 2.5817367285864371e-02, -6.4939093957227453e-02,
                              -1.6997464300167172e-04, -8.7200818996337727e-05,
                              1.8531601637796945e-02 };
            cameraMatrix=cv::Mat(3,3,CV_32F,data);
            distCoeffs=cv::Mat(5,1,CV_32F,data2);
            Size size = Size(1280,800);
            Mat R = Mat::eye(3,3,CV_32F);
            float t[] = {-0.02,0,0};
            Mat T = Mat(3,1,CV_32F,t);

            Mat disparity = grab_stereo::disparityMap(imgL,imgR);

            //             Mat _3dImage = grab_stereo::stereoMatching(disparity,cameraMatrix,distCoeffs,size,R,T);
            imshow("imgL",imgL);
            imshow("imgR",imgR);

            imwrite( "stereo/baxter/left1.png", imgL );
            imwrite( "stereo/baxter/right1.png", imgR );
            imshow("disparity",disparity);
            //             imshow("imgR",imgR);
            waitKey(0);
            destroyAllWindows();






        }

        //TEST track feature
        //filter feature track by workspace and aspect ratio
        if(0){
            cv::Mat cameraFrame,originFrame;
            //subscribe to left hand camera
            SubscribeBaxterHand sbh;
            Access<cv::Mat> images("baxter_left_camera");
            Mat grayPatch = imread("feature_match/stic-part.png",0);
            Point2f averageFeatureCenter = Point2f(640.,400.);
            for(;;){
                //get image
                cameraFrame = images.get()();

                if (cameraFrame.empty()){
                    std::cout << "Failed to load image." << std::endl;
                    mlr::wait(1.);
                    continue;
                }
                originFrame = cameraFrame.clone();
                cv::circle(cameraFrame,averageFeatureCenter,3,Scalar(0,255,0),2);
                //detect and draw workspace
                vector<Point> workspace = detectWorkspace(cameraFrame);
                vector<vector<Point>> temp;
                temp.push_back(workspace);
                cv::drawContours(cameraFrame,temp,0,Scalar(0,255,0),2);

                //detect obj
                vector<Point> obj = objDetect(cameraFrame);
                objPos2grid2(cameraFrame,obj);

                //choose grasping box
                if(obj.size()>0){
                    //choose multiple times!!!

                    //                    for(int i =0;i<5;i++){

                    //                    }
                    RotatedRect graspingBox = chooseGraspingRect(obj,40);
                    graspingBox.center;
                    Point2f rect_points[4];

                    //                    //display grasping box angle for wrist aligning
                    //                    if(graspingBox.size.width < graspingBox.size.height){
                    //                        cout<<"rect angle"<<graspingBox.angle+180<<endl;

                    //                       }else{
                    //                          cout<<"rect angle"<<graspingBox.angle+90<<endl;
                    //                       }

                    graspingBox.points( rect_points );
                    for( int j = 0; j < 4; j++ ){
                        cv::line( cameraFrame, rect_points[j], rect_points[(j+1)%4], Scalar(255,255,255), 2, 2 );
                    }

                    //                    float a = 1.0;
                    //                    Size2f newSize = Size2f(graspingBox.size.width*a,graspingBox.size.height*a);
                    //                    RotatedRect enlarged = RotatedRect(graspingBox.center,newSize,graspingBox.angle);

                    //                    Mat partOfImage = cutRotatedRect(originFrame,enlarged);
                    //                cv::cvtColor(partOfImage, grayPatch, cv::COLOR_BGR2GRAY);
                    //                imshow("feature patch",partOfImage);


                }

                if(!grayPatch.empty()){
                    vector<Point2f> corners;
                    corners = trackFeature(originFrame,grayPatch,workspace);
                    if(corners.at(0).x!=-1){
                        RotatedRect box = cv::minAreaRect(corners);
                        cout<<"box w"<<box.size.width<<endl
                           <<"box h" <<box.size.height<<endl;
                        float aspectRatio;
                        if(box.size.width>box.size.height)
                            aspectRatio = box.size.width/box.size.height;
                        else
                            aspectRatio = box.size.height/box.size.width;

                        //replace 2 with a distance metric between aspectRation of grayPatch and
                        //aspectRation of feature patch
                        if(aspectRatio<4){
                            Point2f center =  (corners.at(0)+ corners.at(1)+ corners.at(2)+ corners.at(3))/4.;
                            averageFeatureCenter = .95*averageFeatureCenter+.05*center;
                        }
                    }
                }

                imshow("cam",cameraFrame);


                if(waitKey(30)>=0)
                    break;
            }
            destroyAllWindows();
        }

        //TEST align with grasping box
        if(0){
            cv::Mat cameraFrame;
            //subscribe to left hand camera
            SubscribeBaxterHand sbh;
            Access<cv::Mat> images("baxter_left_camera");
            for(;;){
                cameraFrame = images.get()();
                if (cameraFrame.empty()){
                    std::cout << "Failed to load image." << std::endl;
                    mlr::wait(1.);
                    continue;
                }

                vector<Point> obj = objDetect(cameraFrame);
                vector<vector<Point>> temp;
                temp.push_back(obj);
                cv::drawContours(cameraFrame,temp,0,Scalar(0,0,255),2);
                if(obj.size()>0){
                    //choose grasping box
                    RotatedRect graspingBox = chooseGraspingRect(obj,20);
                    //twist wrist given rect
                    twistWrist2(R,graspingBox);

                }
                imshow("cam",cameraFrame);


                if(waitKey(30)>=0)
                    break;

            }
            destroyAllWindows();
        }

        //TEST kmean clustering
        // used for contour split
        if(0){
            //test distance function
            Point a = Point2f(100,100);
            Point b = Point2f(201,20);
            Point c = Point2f(300,100);

            double d = computeDistance(a,b);
            cout<<"distance "<< d <<endl;

            //test
            vector<Point> cluster;
            cluster.push_back(a);
            cluster.push_back(b);
            cluster.push_back(c);
            cout<<"mean test 1"<<meanOfCluster(cluster)<<endl;
            //            cluster.clear();
            cout<<"mean test 2 "<<meanOfCluster(cluster)<<endl;

            //test assign2center
            vector<Point> center;
            center.push_back(Point(100,99));
            center.push_back(Point(200,50));
            vector<vector<Point>> newClusters = assign2center(center,cluster);
            cout<<"new cluster "<<newClusters.at(0)<<endl;
            cout<<"new cluster "<<newClusters.at(1)<<endl;
            cluster = randomInit(cluster,200);
            cout<<"knn test "<<splitContour(cluster,2).size()<<endl;

        }

        //TEST grasp obj at fixed pos and ori
        //grip&move strategy
        if(0){
            SubscribeBaxterGripper sbg;
            grip(&R, false);
            mlr::Vector target = {.65,.3,1.05};
            mlr::Vector z = {0.,-1.,0.};
            mlr::Vector x = {0.,0.,1.};
            graspFixedPos(R,target,z,x);
            for(int i =0;i<5;i++){
                //                performRelativeAction(R,4);
            }
            grip(&R, true);
            //check gripper
            if(checkGripper(R,15))
                cout<<"Grip success"<<endl;
            else{
                cout<<"Nothing in gripper"<<endl;
                grip(&R, false);
            }
            int steps = 3;
            while(!checkGripper(R,10.))
                for(int i=0;i<steps;i++)
                    performRelativeAction(R,4,false);
            cout<<"Finish grasp"<<endl;


        }

        //TEST subscribe gripper
        if(0){
            grip(&R,false);
            mlr::wait(1.);
            grip(&R,true);
            SubscribeBaxterGripper sbg;
            for (int i = 0; i < 20; i++)
            {
                mlr::wait(.5);
                checkGripper(R,.05);
            }

        }

        //TEST read policy and web cam
        if(0){
            arr policy = zeros(201,201);
            readPolicy(policy,"policy.csv");
            webCamStart(policy,R);
        }

        //TEST record video
        if(0){
            //*.avi
            //recordVideo("out2.avi");
            recordVideoFromRobot("test.avi");
        }

        mlr::wait();
    }
    return 0;
}

