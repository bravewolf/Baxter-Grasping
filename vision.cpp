#include "basictask.h"
#include "vision.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/aruco.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "datatype.h"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/ximgproc.hpp"

using namespace basictask;
using namespace cv::ximgproc;
using namespace cv::xfeatures2d;
namespace grab_vision{

///convert cv::Mat of size 3x1 to mlr::Vector
mlr::Vector mat2vector(cv::Mat mat){
    return mlr::Vector(mat.at<float>(0,0),mat.at<float>(1,0),mat.at<float>(2,0));
}

/**
 * @brief drawContour, shortcut to draw contour on image
 * @param image
 * @param contour,contour of object
 */
void drawContour(Mat image, vector<Point> contour){
    vector<vector<Point>> contours;
    contours.push_back(contour);
    cv::drawContours(image,contours,0,cv::Scalar(0,0,255),2);
}

/**
 * @brief cutImage, cut rect(not rotated) from image
 * @param image
 * @param x, center of rect
 * @param y
 * @param width
 * @param height
 * @param maxWidth, max width of image
 * @param maxHeight
 * @return
 */
Mat cutImage(Mat image,int x,int y,int width,int height,int maxWidth, int maxHeight){
    return Mat(image,Rect(min(max(x-width/2,0),maxWidth-width),min(max(y-height/2,0),maxHeight-height),width,height)).clone();
}


/**
 * @brief cutRotatedRect, cut a rotated rect from image
 * @param image
 * @param rect, rotatedRect
 * @return
 */
Mat cutRotatedRect(Mat image, RotatedRect rect){
    Mat M, rotated, cropped;
    // get angle and size from the bounding box
    float angle = rect.angle;
    Size rect_size = rect.size;
    // thanks to http://felix.abecassis.me/2011/10/opencv-rotation-deskewing/
    if (rect.angle < -45.) {
        angle += 90.0;
        swap(rect_size.width, rect_size.height);
    }
    // get the rotation matrix
    M = getRotationMatrix2D(rect.center, angle, 1.0);
    // perform the affine transformation
    warpAffine(image, rotated, M, image.size(), INTER_CUBIC);
    // crop the resulting image
    getRectSubPix(rotated, rect_size, rect.center, cropped);
    return cropped;
}

///record video from webcam on laptop
void recordVideo(string filename){
    VideoCapture stream1(0);   //0 is the id of video device.0 if you have only one camera.
    if (!stream1.isOpened()) { //check if video device has been initialised
        cout << "cannot open camera";
    }
    int frame_width = stream1.get(CV_CAP_PROP_FRAME_WIDTH);
    int frame_height = stream1.get(CV_CAP_PROP_FRAME_HEIGHT);
    VideoWriter video(filename,CV_FOURCC('M','J','P','G'),10, Size(frame_width,frame_height),true);
    while(true){
        cv::Mat cameraFrame;
        stream1.read(cameraFrame);
        video.write(cameraFrame);
        imshow( "Cam", cameraFrame );
        char c = (char)waitKey(33);
        if( c == 27 ) break;
    }
}

/**
 * @brief detectWorkspace,return convex hull of work space
 * @param cameraFrame
 * @return
 */
vector<Point> detectWorkspace(cv::Mat cameraFrame){
    cv::Mat hsv_img;
    cv::cvtColor(cameraFrame, hsv_img, cv::COLOR_BGR2HSV);
    cv::Mat blue_hue_image,rest,thres,gray;
    cv::inRange(hsv_img, cv::Scalar(80, 100,100 ), cv::Scalar(130, 255, 255), blue_hue_image);
    cv::bitwise_and(cameraFrame,cameraFrame,rest,blue_hue_image);//modify here to use differnt filter
    /// Convert image to gray
    cv::cvtColor(rest, gray, COLOR_BGR2GRAY );
    cv::threshold(gray,thres,60,255,cv::THRESH_BINARY);//convert to binary image 0 or 255
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    cv::findContours(thres, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

    /// Find the convex hull object for each contour
    vector<vector<Point> >hull( contours.size());
    for( int i = 0; i < contours.size(); i++ ){
        convexHull( Mat(contours[i]), hull[i], false );
    }

    ///find the convex hull with maximal area which is larger than the threshold
    double maxArea = 5000;//needs to modify here to adapt to different camera !!!
    int maxIndex=-1;
    for( int i = 0; i< contours.size(); i++ )
    {
        if(cv::contourArea(contours[i])>maxArea){
            maxIndex = i;
            maxArea = cv::contourArea(contours[i]);
        }
    }
    if(maxIndex!=-1)
        return hull[maxIndex];
    else{
        vector<Point> empty;
        return empty;//return empty vector
    }
}

///use background substraction to detect object in work space
/**
 * @brief objDetect
 * @param cameraFrame
 * @return
 */
vector<Point> objDetect(cv::Mat cameraFrame){
    //    cv::Mat imageCopy = cameraFrame.clone();
    cv::Mat hsv_img;
    cv::cvtColor(cameraFrame, hsv_img, cv::COLOR_BGR2HSV);
    cv::Mat blue_hue_image,black_hue_image,rest,rest3,thres,thres2,thres3,thres4,gray,gray3;

    //change work space color here !!!
    //    cv::inRange(hsv_img, cv::Scalar(0, 0,0 ), cv::Scalar(0, 0, 255), blue_hue_image);
    cv::inRange(hsv_img, cv::Scalar(80, 100,100 ), cv::Scalar(130, 255, 255), blue_hue_image);//blue mask
    cv::inRange(hsv_img, cv::Scalar(0, 0, 0, 0), cv::Scalar(180, 255, 60, 0), black_hue_image);//black mask
    cv::Mat mask_combined = blue_hue_image | black_hue_image;
    //    not_blue = 255-blue_hue_image;//reflect mask
    //    cv::imshow("blue mask",blue_hue_image);
    //    cv::imshow("black mask",black_hue_image);
    //    cv::imshow("combined mask",mask_combined);

    //    cv::GaussianBlur(blue_hue_image, blue_hue_image, cv::Size(9, 9), 2, 2);
    cv::bitwise_and(cameraFrame,cameraFrame,rest,blue_hue_image);//modify here to use differnt filter
    cv::bitwise_and(cameraFrame,cameraFrame,rest3,mask_combined);
    //    imshow("rest 3",rest3);
    /// Convert image to gray
    cv::cvtColor(rest, gray, COLOR_BGR2GRAY );
    cv::cvtColor(rest3, gray3, COLOR_BGR2GRAY );
    //    cv::cvtColor(rest2,gray2, COLOR_BGR2GRAY);

    cv::threshold(gray,thres,60,255,cv::THRESH_BINARY);//convert to binary image 0 or 255
    cv::threshold(gray3,thres3,5,255,cv::THRESH_BINARY);//convert to binary image 0 or 255
    //    cv::threshold(gray2,thres2,60,255,cv::THRESH_BINARY);
    //    cv::bitwise_not(thres,thres2);
    cv::bitwise_not(thres3,thres4);//not blue and black
    //    imshow("blue and black",thres3);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    cv::findContours(thres, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

    /// Find the convex hull object for each contour
    vector<vector<Point> >hull( contours.size());
    for( int i = 0; i < contours.size(); i++ ){
        convexHull( Mat(contours[i]), hull[i], false );
    }

    ///find the convex hull with maximal area which is larger than the threshold
    double maxArea = 5000;//needs to modify here to adapt to different camera !!!
    int maxIndex=-1;
    for( int i = 0; i< contours.size(); i++ )
    {
        if(cv::contourArea(contours[i])>maxArea){
            maxIndex = i;
            maxArea = cv::contourArea(contours[i]);
        }
    }
    //detect object inside workspace
    if(maxIndex!=-1){
        //        cv::drawContours(cameraFrame,hull,maxIndex,Scalar(0,0,255),2);
        cv::Mat blackImage = cv::Mat::zeros(cameraFrame.rows,cameraFrame.cols,cameraFrame.type());
        vector<vector<Point>> p;
        p.push_back(hull[maxIndex]);
        cv::fillPoly(blackImage,p,Scalar(255,255,255));
        cv::cvtColor(blackImage, blackImage, COLOR_BGR2GRAY );
        cv::bitwise_and(thres4,blackImage,blackImage);

        //morphology tranformation
        Mat kernel = Mat::ones(5,5,CV_32F);
        cv::morphologyEx(blackImage,blackImage,cv::MORPH_ERODE,kernel);
        //        cv::imshow("Background substraction",blackImage);
        vector<vector<Point>> c;
        vector<Vec4i> h;
        cv::findContours(blackImage, c, h, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
        int maxArea2 = 500;//threshold for minimal size of obj
        int maxIndex2 = -1;
        for( int i = 0; i< c.size(); i++ )
        {
            if(cv::contourArea(c[i])>maxArea2){
                maxIndex2 = i;
                maxArea2 = cv::contourArea(c[i]);
            }
        }
        if(maxIndex2!=-1) {
            vector<Point> polygon;
            vector<vector<Point>> polygons;
            polygons.push_back(polygon);
            cv::approxPolyDP(c[maxIndex2],polygons.at(0),.5,true);
            //            cv::drawContours(cameraFrame,polygons,0,Scalar(0,255,0),2);
            return polygons.at(0);


        }
        //return empty vector
        vector<Point> empty;
        return  empty;
    }else{
        //return empty vector
        vector<Point> empty;
        return  empty;
    }


}

/**
 * @brief objDetect2, similar to objDetect, but return contours of all objects inside workspace
 * @param cameraFrame
 * @return
 */
vector<vector<Point>> objDetect2(cv::Mat cameraFrame){
    //    cv::Mat imageCopy = cameraFrame.clone();
    cv::Mat hsv_img;
    cv::cvtColor(cameraFrame, hsv_img, cv::COLOR_BGR2HSV);
    cv::Mat blue_hue_image,black_hue_image,rest,rest3,thres,thres2,thres3,thres4,gray,gray3;

    //change work space color here !!!
    //    cv::inRange(hsv_img, cv::Scalar(0, 0,0 ), cv::Scalar(0, 0, 255), blue_hue_image);
    cv::inRange(hsv_img, cv::Scalar(80, 100,100 ), cv::Scalar(130, 255, 255), blue_hue_image);
    cv::inRange(hsv_img, cv::Scalar(0, 0, 0, 0), cv::Scalar(180, 255, 60, 0), black_hue_image);
    cv::Mat mask_combined = blue_hue_image | black_hue_image;

    //    cv::GaussianBlur(blue_hue_image, blue_hue_image, cv::Size(9, 9), 2, 2);
    cv::bitwise_and(cameraFrame,cameraFrame,rest,blue_hue_image);//modify here to use differnt filter
    cv::bitwise_and(cameraFrame,cameraFrame,rest3,mask_combined);
    /// Convert image to gray
    cv::cvtColor(rest, gray, COLOR_BGR2GRAY );
    cv::cvtColor(rest3, gray3, COLOR_BGR2GRAY );

    cv::threshold(gray,thres,60,255,cv::THRESH_BINARY);//convert to binary image 0 or 255
    cv::threshold(gray3,thres3,5,255,cv::THRESH_BINARY);//convert to binary image 0 or 255
    cv::bitwise_not(thres3,thres4);//not blue and black
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    cv::findContours(thres, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

    // Find the convex hull object for each contour
    vector<vector<Point> >hull( contours.size());
    for( int i = 0; i < contours.size(); i++ ){
        convexHull( Mat(contours[i]), hull[i], false );
    }

    //find the convex hull with maximal area which is larger than the threshold
    double maxArea = 5000;//needs to modify here to adapt to different camera !!!
    int maxIndex=-1;
    for( int i = 0; i< contours.size(); i++ )
    {
        if(cv::contourArea(contours[i])>maxArea){
            maxIndex = i;
            maxArea = cv::contourArea(contours[i]);
        }
    }

    //detect workspace
    if(maxIndex!=-1){
        cv::Mat blackImage = cv::Mat::zeros(cameraFrame.rows,cameraFrame.cols,cameraFrame.type());
        vector<vector<Point>> p;
        p.push_back(hull[maxIndex]);
        cv::fillPoly(blackImage,p,Scalar(255,255,255));
        cv::cvtColor(blackImage, blackImage, COLOR_BGR2GRAY );
        cv::bitwise_and(thres4,blackImage,blackImage);

        //morphology tranformation
        Mat kernel = Mat::ones(5,5,CV_32F);
        cv::morphologyEx(blackImage,blackImage,cv::MORPH_ERODE,kernel);

        vector<vector<Point>> c;
        vector<Vec4i> h;
        cv::findContours(blackImage, c, h, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

        if(c.size()>0){
            vector<vector<Point>> polygons( c.size());
            for( int i = 0; i < polygons.size(); i++ ){
                cv::approxPolyDP(c[i],polygons[i],.5,true);
            }
            //draw contour
            //            vector<Point> polygon;
            //            vector<vector<Point>> polygons;
            //            polygons.push_back(polygon);
            //            cv::approxPolyDP(c[maxIndex2],polygons.at(0),.5,true);
            //            cv::drawContours(cameraFrame,polygons,0,Scalar(0,255,0),2);
            return polygons;
        }

        //return empty vector
        vector<vector<Point>> empty;
        return  empty;

    }else{
        //return empty vector
        vector<vector<Point>> empty;
        return  empty;
    }


}


/**
 * @brief objPos2grid2,map object in camera image to grid index (0-201,0-201)
 * @param cameraFrame
 * @param obj,object contour
 * @return
 */
Index objPos2grid2(Mat cameraFrame,vector<Point> obj){
    //    vector<Point> obj = objDetect(cameraFrame);

    ///color of drawing
    Scalar red = Scalar( 0, 0, 255 );
    Scalar green = Scalar(0,255,0);
    Scalar blue = Scalar(255,0,0);

    float c1 = cameraFrame.cols/2.;//center of image
    float c2 = cameraFrame.rows/2.;
    Point2f centerOfImage = Point2f(c1,c2);//center of image
    circle( cameraFrame, centerOfImage, 4,red, -1, 8, 0 );//draw

    if(obj.size()>0){
        //draw contour of obj
        //        vector<vector<Point>> temp;
        //        temp.push_back(obj);
        //        cv::drawContours(cameraFrame,temp,0,green,2);

        ///mapping from image to grid index
        Moments mu = moments( obj, false );
        Point2f mc = Point2f( mu.m10/mu.m00 , mu.m01/mu.m00 );

        int x = (int)(centerOfImage.x-mc.x)/(float)centerOfImage.x*100+100;
        int y  = (int)(centerOfImage.y-mc.y)/(float)centerOfImage.y*100+100;
        Index index = {x,y};
        cout<<"Index "<<x<<","<<y<<endl;
        circle( cameraFrame, mc, 4,red, -1, 8, 0 );//draw
        return index;
    }else{
        return Index{-1,-1};
    }
}

/**
 * @brief trackFeature,track grasping patch by SURF/SIFT
 * @param cameraFrame, , input image should be grey scale
 * @param queryImg, input image should be grey scale
 * @param workspace, filter feature point by workspace
 * @return
 */
vector<Point2f> trackFeature(cv::Mat cameraFrame,cv::Mat queryImg,vector<Point> workspace){
    vector<Point2f> w;
    for(int i=0;i<workspace.size();i++){
        w.push_back(Point2f(workspace.at(i).x,workspace.at(i).y));
    }
    Mat img_object = queryImg.clone();
    Mat img_scene = cameraFrame.clone();


    //-- Step 1: Detect the keypoints and compute descriptor using SURF/SIFT Detector
    //SURF
    //    int minHessian = 100;
    //    Ptr<SURF> detector = SURF::create( minHessian );
    //SIFT
    int 	nfeatures = 0;
    int 	nOctaveLayers = 3;//default 3
    double 	contrastThreshold = 0.08;//default 0.04
    double 	edgeThreshold = 8;//default 10
    double 	sigma = 1. ;//default 1.6
    Ptr<SIFT> detector = SIFT::create(nfeatures,nOctaveLayers,contrastThreshold,edgeThreshold,sigma);

    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;
    detector->detectAndCompute( img_object, Mat(), keypoints_object, descriptors_object );
    detector->detectAndCompute( img_scene, Mat(), keypoints_scene, descriptors_scene );

    //-- Step 2: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_object, descriptors_scene, matches );
    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_object.rows; i++ )
    { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< DMatch > good_matches;
    for( int i = 0; i < descriptors_object.rows; i++ )
    { if( matches[i].distance <= 3*min_dist )
        {
            //only add good match which is inside workspace
            if(w.size()>0)
                if(cv::pointPolygonTest(w,keypoints_scene[ matches[i].trainIdx ].pt,false)>=0)
                {good_matches.push_back( matches[i]);}

        }
    }

    Mat img_matches;
    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        //        if(cv::pointPolygonTest(w,keypoints_scene[ good_matches[i].trainIdx ].pt,false)>=0){
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
        //        }
    }

    ///little bug here, may need size check of obj and scene
    Mat H;
    if(scene.size()>0 && obj.size()>0){
        H = findHomography( obj, scene, RANSAC );
    }
    if(!H.empty()){
        //-- Get the corners from the image_1 ( the object to be "detected" )
        std::vector<Point2f> obj_corners(4);
        obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
        obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
        std::vector<Point2f> scene_corners(4);
        perspectiveTransform( obj_corners, scene_corners, H);
        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
        line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
        Point2f center = (scene_corners[0] + Point2f( img_object.cols, 0)+
                scene_corners[1] + Point2f( img_object.cols, 0)+
                scene_corners[2] + Point2f( img_object.cols, 0)+
                scene_corners[3] + Point2f( img_object.cols, 0))/4;
        //--Draw center
        circle(img_matches,center,20,Scalar(255,0,0),5);
        imshow( "Good Matches & Object detection", img_matches );
        vector<Point2f> corners;
        corners.push_back(scene_corners[0]);
        corners.push_back(scene_corners[1]);
        corners.push_back(scene_corners[2]);
        corners.push_back(scene_corners[3]);
        return corners;
    }else{
        vector<Point2f> corners;
        corners.push_back(Point2f(-1,-1));
        return corners;
    }
    // -- Show detected matches
    //             imshow( "Good Matches & Object detection", img_matches );
    //           cameraFrame = img_matches;



}

///return pixel position of centroid of grasping label, not used
cv::Point2i trackLabel(cv::Mat cameraFrame){
    cv::Mat hsv_img;
    cv::cvtColor(cameraFrame, hsv_img, cv::COLOR_BGR2HSV);
    cv::Mat blue,rest,thres,gray;

    ///generate green mask
    cv::inRange(hsv_img, cv::Scalar(40, 100,100 ), cv::Scalar(80, 255, 255), blue);
    cv::GaussianBlur(blue, blue, cv::Size(9, 9), 2, 2);
    cv::bitwise_and(cameraFrame,cameraFrame,rest,blue);//modify here to use differnt filter

    /// Convert image to gray
    cv::cvtColor(rest, gray, COLOR_BGR2GRAY );
    cv::threshold(gray,thres,60,255,cv::THRESH_BINARY);//convert to binary image 0 or 255

    /// Find the convex hull object for each contour
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    cv::findContours(thres, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
    vector<vector<Point> >hull( contours.size() );
    for( int i = 0; i < contours.size(); i++ ) {
        convexHull( Mat(contours[i]), hull[i], false );
    }

    ///find the convex hull with maximal area which is larger than the threshold
    double maxArea = 1200;//needs to modify here to adapt to different camera !!!
    int maxIndex=-1;
    for( int i = 0; i< contours.size(); i++ )
    {
        if(cv::contourArea(contours[i])>maxArea){
            maxIndex = i;
            maxArea = cv::contourArea(contours[i]);
        }
    }

    if(maxIndex!=-1){
        ///color of drawing
        Scalar color = Scalar( 0, 0, 255 );

        ///draw convex hull
        drawContours( cameraFrame, hull, maxIndex, color, 2, 8, vector<Vec4i>(), 0, Point() );

        ///draw centroid of convex hull
        Moments mu = moments( contours[maxIndex], false );
        Point2f mc = Point2f( mu.m10/mu.m00 , mu.m01/mu.m00 );
        circle( cameraFrame, mc, 4,color, -1, 8, 0 );
        RotatedRect boundRect = minAreaRect( Mat(hull[maxIndex]) );
        // rotated rectangle
        Point2f rect_points[4];
        //cout<<"rect angle"<<boundRect.angle<<endl;
        boundRect.points( rect_points );
        for( int j = 0; j < 4; j++ ){
            line( cameraFrame, rect_points[j], rect_points[(j+1)%4], color, 2, 8 );
        }

        ///fit line to convex hull of object
        Vec4f line;
        fitLine(contours[maxIndex],line,CV_DIST_L2,0,0.01,0.01);
        int cols = cameraFrame.cols;
        int lefty = (int)((-line[2] * line[1] / line[0]) + line[3]);
        int righty = (int)(((cols - line[2]) * line[1] / line[0]) + line[3]);
        Point2f start(cols-1,righty);
        Point2f end(0.,lefty);
        cv::line(cameraFrame,start,end,color,2);
        return Point2i((int)mc.x,(int)mc.y);
    }else{
        return Point2i(-1,-1);
    }
}

/**
 * @brief getEdges,canny edge detector, edge points is filtered by workspace
 * @param image
 * @param workspace, contour of workspace
 * @param p1, params for canny edge detector
 * @param p2
 * @return
 */
vector<Point> getEdges(Mat image, vector<Point> workspace, double p1, double p2){
    Mat edgeImage;
    Canny(image,edgeImage,p1,p2);
    vector<Point> edges,filtered_edges;
    findNonZero(edgeImage,edges);

    //filter edges by worksapce
    for(int i=0;i<edges.size();i++){
        if(workspace.size()>0){
            if(pointPolygonTest(workspace,Point2f(edges.at(i).x,edges.at(i).y),false)>0){
                filtered_edges.push_back(edges.at(i));
            }
        }else{
            filtered_edges = edges;
        }
    }
    return filtered_edges;
}

/**
 * @brief detectMarker, detect id,position,x,y,z of each marker as a list
 * @param R
 * @param cameraFrame
 * @return
 */
vector<MarkerData> detectMarker(Roopi& R,Mat cameraFrame){
    ///calibration matrix
    Mat cameraMatrix, distCoeffs;//camera params
    float data[] = {   4.0364801365017615e+02, 0., 6.3146928830350316e+02, 0.,
                       4.0620637674934119e+02, 3.6213656430315433e+02, 0., 0., 1.  };
    float data2[] = { 2.5817367285864371e-02, -6.4939093957227453e-02,
                      -1.6997464300167172e-04, -8.7200818996337727e-05,
                      1.8531601637796945e-02 };
    cameraMatrix=cv::Mat(3,3,CV_32F,data);
    distCoeffs=cv::Mat(5,1,CV_32F,data2);

    ///detect marker
    vector< int > markerIds;
    vector< vector<Point2f> > markerCorners,rejectedCandidates;
    Ptr<cv::aruco::DetectorParameters> parameters=cv::aruco::DetectorParameters::create();
    Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::detectMarkers(cameraFrame,dictionary,markerCorners,markerIds,parameters,rejectedCandidates,cameraMatrix,distCoeffs);
    cv::aruco::drawDetectedMarkers(cameraFrame,markerCorners,markerIds);

    ///estimate marker pose
    vector< Vec3d > rvecs, tvecs;
    cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.02, cameraMatrix, distCoeffs, rvecs, tvecs);

    ///get marker data
    vector<MarkerData> markerDataList;
    if(markerIds.size()>0){
        for(int i=0;i<rvecs.size();i++){
            ///draw marker axis
            cv::aruco::drawAxis(cameraFrame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.01);

            cv::Mat camera2marker = cv::Mat::zeros(3,3,CV_32FC1);//convert marker coordinate to camera coordinate
            cv::Rodrigues(rvecs[i],camera2marker);
            camera2marker.convertTo(camera2marker,CV_32FC1);

            //compute position of marker in world frame
            float tvec[3][1] = {{tvecs[i][0]},{tvecs[i][1]},{tvecs[i][2]}};
            cv::Mat translation = cv::Mat(3,1,CV_32FC1,tvec);
            float origin[3][1] = {{0.},{.0},{0.2}};//30cm in camera frame. while it is 20cm to gripper
            cv::Mat originOfMarker = cv::Mat(3,1,CV_32FC1,origin);
            cv::Mat posInCamera = camera2marker*originOfMarker+translation;
            mlr::Vector posInGripper = mlr::Vector(posInCamera.at<float>(2,0),-posInCamera.at<float>(0,0),-posInCamera.at<float>(1,0));
            mlr::Vector posInWorld = addRelativeTranslation(R,"endeffL",posInGripper);

            //compute x,y,z axis of marker in world frame
            float vec1[3][1] = {{1.},{.0},{.0}};//track x axis
            float vec2[3][1] = {{.0},{1.},{.0}};//track y axis
            float vec3[3][1] = {{0.},{.0},{1.}};//track z axis
            cv::Mat x = cv::Mat(3,1,CV_32FC1,vec1);
            cv::Mat y = cv::Mat(3,1,CV_32FC1,vec2);
            cv::Mat z = cv::Mat(3,1,CV_32FC1,vec3);
            mlr::Matrix rot = getTransformation(R,"endeffL").rot.getMatrix();
            float rotMat[3][3]= {{rot.m00,rot.m01,rot.m02},
                                 {rot.m10,rot.m11,rot.m12},
                                 {rot.m20,rot.m21,rot.m22}};
            cv::Mat endeff2world = cv::Mat(3,3,CV_32FC1,rotMat);
            Mat world2marker = endeff2world.inv()*camera2marker;

            ///camera frame is different with endeff frame
            ///this rotation mat will correct it
            float rotCorrection[3][3] = {{0,0,-1},
                                         {0,1,0},
                                         {1,0,0}};
            Mat correctionMat = cv::Mat(3,3,CV_32FC1,rotCorrection);//convert camera coordinate to gripper coordinate

            mlr::Vector xInWorld = mat2vector(correctionMat*world2marker*x);
            mlr::Vector yInWorld = mat2vector(correctionMat*world2marker*y);
            mlr::Vector zInWorld = mat2vector(correctionMat*world2marker*z);

            yInWorld.normalize();
            xInWorld.normalize();
            zInWorld.normalize();

            //center of marker in image
            //used for match obj contour with marker
            Point2f centerOfMarker = (markerCorners[i][0]+markerCorners[i][1]+markerCorners[i][2]+markerCorners[i][3])/4;
//            cv::circle(cameraFrame,centerOfMarker,5,cv::Scalar(255,0,0),2);
            MarkerData marker = MarkerData{markerIds[i],posInWorld,xInWorld,yInWorld,zInWorld,centerOfMarker};
            markerDataList.push_back(marker);

        }

    }
    else{
        //return default marker
        mlr::Vector defaultVec = {0.,0.,0.};
        MarkerData marker = MarkerData{-1,defaultVec,defaultVec,defaultVec,defaultVec};
        markerDataList.push_back(marker);
    }
    return markerDataList;
}

/**
 * @brief updateMarkerDate, averaging marker position
 * @param oldMarkerDate
 * @param newMarkerData
 * @param alpha averaging params
 * @param ignoreId, update marker position even marker id is different
 * @return
 */
MarkerData updateMarkerDate(MarkerData oldMarkerDate,MarkerData newMarkerData, double alpha,bool ignoreId){
    if(oldMarkerDate.id == newMarkerData.id || ignoreId){

        int id = oldMarkerDate.id;
        mlr::Vector averagePos = oldMarkerDate.pos;
        mlr::Vector averageX = oldMarkerDate.x;
        mlr::Vector averageY = oldMarkerDate.y;
        mlr::Vector averageZ = oldMarkerDate.z;

        averagePos = (alpha * newMarkerData.pos) + (1.0 - alpha) * averagePos;
        averageX = (alpha * newMarkerData.x) + (1.0 - alpha) * averageX;
        averageY = (alpha * newMarkerData.y) + (1.0 - alpha) * averageY;
        averageZ = (alpha * newMarkerData.z) + (1.0 - alpha) * averageZ;

        return MarkerData{id,averagePos,averageX,averageY,averageZ};
    }else{
        cout<<"Cann't update different marker"<<endl;
        return oldMarkerDate;
    }

}



}
