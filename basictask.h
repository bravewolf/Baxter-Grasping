#ifndef BASICTASK_H
#define BASICTASK_H

#include <Perception/opencv.h>
#include <opencv2/opencv.hpp>
#include <Core/array.h>
#include <Core/thread.h>
#include <Roopi/roopi.h>
#include <Control/taskControl.h>
#include "datatype.h"

///implementation of posDiff, vecDiff, grip
namespace basictask{
///compute position of shape in world frame after adding a relative translation to it
mlr::Vector addRelativeTranslation(Roopi& R,char* shapeName,mlr::Vector translation);

///return transforamtion of shape
mlr::Transformation getTransformation(Roopi& R,char* shapeName);

///perform posDiffTMT
int posTask(Roopi& R,mlr::Vector pos,char* shapeName, char* frameName);

///perform vecDiffTMT
int alignTask(Roopi& R,mlr::Vector vec1,mlr::Vector vec2,char* shapeName,char* frameName);

///perform posDiffTMT and vecDiffTMT
int pos_alignTask(Roopi& R,mlr::Vector pos,
                  mlr::Vector vec1,mlr::Vector vec2,
                  char* shapeName,char* frameName);

///perform posDiffTMT, vecDiffTMT and vecDiffTMT
int pos_align_alignTask(Roopi& R,mlr::Vector pos,
                        mlr::Vector vec11,mlr::Vector vec12,
                        mlr::Vector vec21,mlr::Vector vec22,
                        char* shapeName,char* frameName);
///open/close gripper
int grip(Roopi* r, const bool toGrip);

///perform relative movement in endeff frame
int performRelativeAction(Roopi& R,int action,bool keepVertical, double customStepSize=NULL);

///simple controller for fine adjusting gripper position to obj
int simpleController(Index index,Index center, int margin);

///grasp a obj given marker position and align vector
///from above
int graspFixedPos(Roopi& R,mlr::Vector pos,mlr::Vector z,mlr::Vector x);

///twist wrist given rotated rect
int twistWrist2(Roopi& R,cv::RotatedRect rect);
}


#endif // BASICTASK

