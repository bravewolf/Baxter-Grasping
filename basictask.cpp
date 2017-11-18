#include <Perception/opencv.h>
#include <opencv2/opencv.hpp>
#include <Core/array.h>
#include <Core/thread.h>
#include <Roopi/roopi.h>
#include <Control/taskControl.h>
#include "basictask.h"

namespace basictask{

///return position vector of shape after adding translation
mlr::Vector addRelativeTranslation(Roopi& R,char* shapeName,mlr::Vector translation){
    R.getK()->getShapeByName(shapeName)->X.addRelativeTranslation(translation);
    return R.getK()->getShapeByName(shapeName)->X.pos;
}

///return transforamtion of shape
mlr::Transformation getTransformation(Roopi& R,char* shapeName){
    return R.getK()->getShapeByName(shapeName)->X;
}

///perform posDiffTMT
int posTask(Roopi& R,mlr::Vector pos,char* shapeName, char* frameName){
    if(strcmp(frameName,"world")==0) {
        auto posDiffTask = R.newCtrlTask(new TaskMap_Default(posDiffTMT, R.getK(),
                                                             shapeName, NoVector, NULL, NoVector), {}, {pos.x,pos.y,pos.z}, {1e1});
        R.wait(+posDiffTask);
    }else{
        auto posDiffTask = R.newCtrlTask(new TaskMap_Default(posDiffTMT, R.getK(),
                                                             shapeName, NoVector, frameName, NoVector), {}, {pos.x,pos.y,pos.z}, {1e1});
        R.wait(+posDiffTask);
    }
    return AS_done;
}

///perform vecDiffTMT
int alignTask(Roopi& R,mlr::Vector vec1,mlr::Vector vec2,char* shapeName,char* frameName){
    if(strcmp(frameName,"world")==0) {
        auto vecDiffTask = R.newCtrlTask(new TaskMap_Default(vecDiffTMT, R.getK(),
                                                             shapeName, vec1, NULL, vec2), {}, {}, {1e1});
        R.wait(+vecDiffTask);
    }else{
        auto vecDiffTask = R.newCtrlTask(new TaskMap_Default(vecDiffTMT, R.getK(),
                                                             shapeName, vec1, frameName, vec2), {}, {}, {1e1});
        R.wait(+vecDiffTask);
    }
    return AS_done;
}

///perform posDiffTMT and vecDiffTMT
int pos_alignTask(Roopi& R,mlr::Vector pos,
                  mlr::Vector vec1,mlr::Vector vec2,
                  char* shapeName,char* frameName){
    if(strcmp(frameName,"world")==0) {
        auto posDiffTask = R.newCtrlTask(new TaskMap_Default(posDiffTMT, R.getK(),
                                                             shapeName, NoVector, NULL, NoVector), {}, {pos.x,pos.y,pos.z}, {1e1});
        auto vecDiffTask = R.newCtrlTask(new TaskMap_Default(vecDiffTMT, R.getK(),
                                                             shapeName, vec1, NULL, vec2), {}, {}, {1e1});
        R.wait(posDiffTask+vecDiffTask);
    }else{
        auto posDiffTask = R.newCtrlTask(new TaskMap_Default(posDiffTMT, R.getK(),
                                                             shapeName, NoVector, frameName, NoVector), {}, {pos.x,pos.y,pos.z}, {1e1});
        auto vecDiffTask = R.newCtrlTask(new TaskMap_Default(vecDiffTMT, R.getK(),
                                                             shapeName, vec1, frameName, vec2), {}, {}, {1e1});
        R.wait(posDiffTask+vecDiffTask);
    }
    return AS_done;
}

///perform posDiffTMT, vecDiffTMT and vecDiffTMT
int pos_align_alignTask(Roopi& R,mlr::Vector pos,
                        mlr::Vector vec11,mlr::Vector vec12,
                        mlr::Vector vec21,mlr::Vector vec22,
                        char* shapeName,char* frameName){
    if(strcmp(frameName,"world")==0){
        auto posDiffTask = R.newCtrlTask(new TaskMap_Default(posDiffTMT, R.getK(),
                                                             shapeName, NoVector, NULL, NoVector), {}, {pos.x,pos.y,pos.z}, {1e1});
        auto vecDiffTask1 = R.newCtrlTask(new TaskMap_Default(vecDiffTMT, R.getK(),
                                                              shapeName, vec11, NULL, vec12), {}, {}, {1e1});
        auto vecDiffTask2 = R.newCtrlTask(new TaskMap_Default(vecDiffTMT, R.getK(),
                                                              shapeName, vec21, NULL, vec22), {}, {}, {1e1});
        R.wait(posDiffTask+vecDiffTask1);
    }else{
        auto posDiffTask = R.newCtrlTask(new TaskMap_Default(posDiffTMT, R.getK(),
                                                             shapeName, NoVector, frameName, NoVector), {}, {pos.x,pos.y,pos.z}, {1e1});
        auto vecDiffTask1 = R.newCtrlTask(new TaskMap_Default(vecDiffTMT, R.getK(),
                                                              shapeName, vec11, frameName, vec12), {}, {}, {1e1});
        auto vecDiffTask2 = R.newCtrlTask(new TaskMap_Default(vecDiffTMT, R.getK(),
                                                              shapeName, vec21, frameName, vec22), {}, {}, {1e1});

        R.wait(posDiffTask+vecDiffTask1);

    }
    return AS_done;
}

///open/close gripper
int grip(Roopi* r, const bool toGrip)
{
    /// Find the joint that we care about
    mlr::Joint *j = r->getK()->getJointByName("l_gripper_l_finger_joint");

    // Unlock the joint.
    // This is only necessary, since we want the gripper to normally stay open/closed
    r->getTaskController().lockJointGroupControl("l_gripper", false);

    // Target should be all the way open, or all the way closed
    double target = toGrip ? j->limits(0) : j->limits(1);
//    cout<<target<<endl;
    // Create a control task
    auto gripTask =  r->newCtrlTask(new TaskMap_qItself({j->to->index}, false), ARR(1., .8, 10., 10.), {target}, ARR(10000.)); // Set this to be precise...

    //      cout << "Current:" << gripTask->task->PD().y_ref << " target:" << gripTask->task->PD().y_target << endl;

    // Wait for task to complete
    r->wait({-gripTask});

    // Re-lock the gripper.
    r->getTaskController().lockJointGroupControl("l_gripper", true);

    return AS_done;
}

int performRelativeAction(Roopi& R,int action,bool keepVertical, double customStepSize){
    ///0,1 movement in gripper z axis
    ///2,3 movement in gripper y axis
    ///4,5 movement in gripper x axis
    double stepSize = .012;
    if(customStepSize!=NULL) stepSize=customStepSize;//use custom step size
    mlr::Vector increment;
    switch(action){
    case 0  :
        increment = {0.,.0,stepSize};
        break;
    case 1  :
        increment = {0.,.0,-stepSize};
        break;
    case 2  :
        increment = {.0,stepSize,.0};
        break;
    case 3  :
        increment = {.0,-stepSize,.0};
        break;
    case 4  :
        increment = {stepSize,.0,.0};
        break;
    case 5  :
        increment = {-stepSize,.0,.0};
        break;
    default :
        increment = {stepSize,.0,.0};
        break;
    }

    ///move endeffL to new position and keep vertical
    mlr::Vector newPos = addRelativeTranslation(R,"endeffL",increment);
    if(keepVertical){
        mlr::Vector vec1 = {1.,0.,0.};
        mlr::Vector vec2 ={0.,0.,-1.};
        return pos_alignTask(R,newPos,vec1,vec2,"endeffL","world");
    }else{
        return posTask(R,newPos,"endeffL","world");
    }

}

///simple controller
///move gripper to obj in xy plane
///margin control the accuarcy of alignment
int simpleController(Index index,Index center, int margin){
    int action =-1;
    if(index.y>center.y+margin) action = 0;
    if(index.y<center.y-margin) action = 1;
    if(index.x>center.x+margin) action = 2;
    if(index.x<center.x-margin) action = 3;
    return action;
}

///grasp a obj given marker position and align vector from above
int graspFixedPos(Roopi& R,mlr::Vector pos,mlr::Vector z,mlr::Vector x){
    mlr::Vector ref1 = {1.,0.,0.};
    mlr::Vector ref2 = {0.,0,1.};
    z.normalize();
    x.normalize();
    return pos_align_alignTask(R,pos,ref1,z,ref2,x,"endeffL","world");
}


///twist wrist given rotated rect
int twistWrist2(Roopi& R,cv::RotatedRect rect){
    double degree;
    //convert rect angle from -90,0 deg to 0,180
    if(rect.size.width < rect.size.height){
        degree =  rect.angle+180;
    }else{
        degree = rect.angle+90;
    }
    double rad = degree/180.*3.14;//convert to rad

    mlr::Joint* lWrist = R.getK()->getJointByName("left_w2");
    double originAngle = R.getK()->q(lWrist->qIndex);
    originAngle += rad;

    //check the rad limit
    if(originAngle>3.14){
        originAngle -=3.14;
    }
    if(originAngle<-3.14){
        originAngle += 3.14;
    }
    cout<<"Angle "<<originAngle<<endl;
    if(originAngle<3.14 && originAngle>-3.14){
        auto lwrist = R.newCtrlTask(new TaskMap_qItself(QIP_byJointNames, {"left_w2"}, R.getK()), {}, {originAngle});
        R.wait(+lwrist);
    }else{}
    return AS_done;
}
}
