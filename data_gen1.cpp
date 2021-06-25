#include <stdio.h> 
#include <stdlib.h> 
#include<time.h>

#include <cstdio>
#include <math.h>
#include <vector>
#include "ceres/ceres.h"
#include "ceres/dynamic_autodiff_cost_function.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "random.h"
#include <fstream>

using ceres::AutoDiffCostFunction;
using ceres::DynamicAutoDiffCostFunction;
using ceres::CauchyLoss;
using ceres::CostFunction;
using ceres::LossFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
using ceres::examples::RandNormal;
using std::min;
using std::vector;

int num_steps = 100;
double pi = 3.141592653;
double sqrt2 = 1.4142135623;
double sensor_range = 30.00;
double sensor_angle_range = 2*pi/5;

double odom_stddev_r = 0.1;//0.1m error
double odom_stddev_theeta = 0.2;//about 5 degrees
double sensor_stddev_r = 1;//1m error 
double sensor_stddev_theeta = 0.1;//about 0.5 degrees error

typedef struct point_2d
{
	double x, y;
}point_2d;

typedef struct rel_vec
{
	double theeta;
	double r;
}rel_vec;

typedef struct td_pose
{
	double x, y, theeta;
}td_pose;

double dist_pts(point_2d pt1, point_2d pt2)
{
	double dist = sqrt( pow((pt2.y-pt1.y),2) + pow((pt2.x-pt1.x),2) );
	return dist;
}

double find_angle(point_2d pt1, point_2d pt2)//returns angle in (0,2pi)
{
	int quadrant;
	if(pt2.y>=pt1.y && pt2.x>=pt1.x)
		quadrant = 1;
	else if(pt2.y>pt1.y && pt2.x<pt1.x)
		quadrant = 2;
	else if(pt2.y<pt1.y && pt2.x<pt1.x)
		quadrant = 3;
	else 
		quadrant = 4;

	double a_tan_theeta = atan((pt2.y - pt1.y)/(pt2.x - pt1.x));
	if(quadrant == 1)
		return a_tan_theeta;
	else if(quadrant == 2)
		return (pi+a_tan_theeta);
	else if(quadrant == 3)
		return (pi+a_tan_theeta);
	else 
		return (2*pi+a_tan_theeta);
}

void findLandmarksInRange(vector<int> &arr_indices, td_pose robot_pose_curr, vector<point_2d> landmark_map_for_sim)
{
	int n = landmark_map_for_sim.size();
	point_2d curr_point;
    double theeta_lm_wrto_robot_orientation;
	for(int i=0;i<n;i++)
	{
		curr_point.x = robot_pose_curr.x;
		curr_point.y = robot_pose_curr.y;

		if( dist_pts(curr_point, landmark_map_for_sim[i]) < sensor_range)
		{
			theeta_lm_wrto_robot_orientation = find_angle(curr_point,landmark_map_for_sim[i]);
			theeta_lm_wrto_robot_orientation = remainder(theeta_lm_wrto_robot_orientation, (2*pi)) - robot_pose_curr.theeta;
			theeta_lm_wrto_robot_orientation = remainder(theeta_lm_wrto_robot_orientation,(2*pi));
			if(abs(theeta_lm_wrto_robot_orientation) < sensor_angle_range)
				arr_indices.push_back(i);
		}
	}
}

namespace 
{
	void simulatePoses(vector<td_pose> &robot_poses)
	{
		rel_vec odom_ideal; odom_ideal.r=1; odom_ideal.theeta=0;
		td_pose robot_pose_curr;
		rel_vec odom_observed;
	    robot_pose_curr.x = 0; robot_pose_curr.y = 0; robot_pose_curr.theeta = pi/4;
	    double theeta_temp;
	    for(int i=0;i<=num_steps;i++)
	    {
	    	std::cout<<"Robot pose: ("<<robot_pose_curr.x<<","<<robot_pose_curr.y<<","<<robot_pose_curr.theeta<<")  ";
	    	robot_poses.push_back(robot_pose_curr);
	    	odom_observed.r = (2*((double) rand() / (RAND_MAX))-1)*odom_stddev_r + odom_ideal.r;
		    //odom_observed.r = RandNormal()*odom_stddev_r + odom_ideal.r;

		    odom_observed.theeta = (2*((double) rand() / (RAND_MAX))-1)*odom_stddev_theeta + odom_ideal.theeta;
		   //odom_observed.theeta = remainder(RandNormal()*odom_stddev_theeta + odom_ideal.theeta, (2*pi));

		    std::cout<<"Odom_observed: ("<<odom_observed.r<<","<<odom_observed.theeta<<")"<<std::endl;
		    theeta_temp = remainder(odom_observed.theeta + robot_pose_curr.theeta, (2*pi));
		    robot_pose_curr.x = robot_pose_curr.x + (odom_observed.r)*cos(theeta_temp); 
		    robot_pose_curr.y = robot_pose_curr.y + (odom_observed.r)*sin(theeta_temp);
		    robot_pose_curr.theeta = theeta_temp;
	    }
	    return;
	}

	void simulateSensorReadings(vector<td_pose> &ground_truth_robot_poses, vector<int> &sensor_reading_nums, vector<vector<rel_vec> > &sensor_readings, 
			vector<point_2d> landmark_map_for_sim, vector<int> &landmarks_detected_actually, vector<point_2d> &gen_lms)
	{
		rel_vec odom_ideal; odom_ideal.r=1; odom_ideal.theeta=0;
	    td_pose robot_pose_curr;
	    robot_pose_curr.x = 0; robot_pose_curr.y = 0; robot_pose_curr.theeta = pi/4;
	    vector<int> arr_indices;
	    rel_vec temp_rel_vec;
	    vector<rel_vec> temp_rel_vec_arr;
	    point_2d temp_pt;
	    double theeta_abs;
	    point_2d generated_lm;
	    double temp_angle;
	    for(int i=0;i<=num_steps;i++)
	    {
	    	ground_truth_robot_poses.push_back(robot_pose_curr);
	    	temp_pt.x = robot_pose_curr.x; temp_pt.y = robot_pose_curr.y;
		    findLandmarksInRange(arr_indices, robot_pose_curr, landmark_map_for_sim);
		   	sensor_reading_nums.push_back(arr_indices.size());

		    for(int j=0;j<arr_indices.size();j++)
		    {
		    	landmarks_detected_actually.push_back(arr_indices[j]);
		    	temp_rel_vec.r = dist_pts(landmark_map_for_sim[arr_indices[j]], temp_pt);
		        temp_rel_vec.theeta = find_angle(temp_pt,landmark_map_for_sim[arr_indices[j]]);
		        temp_rel_vec.theeta = remainder(temp_rel_vec.theeta, (2*pi)) - robot_pose_curr.theeta;
		        temp_rel_vec.theeta = remainder(temp_rel_vec.theeta,(2*pi));
		        //is from 0 to pi or 0 to -pi

		        //temp_rel_vec.r = RandNormal() * sensor_stddev_r + temp_rel_vec.r;
		        //temp_rel_vec.theeta = RandNormal() * sensor_stddev_theeta + temp_rel_vec.theeta;
		        temp_rel_vec.r = (2*((double) rand() / (RAND_MAX))-1) * sensor_stddev_r + temp_rel_vec.r;
		        temp_rel_vec.theeta = (2*((double) rand() / (RAND_MAX))-1) * sensor_stddev_theeta + temp_rel_vec.theeta;

		        temp_rel_vec.theeta = remainder(temp_rel_vec.theeta,(2*pi));
		        temp_rel_vec_arr.push_back(temp_rel_vec);

		        temp_angle = remainder(robot_pose_curr.theeta + temp_rel_vec.theeta, (2*pi)); 
		        generated_lm.x = robot_pose_curr.x + (temp_rel_vec.r)*cos(temp_angle);
		        generated_lm.y = robot_pose_curr.y + (temp_rel_vec.r)*sin(temp_angle);
		        gen_lms.push_back(generated_lm);
		    }	
		    sensor_readings.push_back(temp_rel_vec_arr);

		    robot_pose_curr.x = robot_pose_curr.x + (odom_ideal.r)*cos(robot_pose_curr.theeta);
		    robot_pose_curr.y = robot_pose_curr.y + (odom_ideal.r)*sin(robot_pose_curr.theeta);
		    
		    arr_indices.clear();	
		    temp_rel_vec_arr.clear();
	    }
		return;
	}
}  	

int main(int argc, char** argv) 
{
	srand(time(0));
	vector<point_2d> landmark_map_for_sim;//only to generate sample sensor data
	std::ifstream fin; 
	fin.open("data_lm.txt");
	point_2d temp_point;
	fin>>temp_point.x;
	while(!fin.eof())
	{
	  fin>>temp_point.y;
	  landmark_map_for_sim.push_back(temp_point);
	  fin>>temp_point.x;
	}
	std::cout<<"Size of the landmark map for sim:"<<landmark_map_for_sim.size()<<std::endl;

	vector<int> landmarks_detected_actually;
	vector<td_pose> ground_truth_robot_poses;
	vector<td_pose> robot_poses;
	vector<vector<rel_vec> > sensor_readings;
	vector<int> sensor_reading_nums;
	vector<point_2d> gen_lms;
	std::cout<<"Simulating..."<<std::endl;
	simulatePoses(robot_poses);
	simulateSensorReadings(ground_truth_robot_poses, sensor_reading_nums, sensor_readings, landmark_map_for_sim ,landmarks_detected_actually, gen_lms);
	std::cout<<"Finished simulating..."<<std::endl;	

	std::ofstream robot_posns;
    robot_posns.open("robot_posns.txt");
    for(int i=0;i<robot_poses.size();i++)
    	robot_posns<<robot_poses[i].x<<","<<robot_poses[i].y<<","<<robot_poses[i].theeta<<std::endl;
    robot_posns.close();
	std::cout<<"Saved the odometry simulated values into robot_posns.txt"<<std::endl;

	std::ofstream ground_truth_robot_posns;
    ground_truth_robot_posns.open("ground_truth_robot_posns.txt");
    for(int i=0;i<ground_truth_robot_poses.size();i++)
    	ground_truth_robot_posns<<ground_truth_robot_poses[i].x<<","<<ground_truth_robot_poses[i].y<<","<<ground_truth_robot_poses[i].theeta<<std::endl;
    ground_truth_robot_posns.close();
	std::cout<<"Saved the odometry ground truth values into ground_truth_robot_posns.txt"<<std::endl;

	std::ofstream sensor_observns;
    sensor_observns.open("sensor_observns.txt");
    for(int i=0;i<sensor_readings.size();i++){
    	for(int j=0;j<sensor_readings[i].size();j++)
    		sensor_observns<<sensor_readings[i][j].r<<","<<sensor_readings[i][j].theeta<<std::endl;
    }
    sensor_observns.close();
	std::cout<<"Saved the sensor_readings simulated values into sensor_observns.txt"<<std::endl;

	std::ofstream sensor_observn_nums;
    sensor_observn_nums.open("sensor_observn_nums.txt");
    for(int i=0;i<sensor_reading_nums.size();i++)
    		sensor_observn_nums<<sensor_reading_nums[i]<<std::endl;
    sensor_observn_nums.close();
	std::cout<<"Saved the number of sensor_readings at each time stamp into sensor_observn_nums.txt"<<std::endl;

	std::ofstream lms_seen;
    lms_seen.open("seen_lms.txt");
    for(int i=0;i<landmarks_detected_actually.size();i++)
    		lms_seen<<landmarks_detected_actually[i]<<std::endl;
    lms_seen.close();
	std::cout<<"Saved the landmarks from the map that have been observed into seen_lms.txt"<<std::endl;

	std::ofstream gen_lms_for_ver;
    gen_lms_for_ver.open("gen_lms_for_ver.txt");
    for(int i=0;i<gen_lms.size();i++)
    		gen_lms_for_ver<<gen_lms[i].x<<","<<gen_lms[i].y<<std::endl;
    gen_lms_for_ver.close();
	std::cout<<"Generated landmarks for verification have been saved into gen_lms_for_ver.txt"<<std::endl;

	std::ofstream guessed_map_initial;
	point_2d pt_temp;
    guessed_map_initial.open("guessed_map_initial.txt");
    for(int i=0;i<robot_poses.size();i++){
    	for(int j=0;j<sensor_readings[i].size();j++){
    		pt_temp.x = robot_poses[i].x + (sensor_readings[i][j].r)*cos(sensor_readings[i][j].theeta + robot_poses[i].theeta);
    		pt_temp.y = robot_poses[i].y + (sensor_readings[i][j].r)*sin(sensor_readings[i][j].theeta + robot_poses[i].theeta);

    		guessed_map_initial<<pt_temp.x<<","<<pt_temp.y<<std::endl;
    	}
    }
    guessed_map_initial.close();
	std::cout<<"Saved the gen_lms (simulated) values into guessed_map_initial.txt"<<std::endl;

	std::cout<<"landmark_map_for_sim.size(): "<<landmark_map_for_sim.size()<<std::endl;

	return 0;
}

