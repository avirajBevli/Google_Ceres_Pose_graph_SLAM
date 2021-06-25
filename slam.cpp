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

//////used in data_gen.cpp also////
int num_steps = 100;
double threshold_lm_dist = 3;
double pi = 3.141592653;
double sqrt2 = 1.4142135623;
double sensor_range = 30.00;
double sensor_angle_range = 2*pi/5;

double odom_stddev_r = 0.1;//0.1m error
double odom_stddev_theeta = 0.2;//about 5 degrees
double sensor_stddev_r = 0.5;//0.5m error 
double sensor_stddev_theeta = 0.1;//about 2 degrees error
/////////////////////////////////

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
  return sqrt((pt2.y - pt1.y)*(pt2.y - pt1.y) + (pt2.x - pt1.x)*(pt2.x - pt1.x));
}

struct OdometryConstraint 
{
    typedef AutoDiffCostFunction<OdometryConstraint, 3, 3, 3> OdometryCostFunction;//3 is the dimension of the residual, 3 is the dimension of the parameter block

    OdometryConstraint(td_pose odom_val) : odom_val(odom_val) {}

    template <typename T>
    bool operator()(const T* const parameter_block_odom_prev, const T* const parameter_block_odom, T* residual) const {
      residual[0] = (parameter_block_odom[0] -(parameter_block_odom_prev[0] + odom_val.x)) / T(odom_stddev_r);
      residual[1] = (parameter_block_odom[1] -(parameter_block_odom_prev[1] + odom_val.y)) / T(odom_stddev_r);
      residual[2] = (parameter_block_odom[2] -(parameter_block_odom_prev[2] + odom_val.theeta)) / T(odom_stddev_theeta);
      return true;
    }

    static OdometryCostFunction* Create(td_pose odom_val) {
      return new OdometryCostFunction(new OdometryConstraint(odom_val));
    }

    const td_pose odom_val;
};

//problem.AddResidualBlock(OdometryConstraint1::Create(odom_val), NULL, parameter_block_odom[i]);
struct OdometryConstraint1
{
    typedef AutoDiffCostFunction<OdometryConstraint1, 3, 3> OdometryCostFunction1;//3 is the dimension of the residual, 3 is the dimension of the parameter block

    OdometryConstraint1(td_pose odom_val) : odom_val(odom_val) {}

    template <typename T>
    bool operator()(const T* const parameter_block_odom, T* residual) const {
		residual[0] = (parameter_block_odom[0] - odom_val.x) / T(odom_stddev_r);
		residual[1] = (parameter_block_odom[1] - odom_val.y) / T(odom_stddev_r);
		residual[2] = (parameter_block_odom[2] -((pi/4) + odom_val.theeta)) / T(odom_stddev_theeta);
		return true;
    }

    static OdometryCostFunction1* Create(td_pose odom_val) {
      return new OdometryCostFunction1(new OdometryConstraint1(odom_val));
    }

    const td_pose odom_val;
};

//problem.AddResidualBlock(LandmarkConstraint::Create(sj,si), NULL, parameter_block_odom[i], parameter_block_odom[pi]);
struct LandmarkConstraint 
{
    typedef AutoDiffCostFunction<LandmarkConstraint, 2, 3, 3> LandmarkCostFunction;//2 is the dimension of the residual
    //3, 3 are the dimensions of parameter_block_odom[i] and parameter_block_odom[j]

    LandmarkConstraint(rel_vec sj, rel_vec si): sj(sj), si(si) {}

    template <typename T>
    bool operator()(const T* const parameter_block_odom_i, const T* const parameter_block_odom_j, T* residual)  const {
      //std::cout<<"Entered operator function"<<std::endl;
		T si_proj_r(0);  T si_proj_theeta(0);
		T lm_global_x(0); T lm_global_y(0);

		lm_global_x = parameter_block_odom_j[0] + T(sj.r)*cos(parameter_block_odom_j[2] + T(sj.theeta));
		lm_global_y = parameter_block_odom_j[1] + T(sj.r)*sin(parameter_block_odom_j[2] + T(sj.theeta));

		si_proj_r = sqrt( pow(lm_global_x - parameter_block_odom_i[0], 2) + pow(lm_global_y - parameter_block_odom_i[1], 2) );

		if(lm_global_y >= parameter_block_odom_i[1] && lm_global_x >= parameter_block_odom_i[0])
			si_proj_theeta = atan((lm_global_y - parameter_block_odom_i[1])/(lm_global_x - parameter_block_odom_i[0]));

		else if(lm_global_y > parameter_block_odom_i[1] && lm_global_x < parameter_block_odom_i[0])
			si_proj_theeta = (pi + atan((lm_global_y - parameter_block_odom_i[1])/(lm_global_x - parameter_block_odom_i[0])) );

		else if(lm_global_y < parameter_block_odom_i[1] && lm_global_x < parameter_block_odom_i[0])
			si_proj_theeta = (-1*pi + atan((lm_global_y - parameter_block_odom_i[1])/(lm_global_x - parameter_block_odom_i[0])) );
		else
			si_proj_theeta = (atan((lm_global_y - parameter_block_odom_i[1])/(lm_global_x - parameter_block_odom_i[0]) ));	
		/*if(si_proj_theeta > pi)
			si_proj_theeta = (2*pi) - si_proj_theeta;
		si_proj_theeta = si_proj_theeta - parameter_block_odom_i[2];
		if(si_proj_theeta > pi)
			si_proj_theeta = (2*pi) - si_proj_theeta;*/

		residual[0] = (T(si.r) - si_proj_r) / T(sensor_stddev_r);
		residual[1] = (T(si.theeta) - si_proj_theeta) / T(sensor_stddev_theeta);
		return true;
    }

    static LandmarkCostFunction* Create(rel_vec sj, rel_vec si) {
      return new LandmarkCostFunction(new LandmarkConstraint(sj,si));
    }

    const rel_vec sj;
    const rel_vec si;
};

//problem.AddResidualBlock(LandmarkConstraint1::Create(sj,si), NULL, parameter_block_odom[i]);
struct LandmarkConstraint1
{
    typedef AutoDiffCostFunction<LandmarkConstraint1, 2, 3> LandmarkCostFunction1;//2 is the dimension of the residual
    //3, 3 are the dimensions of parameter_block_odom[i] and parameter_block_odom[j]

    LandmarkConstraint1(rel_vec sj, rel_vec si): sj(sj), si(si) {}

    template <typename T>
    bool operator()(const T* const parameter_block_odom_i, T* residual)  const {
      //std::cout<<"Entered operator function"<<std::endl;
		T si_proj_r(0);  T si_proj_theeta(0);
		T lm_global_x(0); T lm_global_y(0);

		lm_global_x = T(sj.r)*cos((pi/4) + T(sj.theeta));
		lm_global_y = T(sj.r)*sin((pi/4) + T(sj.theeta));

		si_proj_r = sqrt( pow(lm_global_x - parameter_block_odom_i[0], 2) + pow(lm_global_y - parameter_block_odom_i[1], 2) );
		//si_proj_theeta will belong to -pi to pi
		if(lm_global_y >= parameter_block_odom_i[1] && lm_global_x >= parameter_block_odom_i[0])
			si_proj_theeta = atan((lm_global_y - parameter_block_odom_i[1])/(lm_global_x - parameter_block_odom_i[0]));

		else if(lm_global_y > parameter_block_odom_i[1] && lm_global_x < parameter_block_odom_i[0])
			si_proj_theeta = (pi + atan((lm_global_y - parameter_block_odom_i[1])/(lm_global_x - parameter_block_odom_i[0])) );

		else if(lm_global_y < parameter_block_odom_i[1] && lm_global_x < parameter_block_odom_i[0])
			si_proj_theeta = (-1*pi + atan((lm_global_y - parameter_block_odom_i[1])/(lm_global_x - parameter_block_odom_i[0])) );
		else
			si_proj_theeta = (atan((lm_global_y - parameter_block_odom_i[1])/(lm_global_x - parameter_block_odom_i[0]) ));
	/*	if(si_proj_theeta > pi)
			si_proj_theeta = (2*pi) - si_proj_theeta;
		si_proj_theeta = si_proj_theeta - parameter_block_odom_i[2];*/
/*		if(si_proj_theeta > pi)
			si_proj_theeta = (2*pi) - si_proj_theeta;*/

		residual[0] = (T(si.r) - si_proj_r) / T(sensor_stddev_r/100);
		residual[1] = (T(si.theeta) - si_proj_theeta) / T(sensor_stddev_theeta/100);
		return true;
    }

    static LandmarkCostFunction1* Create(rel_vec sj, rel_vec si) {
      return new LandmarkCostFunction1(new LandmarkConstraint1(sj,si));
    }

    const rel_vec sj;
    const rel_vec si;
};

void Generate_init_map(vector<point_2d> &landmark_map, vector<vector<int> > &landmark_indices_found, vector<rel_vec> sensor_readings)
{
	point_2d temp;
	vector<int> temp_vec;
	for(int i=0;i<sensor_readings.size();i++){
		temp.x = (sensor_readings[i].r)*cos((pi/4)+sensor_readings[i].theeta);
		temp.y = (sensor_readings[i].r)*sin((pi/4)+sensor_readings[i].theeta);
		landmark_map.push_back(temp);
		temp_vec.push_back(i);
	}
	landmark_indices_found.push_back(temp_vec);
	return;
}

//findAndUpdateLandmarks(robot_pose_curr, landmark_map, sensor_readings[i], landmark_indices_found);
void findAndUpdateLandmarks(td_pose robot_pose_curr, vector<point_2d> &landmark_map, vector<rel_vec> sensor_readings, vector<vector<int> > &landmark_indices_found)
{
	int no_lms = landmark_map.size();
	point_2d proj_lm;
	double min=10000;
	vector<int> temp_vec;
	for(int i=0;i<sensor_readings.size();i++)
	{
		proj_lm.x = robot_pose_curr.x + (sensor_readings[i].r)*cos(robot_pose_curr.theeta + sensor_readings[i].theeta);
		proj_lm.y = robot_pose_curr.y + (sensor_readings[i].r)*sin(robot_pose_curr.theeta + sensor_readings[i].theeta);
		
	    min = 10000;
	    int min_index;
	    for(int j=0;j<no_lms;j++)
	    {
			if(dist_pts(proj_lm,landmark_map[j]) < min)
			{
				min = dist_pts(proj_lm,landmark_map[j]);
				min_index = j;
			}
	    }

	    if(min<threshold_lm_dist)
			temp_vec.push_back(min_index);
	    else{
	    	temp_vec.push_back(landmark_map.size());
	    	std::cout<<"landmark "<<landmark_map.size()<<" added"<<std::endl;
			landmark_map.push_back(proj_lm);
	    }
	}
	
	landmark_indices_found.push_back(temp_vec);
	return;
}

bool isInVec(int key, vector<int> arr){
	for(int i=0;i<arr.size();i++){
		if(arr[i]==key)
			return 1;
	}
	return 0;
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

//findPosesWithThisLandmark(i,pose_list,landmark_indices_found,lm_index)
void findPosesWithThisLandmark(int curr_pose_index, vector<int> &pose_list, vector<vector<int> > landmark_indices_found, int lm_index)
{
	for(int i=0;i<curr_pose_index;i++)
	{
		bool is_found = isInVec(lm_index,landmark_indices_found[i]);
		if(is_found)
			pose_list.push_back(i);
	}
	return;
}

//find_s_pi(landmark_indices_found,pi,lm_index);
//rel_vec sj = find_s_pi(sensor_readings, landmark_indices_found,pi,lm_index)
rel_vec find_s_pi(vector<vector<rel_vec> > sensor_readings, vector<vector<int> > landmark_indices_found, int pi, int lm_index)
{
	int index;
	int n = landmark_indices_found[pi].size();
	for(int i=0;i<n;i++)
	{
		if(landmark_indices_found[pi][i] == lm_index)
			index = i;
	}
	return sensor_readings[pi][index];
}

void printvec(vector<int>arr){
	for(int i=0;i<arr.size();i++)
		std::cout<<arr[i]<<" ";
	std::cout<<std::endl;
}

int main()
{
	vector<td_pose> robot_poses;
	std::ifstream fin_poses; 
	fin_poses.open("robot_posns.txt");
	td_pose temp_pose;
	char ch_temp;
	fin_poses>>temp_pose.x;
	while(!fin_poses.eof())
	{
		fin_poses>>ch_temp;
		fin_poses>>temp_pose.y;
		fin_poses>>ch_temp;
		fin_poses>>temp_pose.theeta;
		robot_poses.push_back(temp_pose);
		fin_poses>>temp_pose.x;
	}
	fin_poses.close();
	std::cout<<"Read robot poses data from robot_posns.txt"<<std::endl;

	vector<int> lm_indices;
	std::ifstream fin_lm_indices; 
	fin_lm_indices.open("seen_lms.txt");
	int temp_int;
	fin_lm_indices>>temp_int;
	while(!fin_lm_indices.eof())
	{
		lm_indices.push_back(temp_int);
		fin_lm_indices>>temp_int;
	}
	fin_lm_indices.close();
	std::cout<<"Read lm_indices data from seen_lms.txt"<<std::endl;

	vector<rel_vec> sensor_read_single_vec;
	std::ifstream fin_observs; 
	fin_observs.open("sensor_observns.txt");
	rel_vec temp_rel_vec;
	fin_observs>>temp_rel_vec.r;
	while(!fin_observs.eof())
	{
		fin_observs>>ch_temp;
		fin_observs>>temp_rel_vec.theeta;
		sensor_read_single_vec.push_back(temp_rel_vec);
		fin_observs>>temp_rel_vec.r;
	}
	fin_observs.close();
	std::cout<<"Read sensor data from sensor_observns.txt"<<std::endl;

	vector<int> sensor_observn_nums;
	std::ifstream fin_observ_nums; 
	fin_observ_nums.open("sensor_observn_nums.txt");
	int temp;
	fin_observ_nums>>temp;
	while(!fin_observ_nums.eof())
	{
		sensor_observn_nums.push_back(temp);
		fin_observ_nums>>temp;
	}
	fin_observ_nums.close();
	std::cout<<"Read data of the number of sensor readings from each pose from sensor_observn_nums.txt"<<std::endl;

	vector<vector<rel_vec> > sensor_readings;
	vector<vector<int> > landmark_indices_found;
	vector<rel_vec> temo_vec;
	vector<int> temo_int_vec;
	int count=0;
	for(int i=0;i<robot_poses.size();i++)
	{
		int n = sensor_observn_nums[i];
		for(int j=0;j<n;j++){
			temo_int_vec.push_back(lm_indices[count]);
			temo_vec.push_back(sensor_read_single_vec[count++]);
		}
		sensor_readings.push_back(temo_vec);
		landmark_indices_found.push_back(temo_int_vec);
		
		temo_vec.clear();
		temo_int_vec.clear();
	}

	int n = robot_poses.size();
	double** parameter_block_odom = (double**)malloc((n)*sizeof(double*));
	for(int i=0;i<n;i++)
	{
		parameter_block_odom[i] = (double*)malloc(3*sizeof(double));
		parameter_block_odom[i][0] = robot_poses[i].x;
		parameter_block_odom[i][1] = robot_poses[i].y;
		parameter_block_odom[i][2] = robot_poses[i].theeta;
	}

/*	vector<point_2d> landmark_map;
	Generate_init_map(landmark_map,landmark_indices_found,sensor_readings[0]);
	std::cout<<"Landmarks found in the initial pose  ";
 	printvec(landmark_indices_found[0]);
*/
	ceres::Problem problem;

	std::cout<<"Landmarks found in the current pose  ";
 	printvec(landmark_indices_found[0]);
	for(int i=1;i<n;i++)
	{
		std::cout<<"i:"<<i<<" ";
		td_pose odom_val;
		odom_val.x = parameter_block_odom[i][0]-parameter_block_odom[i-1][0];
		odom_val.y = parameter_block_odom[i][1]-parameter_block_odom[i-1][1];
		odom_val.theeta = parameter_block_odom[i][2]-parameter_block_odom[i-1][2];
	/*
		if(i==1)
			problem.AddResidualBlock(OdometryConstraint1::Create(odom_val), NULL, parameter_block_odom[i]);
		else*/
		problem.AddResidualBlock(OdometryConstraint::Create(odom_val), NULL, parameter_block_odom[i-1], parameter_block_odom[i]);
 		
 		td_pose robot_pose_curr = robot_poses[i];
 		std::cout<<"robot_pose_curr: ("<<robot_pose_curr.x<<","<<robot_pose_curr.y<<","<<robot_pose_curr.theeta<<")"<<std::endl;
 		std::cout<<"Landmarks found in the current pose  ";
 		printvec(landmark_indices_found[i]);
	 	
 		for(int j=0;j<sensor_readings[i].size();j++)
 		{
 			std::cout<<"j:"<<j<<" ";
 			rel_vec si = sensor_readings[i][j];
 			vector<int> pose_list;
 			int lm_index = landmark_indices_found[i][j];
 			std::cout<<"lm_index: "<<lm_index<<std::endl;
 			findPosesWithThisLandmark(i,pose_list,landmark_indices_found,lm_index);
 			std::cout<<"Poses with the same landmark: ";
 			printvec(pose_list);
 			for(int k=0;k<pose_list.size();k++)
			{
 				std::cout<<"k:"<<k<<std::endl;
				int pi = pose_list[k];
				rel_vec sj = find_s_pi(sensor_readings, landmark_indices_found, pi, lm_index);//this is the index of the sensor_reading in the local sensor_readings[pi]
				
				/*if(pi==0)
					problem.AddResidualBlock(LandmarkConstraint1::Create(sj,si), NULL, parameter_block_odom[i]);
				else*/
				problem.AddResidualBlock(LandmarkConstraint::Create(sj,si), NULL, parameter_block_odom[i], parameter_block_odom[pi]);
			}
 			pose_list.clear();
 		}
	}

	ceres::Solver::Options solver_options;
	solver_options.minimizer_progress_to_stdout = true;
	solver_options.max_num_iterations = 500;

	Solver::Summary summary;
	printf("Solving...\n");
	Solve(solver_options, &problem, &summary);
	printf("Done.\n");

	std::cout << summary.FullReport() << "\n";
/*	double diff_x = 0 - parameter_block_odom[0][0];
	double diff_y = 0 - parameter_block_odom[0][1];
	double diff_z = (pi/4) - parameter_block_odom[0][2];
	double r;
	for(int i=0;i<n;i++){
		parameter_block_odom[i][0] = parameter_block_odom[i][0] + diff_x;
		parameter_block_odom[i][1] = parameter_block_odom[i][1] + diff_y;
		parameter_block_odom[i][2] = parameter_block_odom[i][2] + diff_z;
	}
*/
	point_2d zeroth_point; zeroth_point.x = parameter_block_odom[0][0], zeroth_point.y = parameter_block_odom[0][1];
	point_2d curr_point;
	double diff_theeta = (pi/4) - parameter_block_odom[0][2];
	double shift_x = 0 - parameter_block_odom[0][0];
	double shift_y = 0 - parameter_block_odom[0][1];
	double r, theeta;
	parameter_block_odom[0][0] = parameter_block_odom[0][0] + shift_x;
	parameter_block_odom[0][1] = parameter_block_odom[0][1] + shift_y;
	parameter_block_odom[0][2] = parameter_block_odom[0][0] + diff_theeta;
	for(int i=1;i<n;i++){
		curr_point.x = parameter_block_odom[i][0];
		curr_point.y = parameter_block_odom[i][1];
		r = dist_pts(zeroth_point,curr_point);
		theeta = find_angle(zeroth_point,curr_point);
		theeta = remainder(theeta,(2*pi));
		parameter_block_odom[i][0] = (zeroth_point.x + r*cos(theeta + diff_theeta)) + shift_x;
		parameter_block_odom[i][1] = (zeroth_point.y + r*sin(theeta + diff_theeta)) + shift_y;
		parameter_block_odom[i][2] = parameter_block_odom[i][2] + diff_theeta;
	}

	std::ofstream generated_robot_poses;
	generated_robot_poses.open("generated_robot_poses.txt");
	for(int i=0;i<n;i++)
		generated_robot_poses<<parameter_block_odom[i][0]<<","<<parameter_block_odom[i][1]<<","<<parameter_block_odom[i][2]<<std::endl;
	generated_robot_poses.close();
	std::cout<<"Saved the generated robot poses into generated_robot_poses.txt"<<std::endl;

	point_2d lm_proj;
	std::ofstream generated_map;
	generated_map.open("generated_map.txt");
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<sensor_readings[i].size();j++){
			lm_proj.x = parameter_block_odom[i][0] + (sensor_readings[i][j].r)*cos(sensor_readings[i][j].theeta + parameter_block_odom[i][2]);
			lm_proj.y = parameter_block_odom[i][1] + (sensor_readings[i][j].r)*sin(sensor_readings[i][j].theeta + parameter_block_odom[i][2]);
			generated_map<<lm_proj.x<<","<<lm_proj.y<<std::endl;
		}		
	}
	generated_map.close();
	std::cout<<"Saved the generate map into generated_map.txt"<<std::endl;
/*
	std::cout<<"Size of landmark_map: "<<landmark_map.size()<<std::endl;*/

	return 0;
}