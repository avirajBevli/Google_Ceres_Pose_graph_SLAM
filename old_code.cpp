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

double pi = 3.141592653;
double sqrt2 = 1.4142135623;
double sensor_range = 30.00;
double sensor_angle_range = pi/3;//that is can see within an arc of -60 to 60 degrees with respect to its direction of orientation

#define num_steps 80
double threshold_lm_dist = 10;

///for now, odom_stddev_x , odom_stddev_y, odom_stddev_theeta have been taken to be constants.....this is a simplifying assumption....the code can 
//be easily extended to incorporate the fact that these 3 standard deviation values are not constants
DEFINE_double(pose_separation, sqrt2, "The distance that the robot traverses "
              "between successive odometry updates.");
//pose separation flag defines the length that the robot moves between 
//consecutive odom and sensor measurements 

//the first 4 std devns will be in a scale of 0 to 1

DEFINE_double(odometry_stddev_x, 0.1, "The standard deviation of "
              "odometry error of the robot_x.");

DEFINE_double(odometry_stddev_y, 0.1, "The standard deviation of "
              "odometry error of the robot_y.");

DEFINE_double(odometry_stddev_theeta, 0.1, "The standard deviation of "
              "odometry error of the robot_theeta.");
 
DEFINE_double(range_stddev, 0.1, "The standard deviation of range readings of "
              "the robot.");//assume that for sensor, sttdev in range and sttdev in theeta is the same

DEFINE_double(odometry_stddev_x_simulate, 0.2, "The standard deviation of "
              "odometry error of the robot_x.");

DEFINE_double(odometry_stddev_y_simulate, 0.1, "The standard deviation of "
              "odometry error of the robot_x.");

DEFINE_double(odometry_stddev_theeta_simulate, 0.1, "The standard deviation of "
              "odometry error of the robot_x.");

DEFINE_double(range_stddev_simulate, 0.1, "The standard deviation of "
              "odometry error of the robot_x.");

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

typedef struct sensor_reading_kundli
{
  double r;
  double theeta;
  int lm_no;
  int pose_index;
  int sensor_total_index_number;//starting from zero to total_no_sensor_readings
  int sensor_local_index_number;
}sensor_reading_kundli;

void printPointsVec(vector<point_2d> landmark_map)
{
  int n = landmark_map.size();
  for(int i=0;i<n;i++)
    std::cout<<"("<<landmark_map[i].x<<","<<landmark_map[i].y<<")"<<std::endl;
}

double dist_pts(point_2d pt1, point_2d pt2);
double abs(double a);
void printvec(vector<int> arr);
void printsensorvals(vector<vector<rel_vec> > sensor_readings);
double find_angle(point_2d pt2, point_2d pt1);//returns angle in (0,2pi)
int find_landmarks_in_range_simulate(td_pose robot_pose_curr, vector<point_2d> landmark_map, vector<int>* arr_indices);

/*
problem.AddResidualBlock(OdometryConstraint::Create(odometry_values[i-1]), 
      NULL, parameter_block_odom[i-1], parameter_block_odom[i]);
*/
struct OdometryConstraint 
{
    typedef AutoDiffCostFunction<OdometryConstraint, 3, 3, 3> OdometryCostFunction;//3 is the dimension of the residual, 3 is the dimension of the parameter block

    OdometryConstraint(td_pose odom_val, td_pose odometry_stddev) :
        odom_val(odom_val), odometry_stddev(odometry_stddev) {}

    template <typename T>
    bool operator()(const T* const parameter_block_odom_prev, const T* const parameter_block_odom, T* residual) const {
      residual[0] = (parameter_block_odom[0] -(parameter_block_odom_prev[0] + odom_val.x)) / T(odometry_stddev.x);
      residual[1] = (parameter_block_odom[1] -(parameter_block_odom_prev[1] + odom_val.y)) / T(odometry_stddev.y);
      residual[2] = (parameter_block_odom[2] -(parameter_block_odom_prev[2] + odom_val.theeta)) / T(odometry_stddev.theeta);
      return true;
    }

    static OdometryCostFunction* Create(td_pose odom_val, td_pose odometry_stddev) {
      return new OdometryCostFunction(new OdometryConstraint(odom_val, odometry_stddev));
    }

    const td_pose odom_val;
    const td_pose odometry_stddev;
};
//FLAGS_odometry_stddev_theeta

/*
  problem.AddResidualBlock(LandmarkConstraint::Create(), NULL, 
  parameter_block_odom[i], parameter_block_odom[pi], parameter_block_lm[sensor_reading_index], parameter_block_lm[sj_index]);
*/

struct LandmarkConstraint 
{
    typedef AutoDiffCostFunction<LandmarkConstraint, 2, 3, 3, 2, 2> LandmarkCostFunction;//2 is the dimension of the residual
    //3, 3 are the dimensions of parameter_block_odom[i] and parameter_block_odom[j] and 
    //2, 2 are the dimensions of parameter_block_odom[lmi] and parameter_block_odom[lmj]

    LandmarkConstraint(double range_stddev) : range_stddev(range_stddev) {}

    template <typename T>
    bool operator()(const T* const parameter_block_odom_i, const T* const parameter_block_odom_j,
                    const T* const parameter_block_lm_i, const T* const parameter_block_lm_j, T* residual) const {
      //std::cout<<"Entered operator function"<<std::endl;
      T si_proj_r(0);  T si_proj_theeta(0);
      T lm_global_x(0); T lm_global_y(0);
      
      lm_global_x = parameter_block_odom_j[0] + (parameter_block_lm_j[0])*cos(parameter_block_odom_j[2] + parameter_block_lm_j[1]);
      lm_global_y = parameter_block_odom_j[1] + (parameter_block_lm_j[0])*sin(parameter_block_odom_j[2] + parameter_block_lm_j[1]);

      si_proj_r = sqrt( pow(lm_global_x - parameter_block_odom_i[0], 2) + pow(lm_global_y - parameter_block_odom_i[1], 2) );

      if(lm_global_y >= parameter_block_odom_i[1] && lm_global_x >= parameter_block_odom_i[0])
         si_proj_theeta = atan((lm_global_y - parameter_block_odom_i[1])/(lm_global_x - parameter_block_odom_i[0]));

      else if(lm_global_y > parameter_block_odom_i[1] && lm_global_x < parameter_block_odom_i[0])
         si_proj_theeta = (pi + atan((lm_global_y - parameter_block_odom_i[1])/(lm_global_x - parameter_block_odom_i[0])) );

      else if(lm_global_y < parameter_block_odom_i[1] && lm_global_x < parameter_block_odom_i[0])
         si_proj_theeta = (pi + atan((lm_global_y - parameter_block_odom_i[1])/(lm_global_x - parameter_block_odom_i[0])) );
      else
         si_proj_theeta = (2*pi + atan((lm_global_y - parameter_block_odom_i[1])/(lm_global_x - parameter_block_odom_i[0]) ));

      residual[0] = (parameter_block_lm_i[0] - si_proj_r) / range_stddev;
      residual[1] = (parameter_block_lm_i[1] - si_proj_theeta) / range_stddev;
      return true;
    }

    static LandmarkCostFunction* Create() {
      return new LandmarkCostFunction(new LandmarkConstraint(FLAGS_range_stddev));
    }

    const double range_stddev;
};


namespace 
{
  //SimulateRobot(&odometry_values, &sensor_readings, landmark_map);
  void SimulateRobot(vector<td_pose>* odometry_values, vector<vector<rel_vec> >* sensor_readings, vector<point_2d> landmark_map)
  {
    std::cout<<std::endl<<"Num_steps of robot: "<<num_steps<<std::endl;

    // The robot starts out at the origin.
    td_pose robot_pose_curr;
    robot_pose_curr.x = 0; robot_pose_curr.y = 0; robot_pose_curr.theeta = pi/4;

    vector<int> arr_indices;
    rel_vec temp_rel_vec;
    vector<rel_vec> arr_temp;

    td_pose observed_odometry_value;
    point_2d temp_pt;
    temp_pt.x = robot_pose_curr.x; temp_pt.y = robot_pose_curr.y;

    td_pose actual_odometry_value;
    actual_odometry_value.x = 1; actual_odometry_value.y = 1; actual_odometry_value.theeta = 0;

    int no_landmarks_in_range = find_landmarks_in_range_simulate(robot_pose_curr, landmark_map, &arr_indices);
    for(int j=0;j<no_landmarks_in_range;j++)
    {
      temp_rel_vec.r = dist_pts(landmark_map[arr_indices[j]], temp_pt);
      temp_rel_vec.r = RandNormal() * FLAGS_range_stddev_simulate + temp_rel_vec.r;
      temp_rel_vec.theeta = find_angle(landmark_map[arr_indices[j]], temp_pt) - robot_pose_curr.theeta ;
      arr_temp.push_back(temp_rel_vec);
    }
    sensor_readings->push_back(arr_temp);

    arr_temp.clear();
    arr_indices.clear();
    for (int i = 0; i < num_steps; ++i) 
    {
      observed_odometry_value.x = RandNormal()*FLAGS_odometry_stddev_x_simulate + actual_odometry_value.x;
      observed_odometry_value.y = RandNormal()*FLAGS_odometry_stddev_y_simulate + actual_odometry_value.y;
      observed_odometry_value.theeta = RandNormal()*FLAGS_odometry_stddev_theeta_simulate + actual_odometry_value.theeta;//angle deviation from the previous angle of robot pose

      robot_pose_curr.theeta += observed_odometry_value.theeta;
      robot_pose_curr.theeta = remainder(robot_pose_curr.theeta, (2*pi));
      robot_pose_curr.x += observed_odometry_value.x;
      robot_pose_curr.y += observed_odometry_value.y;
      temp_pt.x = robot_pose_curr.x;   
      temp_pt.y = robot_pose_curr.y;

      no_landmarks_in_range = find_landmarks_in_range_simulate(robot_pose_curr, landmark_map, &arr_indices);
      //printvec(arr_indices);//arr_indices are the indices of the landmarks detected at this particular pose
      for(int j=0;j<no_landmarks_in_range;j++)
      {
        temp_rel_vec.r = dist_pts(landmark_map[arr_indices[j]], temp_pt);
        temp_rel_vec.r = RandNormal() * FLAGS_range_stddev_simulate + temp_rel_vec.r;
        temp_rel_vec.theeta = find_angle(landmark_map[arr_indices[j]], temp_pt) - robot_pose_curr.theeta;
        temp_rel_vec.theeta = RandNormal() * FLAGS_range_stddev_simulate + temp_rel_vec.theeta;
        //std::cout<<temp_rel_vec.theeta<<" ";

        arr_temp.push_back(temp_rel_vec);
      }
     
      //std::cout<<std::endl;
      odometry_values->push_back(observed_odometry_value);
      sensor_readings->push_back(arr_temp);
      arr_indices.clear();
      arr_temp.clear();
    }

  }
  //  PrintState(n, parameter_block_odom, parameter_block_lm, landmark_map, sensor_readings);

  void PrintState(int n, double** parameter_block_odom, double** parameter_block_lm, 
                    vector<point_2d> landmark_map, vector<vector<rel_vec> > sensor_readings)
  {
    printf("pose:  loc.x   loc.y   loc.th   sensor.r   sensor.th\n");
    int sensor_reading_no = 0;
    for (int i = 0; i < n; ++i) 
    {
      for(int j=0;j<sensor_readings[i].size();j++)
      {
          //lm_index = findLandmark(i, j, parameter_block_odom, sensor_readings[i][j], landmark_map);
          
          printf("%-4d: %7.3f %7.3f %8.3f %10.3f %11.3f\n",
            static_cast<int>(i), parameter_block_odom[i][0], parameter_block_odom[i][1], parameter_block_odom[i][2],
            parameter_block_lm[sensor_reading_no][0], parameter_block_lm[sensor_reading_no][1]);

          sensor_reading_no++;
      }
    }
  }
}  

void constructMap(vector<vector<rel_vec> > sensor_readings, vector<point_2d> &landmark_map, vector<vector<int> > &lm_indices)
{
  int n = (sensor_readings[0]).size();
  point_2d temp;
  vector<int> temp_vec;
  for(int i=0;i<n;i++){
    temp.x = ((sensor_readings[0][i]).r)*cos(pi/4 + ((sensor_readings[0][i]).theeta));
    temp.y = ((sensor_readings[0][i]).r)*sin(pi/4 + ((sensor_readings[0][i]).theeta));
    landmark_map.push_back(temp);
    temp_vec.push_back(i);
  }
  lm_indices.push_back(temp_vec);
  return;
}

//findPosesWithThislm(lm_index, kundli_vector, i, pose_list);//the list of time stamps at which the robot saw this particular landmark
void findPosesWithThislm(int lm_index, vector<sensor_reading_kundli> kundli_vector, int pose_index, int j, vector<int> &pose_list, vector<point_2d> landmark_map)
{
  // std::cout<<"entering the functino findPosesWithThislm"<<std::endl;
  int n = kundli_vector.size();
  int i=0;
  while((kundli_vector[i].pose_index)<pose_index)
  {
    //std::cout<<"i:"<<i<<std::endl;
    if((kundli_vector[i].lm_no == lm_index))
      pose_list.push_back(kundli_vector[i].pose_index);
    i++;  
  }

  //std::cout<<std::endl<<"Pose("<<pose_index<<"), j:"<<j<<", Poses with LM:"<<lm_index<<" ("<<landmark_map[lm_index].x<<","<<landmark_map[lm_index].y<<") are: "<<std::endl;
  //printvec(pose_list);
  //std::cout<<"exiting the functino findPosesWithThislm"<<std::endl;
  return;
}

//    findandUpdatelandmarks(i, sensor_reading_index, sensor_readings[i], kundli_vector, landmark_map, parameter_block_odom);
//  updates kundli_vector and landmark_map 
void findandUpdatelandmarks(int i, int sensor_reading_index, vector<rel_vec> readings , vector<sensor_reading_kundli> &kundli_vector, vector<point_2d> &landmark_map, double** parameter_block_odom)
{
  int n = readings.size();
  //std::cout<<"In function findandUpdatelandmarks, n: "<<n<<std::endl;
  point_2d temp;
  int no_lms = landmark_map.size();
  double min = 10000;
  int min_index;
  //vector<int> temp_vec;
/*  if(n==0)
  {
    temp_vec.push_back(0);
    temp_vec.clear();
    return; 
  }
*/
  double* robot_pose_curr = parameter_block_odom[i];
  for(int i=0;i<n;i++)
  {
    //likely position of the landmark
    temp.x = robot_pose_curr[0] + (readings[i].r)*cos(readings[i].theeta + robot_pose_curr[2]);
    temp.y = robot_pose_curr[1] + (readings[i].r)*sin(readings[i].theeta + robot_pose_curr[2]);

    min = 10000;
    for(int j=0;j<no_lms;j++)
    {
      if(dist_pts(temp,landmark_map[j]) < min)
      {
        min = dist_pts(temp,landmark_map[j]);
        min_index = j;
      }
    }

    if(min<threshold_lm_dist){
      kundli_vector[sensor_reading_index].lm_no = min_index;
      //temp_vec.push_back(min_index);
    }

    else
    {
     // temp_vec.push_back(landmark_map.size());//because the last index is size-1
      kundli_vector[sensor_reading_index].lm_no = landmark_map.size();
      landmark_map.push_back(temp);
    }
  
    sensor_reading_index++;
  }

  //lm_indices.push_back(temp_vec);
  return;
}

//findCorrespondingSensorReading(pi, landmark_map, lm_index, parameter_block_odom[i], sensor_readings, kundli_vector);
rel_vec findCorrespondingSensorReading(int pi, vector<point_2d> landmark_map, int lm_index, double* robot_pose_curr, vector<vector<rel_vec> > sensor_readings, vector<sensor_reading_kundli> kundli_vector)
{
  int n = (sensor_readings[pi]).size();
  double min = 10000;
  point_2d temp;
  int min_index;
  /*for(int i=0;i<n;i++)
  {
    temp.x = robot_pose_curr[0] + (sensor_readings[pi][i].r)*cos(sensor_readings[pi][i].theeta + robot_pose_curr[2]);
    temp.y = robot_pose_curr[1] + (sensor_readings[pi][i].r)*sin(sensor_readings[pi][i].theeta + robot_pose_curr[2]);
    if(dist_pts(landmark_map[lm_index], temp) < min)
    {
      min = dist_pts(landmark_map[lm_index], temp);
      min_index = i;
    }
  }*/

  for(int i=0;i<kundli_vector.size();i++)
  {
    if((kundli_vector[i].pose_index == pi) && (kundli_vector[i].lm_no == lm_index)){
      rel_vec templ;
      templ.r = kundli_vector[i].r;
      templ.theeta = kundli_vector[i].theeta;
      return templ;
    }
  }
  // std::cout<<"Reading_index["<<min_index<<"]: ("<<sensor_readings[pi][min_index].r<<","<<sensor_readings[pi][min_index].theeta<<")"<<std::endl;
  //return sensor_readings[pi][min_index];
}

int find_index_of_lm(int lm_index, int pose_index, vector<vector<rel_vec> >sensor_readings, vector<point_2d> landmark_map, double** parameter_block_odom)
{
  point_2d lm = landmark_map[lm_index];
  point_2d lm_proj;
  double min = 1000000;
  int min_index;
  for(int i=0;i<sensor_readings[pose_index].size();i++)
  {
    lm_proj.x = parameter_block_odom[pose_index][0] + (sensor_readings[pi][i].r)*cos(sensor_readings[pi][i].theeta + parameter_block_odom[i][2]);
      lm_proj.y = parameter_block_odom[pose_index][1] + (sensor_readings[pi][i].r)*sin(sensor_readings[pi][i].theeta + parameter_block_odom[i][2]);
      if(dist_pts(landmark_map[lm_index], lm_proj) < min)
      {
        min = dist_pts(landmark_map[lm_index], lm_proj);
        min_index = i;
      }
  }

  return min_index;
}

void printsensorvals_from_param_block(double** parameter_block_lm, int n)
{
  for(int i=0;i<n;i++)
    std::cout<<parameter_block_lm[i][0]<<","<<parameter_block_lm[i][1]<<std::endl;
  return;
}

void printlm_indices(vector<vector<int> > lm_indices, vector<point_2d> landmark_map, double** parameter_block_odom)
{
  int n = lm_indices.size();
  int n_sub;
  std::cout<<"Landmarks for pose "<<n-1<<"("<<parameter_block_odom[n-1][0]<<","<<parameter_block_odom[n-1][1]<<","<<parameter_block_odom[n-1][2]<<") are: "<<std::endl;
  n_sub = lm_indices[n-1].size();
  for(int j=0;j<n_sub;j++)
    std::cout<<lm_indices[n-1][j]<<":("<<landmark_map[lm_indices[n-1][j]].x<<","<<landmark_map[lm_indices[n-1][j]].y<<") , ";
  std::cout<<std::endl;
  return;
}

void printKundli(vector<sensor_reading_kundli> kundli_vector)
{
  int n = kundli_vector.size();
  printf("r       theeta  lm_no   p_index  sens_total_in_no  sens_local_in_no \n");
  for(int i=0;i<n;i++){
    printf("%-7f %-7f %-7d %-9d %-15d %-15d\n", kundli_vector[i].r, kundli_vector[i].theeta, kundli_vector[i].lm_no, kundli_vector[i].pose_index, 
        kundli_vector[i].sensor_total_index_number, kundli_vector[i].sensor_local_index_number);
  }

  return;
}

//int sj_index = find_index_of_s_pi(kundli_vector, lm_index, pi);
int find_index_of_s_pi(vector<sensor_reading_kundli> kundli_vector, int lm_index, int pi)
{
  int i=0;
  while(i<kundli_vector.size())
  {
    if((kundli_vector[i].lm_no == lm_index) && (kundli_vector[i].pose_index == pi))
      return (kundli_vector[i].sensor_total_index_number); 
    i++;
  }

  return -1;
}

int main(int argc, char** argv) 
{
  google::InitGoogleLogging(argv[0]);
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  CHECK_GT(FLAGS_pose_separation, 0.0);
  CHECK_GT(FLAGS_odometry_stddev_x, 0.0);
  CHECK_GT(FLAGS_odometry_stddev_y, 0.0);
  CHECK_GT(FLAGS_odometry_stddev_theeta, 0.0);
  CHECK_GT(FLAGS_range_stddev, 0.0);

  vector<point_2d> landmark_map_for_sim;//only to generate sample sensor data
  std::ifstream fin; 
  fin.open("data_2d_lm2.txt");
  int no_landmarks;
  point_2d temp_point;
  fin>>no_landmarks;
  fin>>temp_point.x;
  while(!fin.eof())
  {
      fin>>temp_point.y;
      landmark_map_for_sim.push_back(temp_point);
      fin>>temp_point.x;
  }

  vector<td_pose> odometry_values;
  vector<vector<rel_vec> > sensor_readings;//range values of the landmarks within range
  std::cout<<std::endl<<std::endl<<"Simulating the robot environment"<<std::endl;
  SimulateRobot(&odometry_values, &sensor_readings, landmark_map_for_sim);//fill in odometry, sensor readings with a gaussian noise
  std::cout<<"Simulation of the robot environment done"<<std::endl;
  
  int n = odometry_values.size();//=num_steps
  //std::cout<<"Num of steps of robot: "<<n<<std::endl;
  td_pose robot_pose_curr;
  robot_pose_curr.x = 0; robot_pose_curr.y = 0; robot_pose_curr.theeta = (pi/4);

  std::ofstream robot_posns;
  robot_posns.open("robot_posns.txt");
  
  //(n+1) poses for n odometry values
  double** parameter_block_odom = (double**)malloc((n+1)*sizeof(double*));
  for(int i=0;i<=n;i++)
    parameter_block_odom[i] = (double*)malloc(3*sizeof(double));

  robot_posns<<robot_pose_curr.x<<","<<robot_pose_curr.y<<","<<robot_pose_curr.theeta<<std::endl;
  parameter_block_odom[0][0] = 0; parameter_block_odom[0][1] = 0;  parameter_block_odom[0][2] = (pi/4); 
  for(int i=1;i<=n;i++)
  { 
    robot_pose_curr.x += odometry_values[i-1].x;
    robot_pose_curr.y += odometry_values[i-1].y;
    robot_pose_curr.theeta += odometry_values[i-1].theeta;
    robot_pose_curr.theeta = remainder(robot_pose_curr.theeta, (2*pi));

    robot_posns<<robot_pose_curr.x<<","<<robot_pose_curr.y<<","<<robot_pose_curr.theeta<<std::endl;
    
    parameter_block_odom[i][0] = robot_pose_curr.x;
    parameter_block_odom[i][1] = robot_pose_curr.y;
    parameter_block_odom[i][2] = robot_pose_curr.theeta;
  }

  std::cout<<"Exiting....."<<std::endl;
  robot_posns.close();
  std::cout<<"Saved the odometry simulated values into robot_posns.txt"<<std::endl;
  
  std::ofstream sensor_observations;
  sensor_observations.open("sensor_observations.txt");
  int total_no_sensor_readings = 0;
  for(int i=0;i<=n;i++)
  {
    int n1 = sensor_readings[i].size();
    total_no_sensor_readings+=n1;
  }

  double** parameter_block_lm = (double**)malloc(total_no_sensor_readings*sizeof(double*));
  for(int i=0;i<total_no_sensor_readings;i++)
    parameter_block_lm[i] = (double*)malloc(2*sizeof(double));

  int sensor_reading_index =0;
  for(int i=0;i<=n;i++)
  {
    for(int j=0;j<sensor_readings[i].size();j++)
    {
      parameter_block_lm[sensor_reading_index][0] = sensor_readings[i][j].r;
      parameter_block_lm[sensor_reading_index][1] = sensor_readings[i][j].theeta;
      sensor_observations<<parameter_block_lm[sensor_reading_index][0]<<","<<parameter_block_lm[sensor_reading_index][1]<<std::endl;
      sensor_reading_index++;
    }
  } 

  sensor_observations.close();
  std::cout<<"Saved the sensor_observations simulated values into sensor_observations.txt"<<std::endl;

  std::cout<<"Original state"<<std::endl;
  PrintState(n, parameter_block_odom, parameter_block_lm, landmark_map_for_sim, sensor_readings);

  vector<point_2d> landmark_map;
  landmark_map.clear();
  vector<vector<int> > lm_indices;

  vector<sensor_reading_kundli > kundli_vector;
  sensor_reading_kundli temp;
  int curr_pose= 0;
  int index_local = 0;
  int index_global = 0;
  int n_sub = sensor_readings[0].size();
  while(index_global<total_no_sensor_readings)
  {
    index_local = 0;
    n_sub = sensor_readings[curr_pose].size();
    while(index_local<n_sub)
    {
      temp.r = sensor_readings[curr_pose][index_local].r;
      temp.theeta = sensor_readings[curr_pose][index_local].theeta;
      temp.pose_index = curr_pose;
      temp.sensor_total_index_number = index_global;
      temp.sensor_local_index_number = index_local;
      temp.lm_no = -1;
      kundli_vector.push_back(temp);

      index_local++;
      index_global++;
    }

    curr_pose++;
  } 

  //std::cout<<"kundli_vector: "<<std::endl;
  //printKundli(kundli_vector);
  
  findandUpdatelandmarks(0, 0, sensor_readings[0], kundli_vector, landmark_map, parameter_block_odom);
/*  std::cout<<"the kundli_vector for pose 0: "<<std::endl;
  int i_temp=0;
  while(kundli_vector[i_temp].pose_index<1){
    printf("%-7f %-7f %-7d %-9d %-15d %-15d\n", kundli_vector[i_temp].r, kundli_vector[i_temp].theeta, kundli_vector[i_temp].lm_no, kundli_vector[i_temp].pose_index, 
        kundli_vector[i_temp].sensor_total_index_number, kundli_vector[i_temp].sensor_local_index_number);
    i_temp++;
  }*/

  sensor_reading_index = (sensor_readings[0].size());
  std::cout<<"sensor_reading_index: "<<sensor_reading_index<<std::endl;
  ceres::Problem problem;
  int local_sr_index;
  point_2d lm_proj_temp;
  //i is the pose index, hence for the first pose, there are no odometry constraints and landmark constraints
  td_pose odometry_stddev;
  for(int i=1;i<=n;i++)
  {
    odometry_stddev.x = (FLAGS_odometry_stddev_x)*(abs(odometry_values[i-1].x) + abs(odometry_values[i-1].theeta));
    odometry_stddev.y = (FLAGS_odometry_stddev_y)*(abs(odometry_values[i-1].y) + abs(odometry_values[i-1].theeta));
    odometry_stddev.theeta = (FLAGS_odometry_stddev_theeta)*abs(odometry_values[i-1].theeta);
    //std::cout<<std::endl<<"i:"<<i<<", ("<<parameter_block_odom[i][0]<<","<<parameter_block_odom[i][1]<<","<<parameter_block_odom[i][2]<<")"<<std::endl;
    problem.AddResidualBlock(OdometryConstraint::Create(odometry_values[i-1], odometry_stddev), 
      NULL, parameter_block_odom[i-1], parameter_block_odom[i]);
    
    findandUpdatelandmarks(i, sensor_reading_index, sensor_readings[i], kundli_vector, landmark_map, parameter_block_odom);
    //printlm_indices(lm_indices, landmark_map, parameter_block_odom);

    for(int j=0;j<sensor_readings[i].size();j++)
    {
      lm_proj_temp.x = parameter_block_odom[i][0] + (sensor_readings[i][j].r)*cos(parameter_block_odom[i][2] + sensor_readings[i][j].theeta);
      lm_proj_temp.y = parameter_block_odom[i][1] + (sensor_readings[i][j].r)*sin(parameter_block_odom[i][2] + sensor_readings[i][j].theeta);
      //std::cout<<"j:"<<j<<", Projected Landmark: ("<<lm_proj_temp.x<<","<<lm_proj_temp.y<<")"<<std::endl;
      //int lm_index = lm_indices[i][j];
      int lm_index = kundli_vector[sensor_reading_index].lm_no;
      vector<int> pose_list;
      pose_list.clear();
      findPosesWithThislm(lm_index, kundli_vector, i, j, pose_list, landmark_map);//the list of time stamps at which the robot saw this particular landmark
      int n_poses = pose_list.size();
      for(int k=0;k<n_poses;k++)
      {
        //std::cout<<"k: "<<k<<std::endl;
        int pi = pose_list[k];//pi is the pose number(time stamp) at which also the landmark with index "lm_index" has been seen
        //std::cout<<"pi:"<<pi<<std::endl;
        int sj_index = find_index_of_s_pi(kundli_vector, lm_index, pi);
        //std::cout<<"sensor_reading_index: "<<sensor_reading_index<<",  sj_index: "<<sj_index<<std::endl;
        //find that sensor_reading(wrto- position of robot at time stamp "pi") which corresponds to the landmark under consideration(i.e. - landmark number "lm_index")
        //rel_vec sj = findCorrespondingSensorReading(pi, landmark_map, lm_index, parameter_block_odom[i], sensor_readings, kundli_vector);
         //std::cout<<"sj: ("<<sj.r<<","<<sj.theeta<<")"<<std::endl;
        //std::cout<<"Projected Landmark of the previous pose("<<pi<<") detected with the same LM: ("<<lm_proj_temp.x<<","<<lm_proj_temp.y<<")"<<std::endl;

        //local_sr_index = find_index_of_lm(lm_index, i, sensor_readings, landmark_map, parameter_block_odom);
        problem.AddResidualBlock(LandmarkConstraint::Create(), NULL, 
          parameter_block_odom[i], parameter_block_odom[pi], parameter_block_lm[sensor_reading_index], parameter_block_lm[sj_index]);
      }
      
      sensor_reading_index++;
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

  vector<point_2d> landmark_map_generated;
  for(int i=0;i<total_no_sensor_readings;i++)
  {
    point_2d proj_lm;
    int curr_pose_index = kundli_vector[i].pose_index;
    proj_lm.x = parameter_block_odom[curr_pose_index][0] + (parameter_block_lm[i][0])*cos(parameter_block_odom[curr_pose_index][2] + parameter_block_lm[i][1]);
    proj_lm.y = parameter_block_odom[curr_pose_index][1] + (parameter_block_lm[i][0])*sin(parameter_block_odom[curr_pose_index][2] + parameter_block_lm[i][1]);
    
    landmark_map_generated.push_back(proj_lm);
  }


  //std::cout<<"Final map: "<<std::endl;
  //printPointsVec(landmark_map);
  printf("Final values:\n");
  PrintState(n, parameter_block_odom, parameter_block_lm, landmark_map_generated, sensor_readings);

  std::ofstream lm_map_obtained;
  lm_map_obtained.open("lm_map_obtained.txt");
  for(int i=0;i<landmark_map.size();i++)
    lm_map_obtained<<landmark_map[i].x<<","<<landmark_map[i].y<<std::endl;
  lm_map_obtained.close();
  std::cout<<"Saved the landmark values generated into lm_map_obtained.txt"<<std::endl;

  std::ofstream generated_lms;
  generated_lms.open("generated_lms.txt");
  for(int i=0;i<landmark_map_generated.size();i++)
    generated_lms<<landmark_map_generated[i].x<<","<<landmark_map_generated[i].y<<std::endl;
  generated_lms.close();
  std::cout<<"Saved the landmark values generated into generated_lms.txt"<<std::endl;

  std::ofstream generated_robot_poses;
  generated_robot_poses.open("generated_robot_poses.txt");
  for(int i=0;i<=n;i++)
    generated_robot_poses<<parameter_block_odom[i][0]<<","<<parameter_block_odom[i][1]<<","<<parameter_block_odom[i][2]<<std::endl;
  generated_robot_poses.close();
  std::cout<<"Saved the generated robot poses into generated_robot_poses.txt"<<std::endl;

  return 0;
}


////////////////////////////////////////////////////////////////////int main() over//////////////////////////////////////////////////////////////////////
int find_landmarks_in_range_simulate(td_pose robot_pose_curr, vector<point_2d> landmark_map, vector<int>* arr_indices)
{
  int n = landmark_map.size();
  point_2d curr_point;
  double theeta_lm_wrto_robot_orientation;
  for(int i=0;i<n;i++)
  {
    curr_point.x = robot_pose_curr.x;
    curr_point.y = robot_pose_curr.y;

    if( dist_pts(curr_point, landmark_map[i]) < sensor_range)
    {
      theeta_lm_wrto_robot_orientation = find_angle(landmark_map[i], curr_point);
      theeta_lm_wrto_robot_orientation = theeta_lm_wrto_robot_orientation - robot_pose_curr.theeta;
      if(theeta_lm_wrto_robot_orientation > (-sensor_angle_range) && theeta_lm_wrto_robot_orientation < (sensor_angle_range))
        arr_indices->push_back(i);
    }
  }

  return arr_indices->size();
}


double dist_pts(point_2d pt1, point_2d pt2)
{
  return sqrt((pt2.y - pt1.y)*(pt2.y - pt1.y) + (pt2.x - pt1.x)*(pt2.x - pt1.x));
}

double abs(double a)
{
  if(a<0)
    return (-1*a);
  else 
    return a;
}

void printvec(vector<int> arr)
{
  int n = arr.size();
  for(int i=0;i<n;i++)
    std::cout<<arr[i]<<" ";
}

void printsensorvals(vector<vector<rel_vec> > sensor_readings)
{
  int n = sensor_readings.size();
  int n1;
  for(int i=0;i<n;i++)
  {
    n1 = (sensor_readings[i]).size();
    for(int j=0;j<n1;j++)
      std::cout<<i<<" "<<j<<" : "<<sensor_readings[i][j].r<<" "<<sensor_readings[i][j].theeta<<std::endl;
  }
  return;
}

double find_angle(point_2d pt2, point_2d pt1)//returns angle in (0,2pi)
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