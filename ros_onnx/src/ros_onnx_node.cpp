#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include "ros/ros.h"
#include "ros/package.h" 
#include "sensor_msgs/LaserScan.h"
#include "geometry_msgs/PoseStamped.h"
#include <torch/script.h>
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace torch::indexing;

struct Quaternion
{
    double w, x, y, z;
};

struct EulerAngles
{
    double roll, pitch, yaw;
};

EulerAngles Q2E(Quaternion q)
{
    EulerAngles angles;

    // roll (x-axis rotation)
    double sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
    double cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
    angles.roll = atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = sqrt(1 + 2 * (q.w * q.y - q.x * q.z));
    double cosp = sqrt(1 - 2 * (q.w * q.y - q.x * q.z));
    angles.pitch = 2 * atan2(sinp, cosp) - M_PI / 2;

    // yaw (z-axis rotation)
    double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
    angles.yaw = atan2(siny_cosp, cosy_cosp);

    return angles;
}

class ObservationListener
{
  public:
    vector<float> raw_scan;
    vector<float> scan;
    vector<float> pose;
    Quaternion quat;
    vector<float> yaw;

  void CallbackLaserScan(const sensor_msgs::LaserScan::ConstPtr& msg)
  {
    scan.clear();
    for (int i=0; i<msg->ranges.size(); ++i)
    {
      float r = msg->ranges[i];
      raw_scan.push_back(r);
      if (r > 20) r = 20;
      if ( r < 0) r = 0;
      scan.push_back(r/20);
    }
  }
  
  void CallbackPose(const geometry_msgs::PoseStamped::ConstPtr& msg)
  {
    pose.clear();
    pose.push_back(msg->pose.position.x);
    pose.push_back(msg->pose.position.y);
    pose.push_back(msg->pose.position.z);

    quat.x = msg->pose.orientation.x;
    quat.y = msg->pose.orientation.y;
    quat.z = msg->pose.orientation.z;
    quat.w = msg->pose.orientation.w;

    yaw.clear();
    yaw.push_back(Q2E(quat).yaw);
  }

  void ObservationManager()
  {
    return;
  }
};

template <typename T>
size_t vectorProduct(const vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, multiplies<T>());
}

int main(int argc, char **argv)
{
  //init ros nodes
  ObservationListener obs_listener;
  ros::init(argc, argv, "onnx_inference_node");
  ros::NodeHandle n;
  ros::Subscriber lidar_sub = n.subscribe("new_scan", 1000, &ObservationListener::CallbackLaserScan, &obs_listener);
  ros::Subscriber pose_sub = n.subscribe("localization_ros", 1000, &ObservationListener::CallbackPose, &obs_listener);
  ros::Publisher lidar_label_pub = n.advertise<sensor_msgs::LaserScan>("segmented_scan", 1000);
  ros::Rate loop_rate(10); //how fast do we get laser scans? can we trigger on new scan?

  // init onnx session
  string instanceName("onnx-inference");
  Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  string policy_name = "/home/kylem/Research/catkin_ws/src/ros_onnx/ros_onnx/models/model.onnx"; //ros::package::getPath("ros_onnx");
  Ort::Session session(env, policy_name.c_str(), session_options);
  ROS_INFO("loaded model %s", policy_name.c_str());
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  //inputs
  vector<Ort::TypeInfo> input_type_infos;
  vector<vector<int64_t>> input_dims;
  vector<size_t> input_sizes;
  vector<const char*> input_names;
  vector<Ort::Value> input_ort;
  
  size_t input_count = session.GetInputCount();
  for (auto i = 0; i < input_count; ++i)
  {
    input_type_infos.push_back(session.GetInputTypeInfo(i));
    auto input_tensor_info = input_type_infos.at(i).GetTensorTypeAndShapeInfo();
    input_dims.push_back(input_tensor_info.GetShape());
    if (input_dims.at(i).at(0) == -1) input_dims.at(i).at(0) = 1; //single batch for inference
    input_sizes.push_back(vectorProduct(input_dims.at(i)));
    Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(i, allocator);
    input_names.push_back(input_name_ptr.release());
  }

  //outputs
  vector<Ort::TypeInfo> output_type_infos;
  vector<vector<int64_t>> output_dims;
  vector<size_t> output_sizes;
  vector<const char*> output_names;
  vector<Ort::Value> output_ort;

  size_t output_count = session.GetOutputCount();
  for (auto i = 0; i < output_count; ++i)
  {
    output_type_infos.push_back(session.GetOutputTypeInfo(i));
    auto output_tensor_info = output_type_infos.at(i).GetTensorTypeAndShapeInfo();
    output_dims.push_back(output_tensor_info.GetShape());
    if (output_dims.at(i).at(0) == -1) output_dims.at(i).at(0) = 1; //single batch for inference
    output_sizes.push_back(vectorProduct(output_dims.at(i)));
    Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(i, allocator);
    output_names.push_back(output_name_ptr.release());
  }

  vector<float> pose_vec; //x,y,yaw

  torch::Tensor new_scan;
  torch::Tensor scans = torch::zeros({1,1,9,897});
  torch::Tensor new_label;
  torch::Tensor labels = torch::zeros({1,1,9,897});
  torch::Tensor new_pose;
  torch::Tensor new_yaw;
  torch::Tensor poses = torch::zeros({1,1,9,3});
  torch::Tensor outputs = torch::zeros({1,1,1,897});
  torch::Tensor exp_weighted_sum;
  auto options = torch::TensorOptions().dtype(torch::kFloat32);

  vector<float> weight_vec {1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8, 1};
  float weight_sum = 0;
  for (auto i = 0; i < 8; ++i) weight_sum += weight_vec[i];
  for (auto i = 0; i < 8; ++i) weight_vec[i]/weight_sum;
  torch::Tensor weights = torch::from_blob(weight_vec.data(), {8}, options);

  vector<float> class_labels;

  while (ros::ok())
  {
    sensor_msgs::LaserScan predicted_labels;
    ros::spinOnce();

    if (obs_listener.scan.size())
    {
      input_ort.clear();
      output_ort.clear();

      new_pose = torch::from_blob(obs_listener.pose.data(), {1,1,1,3}, options);
      new_yaw = torch::from_blob(obs_listener.yaw.data(), {1,1,1,1}, options);
      new_pose.index_put_({"...",-1}, new_yaw);
      poses = torch::cat({poses, new_pose}, 2).index({"...", Slice(1, None), Slice()});
      vector<float> pose_value (poses.data_ptr<float>(), poses.data_ptr<float>() + poses.numel());
      input_ort.push_back(Ort::Value::CreateTensor<float>(
        memory_info, pose_value.data(), input_sizes.at(0), input_dims.at(0).data(), input_dims.at(0).size()));

      //cout << "here" << endl;

      new_scan = torch::from_blob(obs_listener.scan.data(), {1,1,1,897}, options);
      scans = torch::cat({scans, new_scan}, 2).index({"...", Slice(1, None), Slice()});
      vector<float> scan_value (scans.data_ptr<float>(), scans.data_ptr<float>() + scans.numel());
      input_ort.push_back(Ort::Value::CreateTensor<float>(
        memory_info, scan_value.data(), input_sizes.at(1), input_dims.at(1).data(), input_dims.at(1).size()));

      labels = labels.index({"...", Slice(1, None), Slice()}); //remove 0th col
      exp_weighted_sum = torch::einsum("ijkl,k->ijl", {labels,weights});
      labels = torch::cat({labels, torch::zeros({1,1,1,897})}, 2);
      labels.index_put_({"...",-1,Slice()}, exp_weighted_sum);
      vector<float> label_value (labels.data_ptr<float>(), labels.data_ptr<float>() + labels.numel());
      input_ort.push_back(Ort::Value::CreateTensor<float>(
        memory_info, label_value.data(), input_sizes.at(2), input_dims.at(2).data(), input_dims.at(2).size()));

      vector<float> output_value (outputs.data_ptr<float>(), outputs.data_ptr<float>() + outputs.numel());
      output_ort.push_back(Ort::Value::CreateTensor<float>(
        memory_info, output_value.data(), output_sizes.at(0), output_dims.at(0).data(), output_dims.at(0).size()));
      
      session.Run(Ort::RunOptions{nullptr},
        input_names.data(), input_ort.data(), input_count,
        output_names.data(), output_ort.data(), output_count);

      new_label = torch::from_blob(output_value.data(), {1,1,1,897}, options);
      labels.index_put_({"...", -1, Slice()}, new_label);

      for (auto i = 0; i < 897; ++i)
      {
        if (new_label.index({"...",i}).item<float>() >= 0) class_labels.push_back(1);
        else class_labels.push_back(-1);
      }

      predicted_labels.intensities = class_labels;
      predicted_labels.ranges = obs_listener.raw_scan;
      predicted_labels.header.frame_id = "map";
      predicted_labels.header.stamp = ros::Time::now();
      predicted_labels.angle_min = -3.14;
      predicted_labels.angle_max = 3.14;
      predicted_labels.angle_increment = 0.007;
      lidar_label_pub.publish(predicted_labels);
      loop_rate.sleep();
    }
  }
    
  return 0;
}