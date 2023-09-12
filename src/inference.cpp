#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"
#include <vector>
#include <string>

using namespace std;

class ObservationListener
{
  public:
    vector<float> scan;
    vector<float> pose;
    vector<float> quat;
    float yaw;
    //void observationCallback();

  void CallbackLaserScan(const sensor_msgs::LaserScan::ConstPtr& msg)
  {
    scan.clear();
    scan = msg->intensities;
  }

  void CallbackPose(const geometry_msgs::PoseStamped::ConstPtr& msg)
  {
    pose.clear();
    pose.push_back(msg->pose.position.x);
    pose.push_back(msg->pose.position.y);
    pose.push_back(msg->pose.position.z);

    quat.clear();
    quat.push_back(msg->pose.orientation.x);
    quat.push_back(msg->pose.orientation.y);
    quat.push_back(msg->pose.orientation.z);
    quat.push_back(msg->pose.orientation.w);
  }

  void ObservationManager()
  {
    return;
  }
}

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
  ros::Subscriber lidar_sub = n.subscribe("lidar_scans", 1000, &ObservationListener::CallbackLaserScan, &obs_listener);
  ros::Publisher lidar_label_pub = n.advertise<sensor_msgs::LaserScan>("predicted_labels", 1000);
  ros::Rate loop_rate(10); //how fast do we get laser scans? can we trigger on new scan?

  // init onnx session
  string instanceName("onnx-inference");
  Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  string policy_name = "models/model.onnx";
  Ort::Session session(env, policy_name.c_str(), session_options);
  ROS_INFO("loaded model %s", policy_name);
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

  while (ros::ok())
  {
    sensor_msgs::LaserScan predicted_labels;
    ros::spinOnce();


    lidar_label_pub.publish(predicted_labels);
    loop_rate.sleep();
  }
    
  return 0;
}