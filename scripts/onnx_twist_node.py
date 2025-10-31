#!/usr/bin/env python3

"""
ROS node: Subscribe to a SpaceMouse (Twist), optionally subscribe to a policy
observation vector, run an ONNX model to transform it, and publish a resulting Twist.

Parameters (ROS private ~ namespace):
    - input_topic           (str): Input Twist topic. Default: "/spacenav/twist"
    - output_topic          (str): Output Twist topic. Default: "/model_twist"
    - model_path            (str): Path to ONNX model. If empty or loading fails, pass-through.
    - rate_hz               (int): Publish rate if using loop-based publishing. Default: 50
    - pad_value           (float): Pad value for Twist when expanding to model input size. Default: 0.0
    - clip_output          (bool): If True, clip output size to 6 dims for Twist. Default: True
    - obs_topic             (str): Optional observation topic (Float32MultiArray). Default: "" (disabled)
    - obs_required          (bool): If True, require observation before running model. Default: False
    - obs_pad_value       (float): Pad value for observation vector. Default: 0.0
    - concat_single_input   (bool): If model has a single input, concatenate [twist, obs]. Default: True

Notes:
    - This node assumes model inputs are flat vectors. If shapes differ, the node pads/truncates as needed.
    - If no model is provided or onnxruntime is unavailable, the node forwards the input Twist to the output topic.
    - Observation subscription is optional; enable by setting ~obs_topic.
"""

"""
TODO:
- Have the node listen to the robot observations
- Convert the twist into a pose, for input into the policy
    - Take the end effector pose and add v*dt. dt = 1/rate_hz -- in this case 50Hz = 0.02
    - Need to figure out the frame of the end effector pose so we can make sure we are computing the goal pose correctly
        - We think it is world frame. The end effector is the ee_frame not the end_effector_frame
- Put the goal pose in and get the joint positions out of the policy
    - Convert the joint positions so that they are in twist for the robot controller
"""



from typing import Optional, List
import os

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ort = None


def twist_to_vec(msg: Twist) -> np.ndarray:
    """Convert Twist to a 6D numpy vector [lin.x, lin.y, lin.z, ang.x, ang.y, ang.z]."""
    return np.array([
        msg.linear.x,
        msg.linear.y,
        msg.linear.z,
        msg.angular.x,
        msg.angular.y,
        msg.angular.z,
    ], dtype=np.float32)


def vec_to_twist(vec: np.ndarray) -> Twist:
    """Map a vector's first 6 elements back to a Twist."""
    out = Twist()
    v = vec.astype(np.float32).flatten()
    # Ensure length >= 6
    if v.size < 6:
        v = np.pad(v, (0, 6 - v.size), mode='constant')
    out.linear.x = float(v[0])
    out.linear.y = float(v[1])
    out.linear.z = float(v[2])
    out.angular.x = float(v[3])
    out.angular.y = float(v[4])
    out.angular.z = float(v[5])
    return out


class OnnxTwistNode:
    def __init__(self) -> None:
        self.input_topic = rospy.get_param('~input_topic', '/spacenav/twist')
        self.output_topic = rospy.get_param('~output_topic', '/model_twist')
        self.model_path = rospy.get_param('~model_path', '')
        self.rate_hz = int(rospy.get_param('~rate_hz', 50))
        self.pad_value = float(rospy.get_param('~pad_value', 0.0))
        self.clip_output = bool(rospy.get_param('~clip_output', True))
        # Observation-related params
        self.obs_topic = rospy.get_param('~obs_topic', '')
        self.obs_required = bool(rospy.get_param('~obs_required', False))
        self.obs_pad_value = float(rospy.get_param('~obs_pad_value', 0.0))
        self.concat_single_input = bool(rospy.get_param('~concat_single_input', True))

        self.pub = rospy.Publisher(self.output_topic, Twist, queue_size=10)
        self.sub = rospy.Subscriber(self.input_topic, Twist, self._on_twist, queue_size=10)
        self.sub_obs = None
        self.obs_vec = None
        if isinstance(self.obs_topic, str) and len(self.obs_topic) > 0:
            self.sub_obs = rospy.Subscriber(self.obs_topic, Float32MultiArray, self._on_obs, queue_size=10)

        # Inference session handle (created if onnxruntime and model are available)
        self.session = None
        self.input_names = []
        self.input_sizes = []

        self._load_model()
        if self.sub_obs is not None:
            rospy.loginfo('onnx_twist_node: listening twist=%s, obs=%s, publishing to %s',
                          self.input_topic, self.obs_topic, self.output_topic)
        else:
            rospy.loginfo('onnx_twist_node: listening twist=%s, publishing to %s',
                          self.input_topic, self.output_topic)

    def _load_model(self) -> None:
        if not self.model_path:
            # Try to default to package's models/model.onnx if available
            try:
                import rospkg  # type: ignore
                pkg_path = rospkg.RosPack().get_path('ros_onnx')
                default_model = os.path.join(pkg_path, 'models', 'model.onnx')
                if os.path.isfile(default_model):
                    self.model_path = default_model
            except Exception:
                pass

        if ort is None:
            rospy.logwarn('onnxruntime not available; operating in pass-through mode.')
            return

        if not self.model_path or not os.path.isfile(self.model_path):
            rospy.logwarn('No valid model_path provided; operating in pass-through mode. model_path=%s', self.model_path)
            return

        try:
            self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            self.input_names = [i.name for i in self.session.get_inputs()]
            # Track expected flat size from last dim if static; None means dynamic/unknown
            self.input_sizes = []
            for i in self.session.get_inputs():
                shape = list(i.shape)
                last_dim = shape[-1] if shape and isinstance(shape[-1], int) else None
                self.input_sizes.append(last_dim)
            rospy.loginfo('Loaded ONNX model: %s | inputs=%s', self.model_path, self.input_names)
        except Exception as e:
            rospy.logerr('Failed to load ONNX model: %s', e)
            self.session = None

    def _prepare_input(self, x6: np.ndarray, obs: Optional[np.ndarray]) -> dict:
        """Prepare ONNX input dict from Twist (6D) and optional observation vector.

        Mapping policy:
          - If the model has a single input:
              * If obs is provided and concat_single_input is True, concatenate [x6, obs].
              * Else, use x6 alone.
          - If the model has 2+ inputs:
              * Input[0] <= x6 (Twist)
              * Input[1] <= obs (or zeros if not available)
              * Remaining inputs <= x6 by default
        Each input is padded/truncated to match its expected last dimension when static.
        """
        if self.session is None or not self.input_names:
            return {}

        feeds = {}
        num_inputs = len(self.input_names)

        def fit(vec: np.ndarray, expected_last: Optional[int], pad_value: float) -> np.ndarray:
            v = vec.astype(np.float32).flatten()
            if expected_last is None:
                return v[None, ...]
            # sanitize
            if expected_last <= 0:
                expected_last = v.size
            if v.size < expected_last:
                pad = np.full((expected_last - v.size,), pad_value, dtype=np.float32)
                v2 = np.concatenate([v, pad], axis=0)
            elif v.size > expected_last:
                v2 = v[:expected_last]
            else:
                v2 = v
            return v2[None, ...]

        if num_inputs == 1:
            name = self.input_names[0]
            expected_last = self.input_sizes[0]
            merged = x6
            if obs is not None and self.concat_single_input:
                merged = np.concatenate([x6, obs.astype(np.float32).flatten()], axis=0)
            feeds[name] = fit(merged, expected_last, self.pad_value)
            return feeds

        # 2 or more inputs
        for idx, name in enumerate(self.input_names):
            expected_last = self.input_sizes[idx]
            if idx == 0:
                vec = x6
                feeds[name] = fit(vec, expected_last, self.pad_value)
            elif idx == 1:
                if obs is not None:
                    vec = obs.astype(np.float32).flatten()
                    feeds[name] = fit(vec, expected_last, self.obs_pad_value)
                else:
                    # no obs available yet
                    if expected_last is None:
                        vec = np.zeros((6,), dtype=np.float32)
                    else:
                        vec = np.full((max(1, expected_last),), self.obs_pad_value, dtype=np.float32)
                    feeds[name] = fit(vec, expected_last, self.obs_pad_value)
            else:
                vec = x6
                feeds[name] = fit(vec, expected_last, self.pad_value)
        return feeds

    def _run_model(self, x6: np.ndarray, obs: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if self.session is None:
            return None
        try:
            feeds = self._prepare_input(x6, obs)
            if not feeds:
                return None
            outputs = self.session.run(None, feeds)
            if not outputs:
                return None
            out = np.ravel(outputs[0]).astype(np.float32)
            if self.clip_output and out.size > 6:
                out = out[:6]
            return out
        except Exception as e:
            rospy.logerr_throttle(2.0, 'Inference error: %s', e)
            return None

    def _on_obs(self, msg: Float32MultiArray) -> None:
        try:
            arr = np.asarray(msg.data, dtype=np.float32)
            self.obs_vec = arr.copy()
        except Exception:
            # Keep previous observation on failure
            pass

    def _on_twist(self, msg: Twist) -> None:
        x = twist_to_vec(msg)
        # Enforce observation requirement if configured
        if self.obs_required and self.sub_obs is not None and self.obs_vec is None:
            # No observation yet; pass-through
            self.pub.publish(msg)
            return

        y = self._run_model(x, self.obs_vec)
        if y is None:
            # Pass-through when model not available or error
            self.pub.publish(msg)
            return
        out_msg = vec_to_twist(y)
        self.pub.publish(out_msg)


def main() -> None:
    rospy.init_node('onnx_twist_node', anonymous=False)
    _ = OnnxTwistNode()
    rospy.spin()


if __name__ == '__main__':
    main()
