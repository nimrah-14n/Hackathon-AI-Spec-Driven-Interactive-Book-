---
sidebar_position: 8
title: "URDF & Humanoid Robot Modeling"
---

# 🤖 URDF & Humanoid Robot Modeling 🧍

## Learning Outcomes 🎯
By the end of this chapter, the learner will:
- **Understand** 🤔 the fundamentals of URDF (Unified Robot Description Format) for robot modeling
- **Create** 🔧 complex robot models with multiple joints, links, and sensors
- **Implement** ⚙️ humanoid robot models with proper kinematic chains
- **Integrate** 🔗 URDF models with ROS 2 simulation and control systems

## Introduction to URDF 🌟

Unified Robot Description Format (URDF) is the standard XML-based format for representing robot models in ROS. It provides a comprehensive way to describe a robot's physical properties, including its kinematic structure, visual appearance, collision geometry, and sensor placements. URDF serves as the blueprint for robots in ROS-based systems, defining everything from the robot's physical dimensions to its joint configurations and sensor mounting points.

For humanoid robots, URDF becomes particularly important as it must accurately represent the complex kinematic structure that mimics human-like movement capabilities. A well-designed humanoid URDF model includes the torso, head, arms with shoulders, elbows, and wrists, legs with hips, knees, and ankles, and proper joint constraints that allow for human-like mobility while maintaining structural stability.

URDF models serve multiple purposes in the robotics pipeline:
- **Simulation** 🌐: Providing accurate models for physics simulation in Gazebo, Isaac Sim, or other simulators
- **Visualization** 👁️: Enabling 3D visualization in RViz and other tools
- **Kinematics** 🧮: Supporting forward and inverse kinematics calculations
- **Collision Detection** ⚠️: Defining collision geometry for safety and planning
- **Control** ⚙️: Providing joint information for control system integration

The structure of a URDF file reflects the robot's kinematic tree, where each link is connected to its parent via joints. This tree structure defines the robot's degrees of freedom and movement capabilities. For humanoid robots, this structure typically includes multiple serial chains representing limbs, with appropriate joint types and limits that enable human-like motion patterns.

Modern robotics applications increasingly require detailed and accurate URDF models that can seamlessly transition between simulation and real hardware. The model must account for physical constraints, sensor placements, and the dynamic properties that affect robot behavior. Isaac Sim enhances this capability by providing high-fidelity physics simulation and rendering that closely matches real-world robot behavior.

## URDF Fundamentals 🏗️

### Core Elements 🧩

A URDF model consists of several fundamental elements:

- **Links** 🔗: Represent rigid parts of the robot (bodies, chassis, limbs)
- **Joints** ⚙️: Define connections between links with specific degrees of freedom
- **Materials** 🎨: Define visual appearance properties
- **Gazebos** 🌐: Simulation-specific extensions and plugins
- **Transmissions** ⚙️: Define how actuators connect to joints
- **Sensors** 🔍: Define sensor mounting and properties

### Link Structure

Links represent rigid bodies in the robot model:

```xml
<link name="link_name">
  <!-- Inertial properties for physics simulation -->
  <inertial>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <mass value="1.0"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
  </inertial>

  <!-- Visual properties for rendering -->
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <cylinder length="0.1" radius="0.05"/>
    </geometry>
    <material name="blue">
      <color rgba="0 0 1 1"/>
    </material>
  </visual>

  <!-- Collision properties for physics -->
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <cylinder length="0.1" radius="0.05"/>
    </geometry>
  </collision>
</link>
```

### Joint Definitions ⚙️

Joints connect links and define their relative motion:

```xml
<joint name="joint_name" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

Joint types include:
- **Revolute** 🔄: Rotational joint with limited range
- **Continuous** 🔄: Rotational joint with unlimited range
- **Prismatic** ↔️: Linear sliding joint
- **Fixed** 🔒: No relative motion between links
- **Floating** 🌊: Six degrees of freedom
- **Planar** 📐: Motion constrained to a plane

## Humanoid Robot Modeling 🧍

### Kinematic Structure 🏗️

Humanoid robots require careful modeling of human-like kinematic chains. The typical humanoid structure includes:

- **Torso** 🧍: Central body with head attachment
- **Head** 🧠: With neck joint and sensor mounting
- **Arms** 🦾: Shoulders, elbows, wrists, and optional hands
- **Legs** 🦿: Hips, knees, ankles, and feet
- **Additional joints** ⚙️: Spine, waist, or other anthropomorphic features

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">

  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>

    <visual>
      <geometry>
        <box size="0.3 0.2 0.1"/>
      </geometry>
    </visual>

    <collision>
      <geometry>
        <box size="0.3 0.2 0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Torso -->
  <link name="torso">
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0.3"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.1"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.3"/>
      <geometry>
        <capsule length="0.4" radius="0.1"/>
      </geometry>
    </visual>

    <collision>
      <origin xyz="0 0 0.3"/>
      <geometry>
        <capsule length="0.4" radius="0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Torso joint -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.05"/>
  </joint>

  <!-- Head -->
  <link name="head">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0.1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.1"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </visual>

    <collision>
      <origin xyz="0 0 0.1"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Neck joint -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.5"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.785" upper="0.785" effort="10" velocity="2.0"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_upper_arm">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0 -0.1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.005"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.1"/>
      <geometry>
        <capsule length="0.2" radius="0.05"/>
      </geometry>
    </visual>

    <collision>
      <origin xyz="0 0 -0.1"/>
      <geometry>
        <capsule length="0.2" radius="0.05"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.1 0.15 0.4"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
  </joint>

  <!-- Additional arm links and joints would continue similarly -->

</robot>
```

### Proper Inertial Properties 📊

Accurate inertial properties are crucial for realistic physics simulation:

```xml
<inertial>
  <!-- Mass in kilograms -->
  <mass value="2.5"/>

  <!-- Origin relative to link frame -->
  <origin xyz="0.0 0.0 0.1" rpy="0 0 0"/>

  <!-- Inertia tensor (calculated based on geometry) -->
  <inertia
    ixx="0.02" ixy="0.0" ixz="0.0"
    iyy="0.03" iyz="0.0"
    izz="0.01"/>
</inertial>
```

The inertia tensor values depend on the shape and mass distribution of the link. For common shapes:
- **Box**: Ixx = m*(h² + d²)/12, Iyy = m*(w² + d²)/12, Izz = m*(w² + h²)/12
- **Cylinder**: Ixx = Iyy = m*(3*r² + h²)/12, Izz = m*r²/2
- **Sphere**: Ixx = Iyy = Izz = 2*m*r²/5

Where m=mass, w=width, h=height, d=depth, r=radius, h=height.

## Advanced URDF Features 🔧

### Transmission Definitions ⚙️

Defining how actuators connect to joints:

```xml
<transmission name="left_elbow_transmission">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="left_elbow_joint">
    <hardwareInterface>PositionJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_elbow_motor">
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

### Sensor Integration 🔍

Adding sensors to the robot model:

```xml
<gazebo reference="head">
  <!-- RGB-D Camera -->
  <sensor type="depth" name="rgbd_camera">
    <pose>0.05 0 0.05 0 0.5 0</pose>
    <visualize>true</visualize>
    <update_rate>30</update_rate>
    <camera>
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>head_camera_optical_frame</frame_name>
      <min_depth>0.1</min_depth>
      <max_depth>10.0</max_depth>
    </plugin>
  </sensor>

  <!-- IMU Sensor -->
  <sensor type="imu" name="imu_sensor">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>false</visualize>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.001</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.001</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.001</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>
```

### Mesh Geometry 🧩

For complex geometries, using mesh files:

```xml
<visual>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <mesh filename="package://humanoid_description/meshes/head.stl" scale="1 1 1"/>
  </geometry>
  <material name="light_gray">
    <color rgba="0.7 0.7 0.7 1.0"/>
  </material>
</visual>

<collision>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <mesh filename="package://humanoid_description/meshes/head_collision.stl" scale="1 1 1"/>
  </geometry>
</collision>
```

Often, simplified collision meshes are used for better performance while visual meshes can be more detailed.

## Isaac Sim Integration 🌐

### Isaac Sim URDF Support 🤖

Isaac Sim provides excellent support for URDF models with enhanced physics and rendering:

```python
# Example Python script to load URDF in Isaac Sim
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.articulations import Articulation
import carb

class IsaacSimHumanoidLoader:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.assets_root_path = get_assets_root_path()

    def load_urdf_robot(self, urdf_path, prim_path="/World/HumanoidRobot"):
        """Load URDF robot into Isaac Sim"""
        # Add URDF to stage
        add_reference_to_stage(
            usd_path=urdf_path,
            prim_path=prim_path
        )

        # Create articulation from loaded robot
        robot_articulation = self.world.scene.add(
            Articulation(
                prim_path=prim_path,
                name="humanoid_robot"
            )
        )

        return robot_articulation

    def setup_physics_properties(self, robot_prim_path):
        """Configure physics properties for realistic simulation"""
        # Access the robot prim
        robot_prim = get_prim_at_path(robot_prim_path)

        # Set up joint limits and stiffness if needed
        # Isaac Sim will automatically parse joint limits from URDF
        # but you can override them if needed

        # Configure friction and restitution properties
        # for more realistic contact behavior
        pass

    def add_sensors_to_robot(self, robot_prim_path):
        """Add Isaac Sim sensors to the robot"""
        # Add camera sensors
        # Add LIDAR sensors
        # Add IMU sensors
        # Add force/torque sensors
        # Implementation details...
        pass

    def run_simulation(self):
        """Run the simulation with the humanoid robot"""
        self.world.reset()

        # Main simulation loop
        for i in range(10000):  # Run for 10000 steps
            self.world.step(render=True)

            # Get robot state
            robot_state = self.get_robot_state()

            # Apply control commands if needed
            # self.apply_control_commands(robot_state)

        return self.world
```

### Isaac Sim Specific Extensions 🧩

Enhancing URDF with Isaac Sim-specific features:

```xml
<gazebo reference="left_hand">
  <!-- Isaac Sim specific properties -->
  <disable_gravity>true</disable_gravity>
  <self_collide>false</self_collide>

  <!-- Contact sensor for tactile feedback -->
  <sensor type="contact" name="hand_contact_sensor">
    <always_on>true</always_on>
    <update_rate>1000</update_rate>
    <contact>
      <collision>left_hand_collision</collision>
    </contact>
  </sensor>

  <!-- Isaac Sim plugin for enhanced physics -->
  <plugin name="contact_sensor" filename="libIsaacSimContactSensor.so">
    <sensor_name>left_hand_contact_sensor</sensor_name>
    <threshold>0.1</threshold>
  </plugin>
</gazebo>

<!-- Material properties for Isaac Sim rendering -->
<gazebo reference="torso_visual">
  <material>Gazebo/Blue</material>
  <script_uri>Omni/Basic/Flat</script_uri>
  <script_name>OmniBasicFlat</script_name>
</gazebo>
```

## Kinematics and Dynamics 🧮

### Forward and Inverse Kinematics 🔄

URDF models enable both forward and inverse kinematics calculations:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
from urdf_parser_py.urdf import URDF

class HumanoidKinematics:
    def __init__(self, urdf_file):
        self.robot = URDF.from_xml_file(urdf_file)
        self.chain = self.robot.get_chain('base_link', 'left_hand')

    def forward_kinematics(self, joint_angles):
        """Calculate end-effector position from joint angles"""
        # Calculate transformation matrices for each joint
        transforms = []
        current_transform = np.eye(4)

        for i, (link_name, joint_name) in enumerate(zip(self.chain[1:], self.chain[:-1])):
            joint = self.robot.joint_map[joint_name]

            # Calculate joint transformation
            if joint.type == 'revolute':
                angle = joint_angles[i]
                # Calculate rotation matrix for revolute joint
                axis = joint.axis
                rot_matrix = self.angle_axis_to_rotation_matrix(axis, angle)

                # Apply translation and rotation
                translation = joint.origin.xyz
                transform = np.eye(4)
                transform[:3, :3] = rot_matrix
                transform[:3, 3] = translation

                current_transform = current_transform @ transform
            elif joint.type == 'prismatic':
                # Handle prismatic joints
                pass

        return current_transform

    def angle_axis_to_rotation_matrix(self, axis, angle):
        """Convert angle-axis representation to rotation matrix"""
        axis = np.array(axis) / np.linalg.norm(axis)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # Rodrigues' rotation formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        R = np.eye(3) + sin_angle * K + (1 - cos_angle) * K @ K
        return R

    def inverse_kinematics(self, target_pose, initial_angles):
        """Calculate joint angles for desired end-effector pose"""
        # Implement inverse kinematics algorithm
        # This could use analytical methods for simple chains
        # or numerical methods (Jacobian transpose, pseudoinverse, etc.) for complex chains

        # Example using Jacobian-based method
        current_pose = self.forward_kinematics(initial_angles)
        error = self.calculate_pose_error(current_pose, target_pose)

        # Iteratively update joint angles
        joint_angles = initial_angles.copy()
        max_iterations = 100
        tolerance = 0.001

        for iteration in range(max_iterations):
            if np.linalg.norm(error) < tolerance:
                break

            jacobian = self.calculate_jacobian(joint_angles)
            delta_thetas = np.linalg.pinv(jacobian) @ error

            joint_angles += delta_thetas
            current_pose = self.forward_kinematics(joint_angles)
            error = self.calculate_pose_error(current_pose, target_pose)

        return joint_angles

    def calculate_jacobian(self, joint_angles):
        """Calculate geometric Jacobian for the kinematic chain"""
        # Calculate Jacobian matrix
        # Implementation details...
        pass

    def calculate_pose_error(self, current_pose, target_pose):
        """Calculate error between current and target poses"""
        pos_error = target_pose[:3, 3] - current_pose[:3, 3]

        # Calculate orientation error using rotation matrix difference
        R_current = current_pose[:3, :3]
        R_target = target_pose[:3, :3]

        # Convert to quaternion difference
        R_diff = R_target @ R_current.T
        rotation_vec = R.from_matrix(R_diff).as_rotvec()

        error = np.concatenate([pos_error, rotation_vec])
        return error
```

### Dynamic Simulation Considerations ⚙️

For realistic dynamic simulation:

```python
class DynamicSimulationSetup:
    def __init__(self, robot_urdf):
        self.urdf = robot_urdf
        self.gravity = [0, 0, -9.81]

    def calculate_dynamic_parameters(self):
        """Calculate dynamic parameters for simulation"""
        # Calculate mass matrix
        # Calculate Coriolis and centrifugal forces
        # Calculate gravitational forces
        # Calculate actuator dynamics

        # These parameters affect how the robot moves and responds to forces
        pass

    def setup_control_gains(self):
        """Set up PID controller gains for each joint"""
        control_gains = {}

        for joint_name, joint in self.urdf.joints.items():
            if joint.type in ['revolute', 'prismatic']:
                # Set appropriate gains based on joint type and load
                control_gains[joint_name] = {
                    'p': self.calculate_p_gain(joint),
                    'i': self.calculate_i_gain(joint),
                    'd': self.calculate_d_gain(joint)
                }

        return control_gains

    def calculate_p_gain(self, joint):
        """Calculate proportional gain based on joint properties"""
        # Calculate based on joint load, speed requirements, etc.
        # Heavier joints may need higher gains
        # Higher speed joints may need higher gains
        # Implementation details...
        return 100.0  # Default value

    def calculate_damping_ratios(self):
        """Calculate appropriate damping ratios for joints"""
        # Proper damping prevents oscillations
        # Too little damping causes oscillations
        # Too much damping slows response
        damping_ratios = {}

        for joint_name, joint in self.urdf.joints.items():
            if joint.type in ['revolute', 'prismatic']:
                # Calculate based on joint characteristics
                damping_ratios[joint_name] = 0.7  # Critical damping ratio

        return damping_ratios
```

## Best Practices for Humanoid Modeling ✅

### Design Principles 🏗️

When creating humanoid robot models, follow these best practices:

1. **Anthropomorphic Proportions** 🧍: Use realistic human-like proportions for believable motion
2. **Proper Joint Limits** ⚙️: Set realistic joint limits that match human capabilities
3. **Balanced Inertial Properties** 📊: Ensure realistic mass distribution for stable simulation
4. **Collision-Free Design** ⚠️: Prevent self-collision in typical poses
5. **Modular Structure** 🧩: Organize model in logical subassemblies

### Performance Optimization ⚡

- **Mesh Simplification** 🧹: Use simpler collision meshes than visual meshes
- **Level of Detail** 🎯: Implement different detail levels for different simulation needs
- **Joint Limit Optimization** ⚙️: Set appropriate limits to prevent extreme poses
- **Mass Distribution** 📊: Balance realism with simulation stability

### Simulation Stability 🛡️

- **Consistent Units** 📏: Use consistent units throughout the model
- **Realistic Inertias** 📊: Calculate inertial properties based on actual geometry
- **Proper Scaling** 📏: Ensure model is properly scaled for physics simulation
- **Joint Damping** ⚙️: Apply appropriate damping to prevent unrealistic oscillations

## Troubleshooting Common Issues 🔧

### Self-Collision Problems ⚠️
- **Solution** 🛠️: Carefully adjust joint limits and collision geometries
- **Diagnosis** 🔍: Use visualization tools to identify collision points
- **Prevention** 🛡️: Add collision checking to forward kinematics

### Physics Instability 🌊
- **Solution** 🛠️: Adjust mass properties and joint damping
- **Diagnosis** 🔍: Check for unrealistic inertial tensors
- **Prevention** 🛡️: Use realistic physical properties

### Kinematic Issues 🧮
- **Solution** 🛠️: Verify joint axis directions and limits
- **Diagnosis** 🔍: Test individual joint ranges of motion
- **Prevention** 🛡️: Plan kinematic structure before modeling

## Tools and Utilities 🔧

### URDF Development Tools 🛠️
- **URDF Editor** 🖼️: Visual tools for creating and modifying URDF
- **RViz** 👁️: Visualization of robot models and kinematics
- **Gazebo/Isaac Sim** 🌐: Physics simulation and testing
- **URDF Validator** ✅: Check URDF syntax and structure

### Visualization and Debugging 🔍
- **Robot State Publisher** 👁️: Visualize joint positions in RViz
- **TF Tree** 🌳: Examine coordinate frame relationships
- **Joint State Publisher** ⚙️: Simulate joint movements
- **Forward/Inverse Kinematics Tools** 🔄: Test kinematic solutions

## Future Directions 🔮

### Advanced Modeling Techniques 🚀
- **Soft Body Simulation** 🧊: Modeling flexible components
- **Variable Stiffness** ⚙️: Adjustable joint properties
- **Morphological Adaptation** 🔄: Self-changing robot structures
- **Bio-inspired Design** 🧬: More human-like movement capabilities

### Integration with AI 🧠
- **Learning-based Modeling** 📚: AI-generated robot designs
- **Adaptive Simulation** 🔄: Simulation parameters that adapt to real performance
- **Digital Twins** 🪞: Real-time synchronized simulation models
- **Generative Design** 🎨: AI-assisted robot design optimization

## Summary 📝

URDF serves as the foundational language for robot modeling in ROS-based systems, providing a comprehensive framework for describing robot geometry, kinematics, and dynamics. For humanoid robots, proper URDF modeling requires careful attention to anthropomorphic proportions, realistic joint constraints, and accurate physical properties that enable both realistic simulation and effective control.

The integration of URDF models with Isaac Sim provides enhanced physics simulation and rendering capabilities that closely match real-world robot behavior. This integration enables developers to test and validate humanoid robot designs in high-fidelity simulation environments before deploying to physical hardware.

Modern humanoid robot modeling combines traditional mechanical engineering principles with advanced simulation techniques to create robots that can effectively operate in human-centric environments. The success of these robots depends heavily on accurate and well-designed URDF models that properly represent the physical and kinematic properties of the real robot.

The next chapter will explore Launch Files and Configuration Management for organizing and launching complex robotic systems.