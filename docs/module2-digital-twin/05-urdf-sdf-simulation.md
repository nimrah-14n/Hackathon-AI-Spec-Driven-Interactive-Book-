---
sidebar_position: 5
title: "URDF & SDF Simulation"
---

# URDF & SDF Simulation

## Learning Outcomes
By the end of this chapter, you will be able to:
- Understand the structure and components of URDF and SDF files
- Create robot models using URDF for ROS integration
- Design simulation environments using SDF
- Implement sensor and actuator configurations in simulation

## Introduction to Robot Description Formats

Robot description formats are essential for defining robot models, environments, and simulation parameters. The two primary formats used in robotics are URDF (Unified Robot Description Format) for ROS-based systems and SDF (Simulation Description Format) for Gazebo simulation.

### Why Robot Description Formats Matter

Robot description formats provide:
- **Standardization**: Common format for robot model exchange
- **Simulation Integration**: Direct compatibility with simulation environments
- **Visualization**: 3D visualization of robot models
- **Kinematic Analysis**: Forward and inverse kinematics calculations
- **Dynamics Simulation**: Accurate physics modeling

```
Robot Description Workflow
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   URDF/SDF      │───→│  Simulation     │───→│  Visualization  │
│   Model File    │    │  Environment    │    │  & Analysis     │
│                 │    │                 │    │                 │
│ • Links         │    │ • Physics       │    │ • 3D Rendering  │
│ • Joints        │    │ • Collision     │    │ • Kinematics    │
│ • Materials     │    │ • Dynamics      │    │ • Dynamics      │
│ • Sensors       │    │ • Constraints   │    │ • Control       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## URDF (Unified Robot Description Format)

### URDF Structure

URDF is an XML-based format that describes robot models for ROS:

#### Basic Structure
```xml
<?xml version="1.0"?>
<robot name="my_robot">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Joints connect links -->
  <joint name="joint_name" type="revolute">
    <parent link="base_link"/>
    <child link="child_link"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
  </joint>
</robot>
```

### Link Elements

Links represent rigid bodies in the robot:

#### Visual Element
- **Geometry**: Shape definition (box, cylinder, sphere, mesh)
- **Material**: Color and texture properties
- **Origin**: Position and orientation relative to joint

#### Collision Element
- **Geometry**: Collision shape (often simplified from visual)
- **Surface Properties**: Friction, bounce, and contact parameters

#### Inertial Element
- **Mass**: Mass of the link
- **Inertia Tensor**: Moment of inertia properties
- **Origin**: Center of mass location

### Joint Elements

Joints define the connection between links:

#### Joint Types
- **Revolute**: Rotational joint with one degree of freedom
- **Prismatic**: Linear joint with one degree of freedom
- **Fixed**: No movement (welded connection)
- **Continuous**: Rotational joint without limits
- **Planar**: Motion in a plane
- **Floating**: Six degrees of freedom

#### Joint Properties
- **Limits**: Range of motion and physical constraints
- **Dynamics**: Damping and friction coefficients
- **Safety**: Safety controller parameters

### URDF Best Practices

#### Model Organization
- **Hierarchical Structure**: Organize links in kinematic chains
- **Naming Conventions**: Use consistent, descriptive names
- **Coordinate Frames**: Define clear coordinate systems
- **Units**: Use consistent units (SI system)

#### Performance Considerations
- **Simplified Collision Models**: Use simpler shapes for collision
- **Appropriate Mesh Resolution**: Balance detail with performance
- **Realistic Inertial Properties**: Accurate mass and inertia values

## SDF (Simulation Description Format)

### SDF Structure

SDF is the native format for Gazebo simulation:

#### Basic World Structure
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="default">
    <!-- Models in the world -->
    <model name="my_robot">
      <pose>0 0 0 0 0 0</pose>
      <link name="chassis">
        <pose>0 0 0.1 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.5 0.5 0.1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.5 0.5 0.1</size>
            </box>
          </geometry>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.01</ixx>
            <iyy>0.01</iyy>
            <izz>0.01</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- World properties -->
    <physics name="default_physics" default="0" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <gravity>0 0 -9.8</gravity>
    </physics>
  </world>
</sdf>
```

### Model Elements

Models in SDF represent robots, objects, or environments:

#### Model Properties
- **Name**: Unique identifier for the model
- **Pose**: Position and orientation in the world
- **Static**: Whether the model is fixed in place
- **Canonical Link**: Reference link for the model

#### Link Elements
Similar to URDF but with additional simulation-specific properties:
- **Inertial**: Mass properties for physics simulation
- **Visual**: Rendering properties
- **Collision**: Collision detection properties
- **Sensor**: Sensor definitions attached to the link

### SDF Advanced Features

#### Plugins
SDF supports plugins for custom simulation behavior:
```xml
<plugin name="my_controller" filename="libMyController.so">
  <param1>value1</param1>
  <param2>value2</param2>
</plugin>
```

#### Nested Models
- **Model Composition**: Combine multiple models
- **Relative Poses**: Define models relative to each other
- **Inheritance**: Share common properties

## Sensor Integration

### Common Sensor Types in Simulation

#### Camera Sensors
```xml
<sensor name="camera" type="camera">
  <camera>
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10</far>
    </clip>
  </camera>
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
</sensor>
```

#### LIDAR Sensors
```xml
<sensor name="lidar" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples>
        <resolution>1</resolution>
        <min_angle>-1.570796</min_angle>
        <max_angle>1.570796</max_angle>
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>10.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>
</sensor>
```

#### IMU Sensors
```xml
<sensor name="imu" type="imu">
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
    </angular_velocity>
  </imu>
</sensor>
```

## URDF to SDF Conversion

### Automatic Conversion

Gazebo can automatically convert URDF to SDF:
- **URDF Plugin**: Gazebo loads URDF files directly
- **xacro Processing**: Expand macros before conversion
- **Parameter Substitution**: Replace $(arg) and $(env) tags

### Conversion Process
```
URDF File → Robot State Publisher → Gazebo URDF Plugin → SDF → Simulation
```

### Limitations and Considerations
- **Feature Mismatch**: Some URDF features may not map directly to SDF
- **Plugin Integration**: Custom plugins may need SDF-specific configuration
- **Performance**: Conversion overhead in large models

## Simulation Environment Design

### World Creation

#### Environment Elements
- **Terrain**: Ground planes, elevation maps, or custom meshes
- **Obstacles**: Static objects in the environment
- **Lighting**: Sun position, ambient light, and shadows
- **Physics Properties**: Gravity, damping, and solver parameters

#### Example World File
```xml
<sdf version="1.7">
  <world name="my_world">
    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Custom objects -->
    <model name="table">
      <pose>2 0 0 0 0 0</pose>
      <link name="table_surface">
        <visual name="visual">
          <geometry>
            <box><size>1 0.8 0.8</size></box>
          </geometry>
        </visual>
        <collision name="collision">
          <geometry>
            <box><size>1 0.8 0.8</size></box>
          </geometry>
        </collision>
      </link>
    </model>

    <!-- Physics -->
    <physics name="ode" type="ode">
      <gravity>0 0 -9.8</gravity>
    </physics>
  </world>
</sdf>
```

## Advanced Simulation Features

### Multi-robot Simulation

#### Robot Spawning
- **Model Database**: Centralized model repository
- **Unique Names**: Avoid naming conflicts
- **Coordinate Frames**: Proper TF tree management
- **Communication**: Inter-robot communication channels

### Dynamic Environments

#### Moving Objects
- **Joint Actuation**: Control object movement through joints
- **External Forces**: Apply forces and torques programmatically
- **Scripted Behaviors**: Predefined motion patterns
- **Physics Properties**: Dynamic mass and friction changes

### Realistic Sensor Simulation

#### Noise Modeling
- **Gaussian Noise**: Add realistic sensor noise
- **Bias and Drift**: Model sensor imperfections
- **Latency**: Simulate communication delays
- **Dropout**: Model sensor failures and intermittent data

## Tools and Utilities

### Model Validation

#### URDF Validation
- **check_urdf**: Basic syntax and structure validation
- **urdf_to_graphiz**: Generate kinematic tree diagrams
- **Robot Model Viewer**: Visual inspection tools

#### SDF Validation
- **gz sdf**: Validate SDF syntax and structure
- **Schema Validation**: Check against XSD schema
- **Gazebo Integration**: Test in simulation environment

### Visualization Tools

#### RViz
- **Robot Model Display**: Visualize URDF models
- **Sensor Data**: Display sensor streams
- **Path Planning**: Visualize planned trajectories
- **Coordinate Frames**: Show TF tree

#### Gazebo GUI
- **Model Editor**: Create and modify models visually
- **World Editor**: Design simulation environments
- **Real-time Visualization**: Monitor simulation state

## Best Practices

### Model Design

#### Modularity
- **Reusable Components**: Design modular, reusable parts
- **Parameterization**: Use xacro for parameterized models
- **Standard Interfaces**: Follow ROS conventions for joints and sensors

#### Performance Optimization
- **Simplified Collision Models**: Use convex hulls and primitive shapes
- **Appropriate Mesh Resolution**: Balance detail with performance
- **Efficient Joint Configuration**: Minimize unnecessary joints

### Simulation Accuracy

#### Physical Properties
- **Realistic Inertial Values**: Accurate mass and moment of inertia
- **Appropriate Friction**: Realistic friction and damping coefficients
- **Material Properties**: Accurate material characteristics

#### Sensor Modeling
- **Realistic Noise**: Add appropriate sensor noise models
- **Limited Range**: Model actual sensor limitations
- **Update Rates**: Match real sensor update rates

## Learning Summary

URDF and SDF are fundamental to robotics simulation:

- **URDF** provides robot model descriptions for ROS integration
- **SDF** offers native simulation environment descriptions
- **Sensor Integration** enables realistic perception simulation
- **Conversion Tools** bridge the gap between formats
- **Best Practices** ensure accurate and efficient simulation

Understanding these formats is crucial for effective robot simulation and development.

## Exercises

1. Create a simple URDF model of a mobile robot with differential drive, including proper inertial properties and sensor configurations.

2. Design an SDF world file that includes a robot model, obstacles, and lighting, then simulate it in Gazebo.

3. Compare the features and limitations of URDF vs. SDF for robot modeling and simulation, providing specific examples where one format might be preferred over the other.