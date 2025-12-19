---
sidebar_position: 3
title: "ROS 2 Overview and Architecture"
---

# ROS 2 Overview and Architecture

## Learning Outcomes
By the end of this chapter, you will be able to:
- Explain the purpose and architecture of ROS 2
- Identify key components of the ROS 2 ecosystem
- Understand the differences between ROS 1 and ROS 2
- Describe how ROS 2 enables robotic applications development

## Introduction to ROS 2

Robot Operating System 2 (ROS 2) is not an operating system but a flexible framework for writing robot software. It provides libraries, tools, and conventions that simplify the development of complex robotic applications by handling communication between robot components.

### What is ROS 2?

ROS 2 is the next generation of the Robot Operating System, designed to address the limitations of ROS 1 and provide:

- **Production-Ready**: Built for commercial and industrial applications
- **Real-Time Support**: Deterministic behavior for time-critical applications
- **Security**: Built-in security features for safe robot operation
- **Quality of Service**: Configurable communication guarantees
- **Cross-Platform**: Runs on various operating systems and architectures

### Why ROS 2 Matters for Physical AI

ROS 2 serves as the "nervous system" for humanoid robots by:

- **Enabling Modularity**: Different components can be developed independently
- **Facilitating Communication**: Standardized message passing between components
- **Providing Reusability**: Common tools and libraries for different robots
- **Supporting Distribution**: Components can run on different machines
- **Ensuring Real-Time**: Critical for robot control and safety

## ROS 2 Architecture

### DDS-Based Communication

ROS 2 uses Data Distribution Service (DDS) as its communication middleware:

```
┌─────────────────┐    DDS Communication    ┌─────────────────┐
│   Node A        │ ←─────────────────────→ │   Node B        │
│                 │                         │                 │
│ ┌─────────────┐ │                         │ ┌─────────────┐ │
│ │ Publisher   │ │                         │ │ Subscriber  │ │
│ │ /cmd_vel    │ │ ←─── Topic: /cmd_vel ───→ │ │ /cmd_vel    │ │
│ └─────────────┘ │                         │ └─────────────┘ │
└─────────────────┘                         └─────────────────┘
```

DDS provides:
- **Decoupled Communication**: Publishers and subscribers don't need direct connections
- **Discovery**: Automatic discovery of nodes and topics
- **Quality of Service**: Configurable reliability and performance settings
- **Language Independence**: Support for multiple programming languages

### Core Architecture Components

#### 1. Nodes
Nodes are the basic execution units in ROS 2:

- **Definition**: Processes that perform computation
- **Purpose**: Encapsulate robot functionality
- **Communication**: Through publishers, subscribers, services, and actions
- **Implementation**: Can be written in C++, Python, or other supported languages

```python
import rclpy
from rclpy.node import Node

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        # Node initialization code
```

#### 2. Topics and Messages
Topics enable asynchronous communication:

- **Publisher**: Sends data to a topic
- **Subscriber**: Receives data from a topic
- **Message Types**: Defined in .msg files with specific data structures
- **Communication Pattern**: Many-to-many, asynchronous

#### 3. Services
Services enable synchronous request-response communication:

- **Client**: Sends a request and waits for a response
- **Server**: Receives requests and sends responses
- **Communication Pattern**: One-to-one, synchronous
- **Use Case**: Actions that require confirmation or return specific results

#### 4. Actions
Actions enable goal-oriented communication with feedback:

- **Goal**: Request to perform a long-running task
- **Feedback**: Updates on task progress
- **Result**: Final outcome of the task
- **Use Case**: Navigation, manipulation, calibration

## ROS 2 vs ROS 1

| Feature | ROS 1 | ROS 2 |
|---------|-------|-------|
| **Communication** | Custom (roscpp/rosjava) | DDS-based |
| **Real-Time** | Limited | Full support |
| **Security** | None | Built-in security |
| **Quality of Service** | Fixed | Configurable |
| **Cross-Platform** | Linux-focused | Multi-platform |
| **Deployment** | Single machine | Distributed |
| **Maintenance** | End-of-life | Actively maintained |

### Key Improvements in ROS 2

#### 1. Production-Ready Architecture
- **DDS Middleware**: Industry-standard communication
- **Quality of Service**: Configurable reliability and performance
- **Security**: Authentication, encryption, and access control

#### 2. Real-Time Support
- **Deterministic Communication**: Predictable timing
- **Real-Time Scheduling**: Integration with RTOS features
- **Low Latency**: Optimized for time-critical applications

#### 3. Cross-Platform Compatibility
- **Multiple OS**: Linux, Windows, macOS, embedded systems
- **Multiple Architectures**: x86, ARM, and others
- **Container Support**: Docker and other containerization

## ROS 2 Ecosystem

### Core Packages

#### 1. rclpy and rclcpp
- **rclpy**: Python client library for ROS 2
- **rclcpp**: C++ client library for ROS 2
- **Purpose**: Provide interfaces to ROS 2 functionality

#### 2. rmw (ROS Middleware Interface)
- **Abstraction Layer**: Interface between ROS 2 and DDS implementations
- **Flexibility**: Switch between different DDS vendors
- **Implementation**: rmw_fastrtps, rmw_cyclonedds, rmw_connextdds

#### 3. rcl (ROS Client Library)
- **Core Library**: Implements ROS concepts like nodes, publishers, subscribers
- **Language Agnostic**: Foundation for client libraries in different languages

### Development Tools

#### 1. ros2 CLI
Command-line interface for ROS 2 operations:

```bash
# Run a node
ros2 run package_name executable_name

# List topics
ros2 topic list

# Echo a topic
ros2 topic echo /topic_name

# List services
ros2 service list
```

#### 2. rviz2
3D visualization tool for robot data:

- **Purpose**: Visualize robot state, sensors, and environment
- **Features**: Point clouds, robot models, paths, markers
- **Extensibility**: Custom plugins for specific data types

#### 3. ros2doctor
Diagnostic tool for ROS 2 systems:

- **Purpose**: Check system configuration and connectivity
- **Features**: Network diagnostics, configuration validation
- **Use Case**: Troubleshooting communication issues

## ROS 2 Communication Patterns

### 1. Publisher-Subscriber (Topics)
Asynchronous, many-to-many communication:

```python
# Publisher example
publisher = node.create_publisher(String, 'chatter', 10)
msg = String()
msg.data = 'Hello World'
publisher.publish(msg)

# Subscriber example
def chatter_callback(msg):
    node.get_logger().info('I heard: %s' % msg.data)

subscription = node.create_subscription(
    String, 'chatter', chatter_callback, 10)
```

### 2. Client-Server (Services)
Synchronous, request-response communication:

```python
# Service server
from example_interfaces.srv import AddTwoInts

def add_two_ints_callback(request, response):
    response.sum = request.a + request.b
    return response

service = node.create_service(AddTwoInts, 'add_two_ints', add_two_ints_callback)

# Service client
client = node.create_client(AddTwoInts, 'add_two_ints')
```

### 3. Goal-Feedback-Result (Actions)
Goal-oriented communication with progress tracking:

```python
# Action client
from example_interfaces.action import Fibonacci

action_client = ActionClient(node, Fibonacci, 'fibonacci')

# Send goal with feedback callback
send_goal_future = action_client.send_goal_async(
    goal_msg,
    feedback_callback=feedback_callback)
```

## Quality of Service (QoS) Settings

ROS 2 provides configurable communication behavior:

### Reliability
- **Reliable**: All messages are delivered (like TCP)
- **Best Effort**: Messages may be lost (like UDP)

### Durability
- **Transient Local**: Late-joining subscribers receive last message
- **Volatile**: Only new messages are sent

### History
- **Keep Last**: Store last N messages
- **Keep All**: Store all messages

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE
)
```

## ROS 2 for Humanoid Robotics

### Modular Architecture Benefits
ROS 2 enables modular development of humanoid robots:

- **Sensor Processing**: Separate nodes for vision, IMU, force/torque
- **Control Systems**: Joint controllers, balance controllers
- **Perception**: Object detection, SLAM, person tracking
- **Planning**: Path planning, motion planning, task planning
- **Behavior**: State machines, decision making

### Example Humanoid Robot Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Perception    │    │   Planning      │    │   Control       │
│   Nodes         │───→│   Nodes         │───→│   Nodes         │
│                 │    │                 │    │                 │
│ • Vision        │    │ • Path Planning │    │ • Joint Ctrl    │
│ • SLAM          │    │ • Motion Plan   │    │ • Balance Ctrl  │
│ • Object Detect │    │ • Task Plan     │    │ • Trajectory    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │   Communication │
                    │   Infrastructure│
                    │   (DDS/QoS)     │
                    └─────────────────┘
```

## Learning Summary

ROS 2 serves as the foundational framework for developing humanoid robots and other complex robotic systems:

- **Architecture**: DDS-based communication with nodes, topics, services, and actions
- **Improvements**: Production-ready, real-time support, security, and cross-platform compatibility
- **Communication**: Multiple patterns for different use cases and requirements
- **Ecosystem**: Rich set of tools, libraries, and packages for robot development

Understanding ROS 2 architecture is crucial for building the "nervous system" of humanoid robots, enabling different components to communicate effectively while maintaining modularity and scalability.

## Exercises

1. Create a simple ROS 2 publisher and subscriber in Python that communicate a custom message.
2. Research and compare three different DDS implementations used with ROS 2 (FastDDS, CycloneDDS, RTI Connext).
3. Design a ROS 2 node architecture for a simple humanoid robot with walking, vision, and manipulation capabilities.

import AIAssistant from '@site/src/components/AIAssistant/AIAssistant';

<AIAssistant chapterTitle="ROS 2 Overview and Architecture" />