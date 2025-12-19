---
sidebar_position: 4
title: "Nodes, Topics, Services & Actions"
---

# Nodes, Topics, Services & Actions

## Learning Outcomes
By the end of this chapter, the learner will:
- Understand the fundamental communication patterns in ROS 2
- Implement nodes with different communication paradigms
- Distinguish between topics, services, and actions
- Apply appropriate communication patterns for different robotic scenarios

## Introduction

Nodes, topics, services, and actions form the fundamental building blocks of ROS 2's communication architecture, representing the core abstractions that enable distributed robotic systems to coordinate and collaborate effectively. These communication patterns mirror the functional components of a biological nervous system: nodes act as processing centers, topics serve as sensory and motor pathways, services function as reflex arcs for immediate responses, and actions coordinate complex, goal-oriented behaviors.

The design of these communication patterns reflects the need for different types of interactions within robotic systems. Topics provide asynchronous, one-way communication ideal for continuous data streams such as sensor readings or robot state information. Services enable synchronous request-response interactions suitable for immediate computations or state queries. Actions coordinate long-running, goal-oriented behaviors that may require feedback and the ability to cancel ongoing operations.

Understanding these communication patterns is essential for designing robust robotic systems that can scale from simple single-robot applications to complex multi-robot systems. Each pattern addresses specific requirements for timing, reliability, and interaction complexity that arise in real-world robotic applications. The choice of communication pattern significantly impacts system performance, reliability, and maintainability.

The biological nervous system metaphor extends to these patterns: topics resemble the constant flow of sensory information and motor commands, services correspond to reflexive responses that require immediate action, and actions represent higher-level behaviors that involve planning, execution, and monitoring over extended periods.

## Core Concepts

- **Nodes**: Autonomous processing units that encapsulate robot functionality
- **Topics**: Asynchronous publish-subscribe communication for streaming data
- **Services**: Synchronous request-response communication for immediate interactions
- **Actions**: Asynchronous goal-oriented communication with feedback and cancellation
- **Parameters**: Configuration values that can be dynamically adjusted
- **Interfaces**: Type definitions that specify message formats and structures

Each communication pattern serves specific purposes:

- **Topics** are ideal for continuous data streams like camera feeds, laser scans, or robot joint states
- **Services** work best for immediate queries like inverse kinematics calculations or state requests
- **Actions** handle complex, long-running tasks like navigation goals or manipulation sequences
- **Parameters** manage configuration that may need runtime adjustment

## Practical Relevance

The communication patterns in ROS 2 enable the development of modular, maintainable robotic systems that can adapt to changing requirements. In real-world applications, these patterns allow:

- **Sensor Integration**: Publishing sensor data through topics for multiple subscribers to process
- **Control Systems**: Using services for immediate actuator commands or state queries
- **Mission Planning**: Employing actions for complex, multi-step tasks that require monitoring
- **Multi-robot Coordination**: Sharing information between robots through topics and services
- **Human-Robot Interaction**: Providing interfaces for operators to monitor and control robot behavior

ROS 2's communication patterns support the development of reusable robot components that can be combined in various configurations to create different robot systems. This modularity reduces development time and increases system reliability by enabling thorough testing of individual components before integration.

The patterns also facilitate the development of robust error handling and recovery mechanisms. Topics can include quality of service settings that ensure critical data reaches its destinations, services can implement timeouts and retry mechanisms, and actions provide built-in feedback channels for monitoring progress and detecting failures.

In industrial applications, these communication patterns enable the creation of robot systems that can adapt to changing production requirements, integrate with existing factory systems, and operate safely alongside human workers. The patterns provide the flexibility needed to handle the variability and unpredictability of real-world manufacturing environments.

## Communication Patterns

### Topics (Publish-Subscribe)

Topics implement the publish-subscribe pattern, where publishers send messages to named topics without knowing who will receive them, and subscribers receive messages from topics without knowing who published them. This loose coupling enables flexible system architectures where components can be added, removed, or replaced without affecting other system components.

Topics are ideal for continuous data streams where the latest value is typically more important than historical values. Examples include sensor data, robot state information, and visualization messages. Quality of service settings allow fine-tuning of reliability, durability, and ordering characteristics to meet specific application requirements.

### Services (Request-Response)

Services implement synchronous request-response communication where a client sends a request and waits for a response. This pattern is appropriate for operations that have a clear beginning and end, can complete relatively quickly, and require a definitive result. Service calls block until completion, making them unsuitable for long-running operations.

Services work well for computations that can be completed quickly, such as inverse kinematics calculations, map queries, or state information requests. The synchronous nature ensures that clients receive responses before continuing, providing a clear execution flow.

### Actions (Goal-Focused)

Actions coordinate long-running, goal-oriented behaviors that may take seconds, minutes, or hours to complete. Unlike services, actions are asynchronous and provide continuous feedback about progress. They also support cancellation, allowing ongoing operations to be stopped gracefully.

Actions are ideal for complex robot behaviors like navigation to a goal location, manipulation of objects, or inspection tasks. The feedback mechanism allows clients to monitor progress and make informed decisions about whether to continue or cancel ongoing operations.

## Architecture Considerations

When designing robot systems using these communication patterns, several architectural considerations apply:

- **Timing Requirements**: Choose patterns that match the timing constraints of the application
- **Reliability Needs**: Apply appropriate quality of service settings for critical communications
- **Scalability**: Design systems that can accommodate additional nodes and increased communication load
- **Debugging and Monitoring**: Structure communications to facilitate system observation and troubleshooting
- **Security**: Consider authentication and encryption requirements for sensitive communications

The selection of communication patterns significantly impacts system performance and reliability. Proper use of these patterns results in robot systems that are responsive, robust, and maintainable. Understanding when to use each pattern is crucial for developing effective robotic applications.