// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module-1/chapter-1',
        'module-1/chapter-2',
        'module-1/chapter-3',
        'module1-ros2/nodes-topics-services-actions',
        'module1-ros2/ros2-data-flow',
        'module1-ros2/python-rclpy-development',
        'module1-ros2/bridging-ai-controllers',
        'module1-ros2/urdf-humanoid-modeling',
        'module1-ros2/launch-files-configuration',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module2-digital-twin/digital-twin-physical-ai',
        'module2-digital-twin/simulation-real-world',
        'module2-digital-twin/gazebo-architecture',
        'module2-digital-twin/physics-simulation',
        'module2-digital-twin/urdf-sdf-simulation',
        'module2-digital-twin/sensor-simulation',
        'module2-digital-twin/unity-rendering',
        'module2-digital-twin/human-robot-interaction',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI–Robot Brain (NVIDIA Isaac)',
      items: [
        'module3-ai-brain/ai-brain-humanoid',
        'module3-ai-brain/nvidia-isaac-overview',
        'module3-ai-brain/isaac-sim-photorealistic',
        'module3-ai-brain/synthetic-data-generation',
        'module3-ai-brain/isaac-ros-architecture',
        'module3-ai-brain/vslam-fundamentals',
        'module3-ai-brain/navigation-nav2',
        'module3-ai-brain/reinforcement-learning',
        'module3-ai-brain/sim-to-real-transfer',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module4-vla/vla-paradigm',
        'module4-vla/conversational-robotics',
        'module4-vla/voice-to-action',
        'module4-vla/natural-language-understanding',
        'module4-vla/cognitive-planning-llm',
        'module4-vla/language-to-ros2-sequences',
        'module4-vla/multi-modal-interaction',
        'module4-vla/capstone-autonomous-humanoid',
      ],
    },
  ],
};

module.exports = sidebars;