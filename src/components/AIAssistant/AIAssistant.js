import React, { useState } from 'react';
import styles from './AIAssistant.module.css';

const AIAssistant = ({ chapterTitle }) => {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Simple knowledge base for the three chapters we created
  const knowledgeBase = {
    "Physical AI & Embodied Intelligence": [
      {
        question: "What is Physical AI?",
        answer: "Physical AI represents a paradigm shift from traditional digital AI systems that operate in virtual spaces to AI systems that interact with the physical world. Unlike classical AI that processes abstract data, Physical AI systems must navigate the complexities of real-world physics, uncertainty, and dynamic environments."
      },
      {
        question: "What is Embodied Intelligence?",
        answer: "Embodied Intelligence is the concept that intelligence emerges from the interaction between an agent and its environment. This challenges the traditional view of intelligence as purely computational, suggesting instead that physical constraints and affordances shape intelligent behavior."
      },
      {
        question: "What are the key challenges in Physical AI?",
        answer: "Key challenges include dealing with physical constraints (gravity, energy, materials), handling uncertainty and noise in sensor data, meeting real-time requirements, and ensuring safety in physical interactions."
      }
    ],
    "From Digital AI to Robots in the Physical World": [
      {
        question: "What's the difference between digital AI and physical AI?",
        answer: "Digital AI operates in virtual environments with perfect information and no physical constraints, while physical AI must deal with real-world physics, uncertainty, real-time requirements, and safety concerns."
      },
      {
        question: "What is the reality gap?",
        answer: "The reality gap refers to the performance difference between simulated and real-world systems. AI trained in simulation often fails when deployed in reality because perfect models don't capture all real-world complexities."
      },
      {
        question: "What strategies help bridge the reality gap?",
        answer: "Strategies include domain randomization (training in varied simulated environments), sim-to-real transfer techniques, and robust design principles that account for uncertainty and real-world conditions."
      }
    ],
    "ROS 2 Overview and Architecture": [
      {
        question: "What is ROS 2?",
        answer: "ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It provides libraries, tools, and conventions that simplify the development of complex robotic applications by handling communication between robot components."
      },
      {
        question: "What improvements does ROS 2 have over ROS 1?",
        answer: "ROS 2 offers production-ready features, real-time support, built-in security, configurable Quality of Service, and better cross-platform compatibility compared to ROS 1."
      },
      {
        question: "What is DDS in ROS 2?",
        answer: "DDS (Data Distribution Service) is the communication middleware used by ROS 2. It provides decoupled communication, automatic discovery, Quality of Service configuration, and language independence."
      }
    ]
  };

  const handleAsk = async () => {
    if (!question.trim()) return;

    setIsLoading(true);

    // Simple matching logic - in a real implementation, this would call an AI API
    const currentChapterKB = knowledgeBase[chapterTitle] || [];
    let foundAnswer = "I don't have specific information about that in this chapter. Please refer to the chapter content for more details.";

    // Look for similar questions in the knowledge base
    for (const qa of currentChapterKB) {
      if (question.toLowerCase().includes(qa.question.toLowerCase()) ||
          qa.question.toLowerCase().includes(question.toLowerCase())) {
        foundAnswer = qa.answer;
        break;
      }
    }

    // Simulate API delay
    setTimeout(() => {
      setAnswer(foundAnswer);
      setIsLoading(false);
    }, 1000);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleAsk();
    }
  };

  return (
    <div className={styles.aiAssistant}>
      <h3>AI Assistant for {chapterTitle}</h3>
      <div className={styles.chatContainer}>
        <div className={styles.questionInput}>
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask a question about this chapter..."
            className={styles.questionBox}
          />
          <button
            onClick={handleAsk}
            disabled={isLoading}
            className={styles.askButton}
          >
            {isLoading ? 'Thinking...' : 'Ask'}
          </button>
        </div>

        {answer && (
          <div className={styles.answerContainer}>
            <div className={styles.answer}>
              <strong>AI Response:</strong> {answer}
            </div>
          </div>
        )}

        <div className={styles.knowledgeHint}>
          <p><strong>Tip:</strong> Ask questions about concepts from this chapter. The AI assistant has knowledge about:</p>
          <ul>
            {knowledgeBase[chapterTitle]?.map((qa, index) => (
              <li key={index}>{qa.question}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default AIAssistant;