import React from 'react';
import { useChatbot } from './ChatbotProvider';
import './Chatbot.css';

const ChatbotToggle = () => {
  const { toggleChatbot, isChatbotOpen } = useChatbot();

  if (isChatbotOpen) return null;

  return (
    <button className="chatbot-toggle" onClick={toggleChatbot} aria-label="Open chatbot">
      <span role="img" aria-label="robot">🤖</span>
    </button>
  );
};

export default ChatbotToggle;