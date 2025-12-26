import React, { useState, useEffect } from 'react';
import { useChatbot } from './ChatbotProvider';
import './ChatbotToggle.css';

const ChatbotToggle = () => {
  const { isOpen, toggleChat, messages, isLoading, sendMessage } = useChatbot();
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    const chatContainer = document.getElementById('chat-messages-container');
    if (chatContainer) {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
  }, [messages]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (inputValue.trim() && !isLoading) {
      await sendMessage(inputValue);
      setInputValue('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend(e);
    }
  };

  if (!isOpen) {
    return (
      <div className="chatbot-toggle-button" onClick={toggleChat}>
        <div className="chatbot-icon">🤖</div>
      </div>
    );
  }

  return (
    <div className="chatbot-container">
      <div className="chatbot-header">
        <div className="chatbot-title">AI Assistant</div>
        <button className="chatbot-close" onClick={toggleChat}>
          ×
        </button>
      </div>

      <div id="chat-messages-container" className="chatbot-messages">
        {messages.length === 0 ? (
          <div className="chatbot-welcome">
            <p>Hello! I'm your AI assistant for the AI & Robotics Learning Platform.</p>
            <p>Ask me anything about the book content!</p>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`chatbot-message ${message.sender}-message`}
            >
              <div className="message-content">
                {message.sender === 'bot' && <span className="bot-icon">🤖</span>}
                <div className="message-text">
                  {message.text}
                  {message.sources && message.sources.length > 0 && (
                    <div className="message-sources">
                      <small>Sources: {message.sources.join(', ')}</small>
                    </div>
                  )}
                </div>
                {message.sender === 'user' && <span className="user-icon">👤</span>}
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="chatbot-message bot-message">
            <div className="message-content">
              <span className="bot-icon">🤖</span>
              <div className="message-text typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
      </div>

      <form className="chatbot-input-form" onSubmit={handleSend}>
        <textarea
          className="chatbot-input"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about the book content..."
          rows="1"
          disabled={isLoading}
        />
        <button
          type="submit"
          className="chatbot-send-button"
          disabled={isLoading || !inputValue.trim()}
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  );
};

export default ChatbotToggle;