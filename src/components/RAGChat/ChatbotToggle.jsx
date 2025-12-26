import React, { useState, useEffect } from 'react';
import { useChatbot } from './ChatbotProvider';
import './ChatbotToggle.css';

const ChatbotToggle = () => {
  const { isOpen, toggleChat, messages, isLoading, sendMessage, languagePreference, toggleLanguage } = useChatbot();
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
        <div className="chatbot-controls">
          <button
            className={`language-toggle-btn ${languagePreference === 'ur' ? 'urdu-active' : 'english-active'}`}
            onClick={toggleLanguage}
            title={languagePreference === 'en' ? 'Switch to Urdu' : 'Switch to English'}
          >
            <span className="urdu-toggle-icon">🌐</span>
            <span className="urdu-toggle-text">{languagePreference === 'en' ? 'EN' : 'UR'}</span>
          </button>
          <button
            className="profile-btn"
            onClick={() => {
              // Check if user is authenticated
              const token = localStorage.getItem('authToken');
              if (token) {
                // If authenticated, show user info tooltip or profile options
                const user = JSON.parse(localStorage.getItem('user') || '{}');
                alert(`Signed in as: ${user.name || 'User'}\nEmail: ${user.email || ''}\n\nClick OK to continue chatting.`);
              } else {
                // If not authenticated, show a login prompt
                if (confirm('Sign in to personalize your experience and save your chat history. Would you like to sign in now?')) {
                  // In a real app, we would open an auth modal here
                  console.log('Opening auth modal...');
                  // You could trigger a login modal here
                }
              }
            }}
            title={localStorage.getItem('authToken') ? 'View Profile' : 'Sign In'}
          >
            <span className="profile-icon">{localStorage.getItem('authToken') ? '👤' : '🔒'}</span>
          </button>
          <button className="chatbot-close" onClick={toggleChat}>
            ×
          </button>
        </div>
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
                {message.sender === 'bot' && (
                  <>
                    <span className="response-badge">
                      {(() => {
                        const text = message.text;
                        // Check if the response mentions book content
                        if (text.toLowerCase().includes('book') ||
                            text.toLowerCase().includes('content') ||
                            text.toLowerCase().includes('chapter') ||
                            text.toLowerCase().includes('page') ||
                            text.toLowerCase().includes('document') ||
                            text.toLowerCase().includes('provided context')) {
                          return '📘'; // Book Answer
                        }
                        // Check if the response is general AI/Robotics knowledge
                        else if (text.toLowerCase().includes('ai') ||
                                 text.toLowerCase().includes('robotics') ||
                                 text.toLowerCase().includes('machine learning') ||
                                 text.toLowerCase().includes('ml') ||
                                 text.toLowerCase().includes('neural network') ||
                                 text.toLowerCase().includes('algorithm') ||
                                 text.toLowerCase().includes('i don\'t have that information in the book') ||
                                 text.toLowerCase().includes('couldn\'t find this information in the book')) {
                          return '🤖'; // AI Knowledge
                        }
                        // If current language is Urdu, show Urdu badge
                        else if (languagePreference === 'ur') {
                          return '🌐'; // Urdu
                        }
                        // Default to AI knowledge
                        else {
                          return '🤖'; // AI Knowledge
                        }
                      })()}
                    </span>
                    <span className="bot-icon">🤖</span>
                  </>
                )}
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