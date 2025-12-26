import React, { useState, useRef, useEffect } from 'react';
import './Chatbot.css';

const Chatbot = ({ isOpen, onClose, onPageTextSelection }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [contextMode, setContextMode] = useState('book_wide'); // 'book_wide' or 'selected_text'
  const [selectedText, setSelectedText] = useState('');
  const [languagePreference, setLanguagePreference] = useState('en'); // 'en' for English, 'ur' for Urdu
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Function to determine response badge based on content
  const getResponseBadge = (text) => {
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
  };

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = { id: Date.now(), text: inputValue, sender: 'user', timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Get currently selected text from the page
      const currentSelectedText = window.getSelection().toString().trim();

      // Use the current selection if in selected_text mode, or fallback to stored selected text
      const textToSend = contextMode === 'selected_text'
        ? (currentSelectedText || selectedText)
        : null;

      const response = await fetch('/api/v1/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputValue,
          context_mode: contextMode,
          selected_text: contextMode === 'selected_text' ? textToSend : null,
          language_preference: languagePreference,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const botMessage = {
        id: Date.now() + 1,
        text: data.response,
        sender: 'bot',
        sources: data.sources || [],
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        id: Date.now() + 1,
        text: 'Sorry, I encountered an error processing your request. Please try again.',
        sender: 'bot',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const toggleContextMode = () => {
    setContextMode(prev => prev === 'book_wide' ? 'selected_text' : 'book_wide');
  };

  // Get selected text when it changes
  useEffect(() => {
    const handleSelectionChange = () => {
      const selection = window.getSelection();
      const selectedText = selection.toString().trim();

      if (selectedText) {
        setSelectedText(selectedText);
        // Optionally notify parent component about selected text
        if (onPageTextSelection) {
          onPageTextSelection(selectedText);
        }
      }
    };

    document.addEventListener('selectionchange', handleSelectionChange);
    return () => {
      document.removeEventListener('selectionchange', handleSelectionChange);
    };
  }, [onPageTextSelection]);

  if (!isOpen) return null;

  return (
    <div className="chatbot-container">
      <div className="chatbot-header">
        <div className="chatbot-title">AI Learning Assistant</div>
        <div className="chatbot-controls">
          <button
            className={`context-mode-btn ${contextMode === 'selected_text' ? 'active' : ''}`}
            onClick={toggleContextMode}
            title={contextMode === 'book_wide'
              ? 'Switch to selected text mode'
              : 'Switch to book-wide context mode'}
          >
            {contextMode === 'book_wide' ? '📖 Book Context' : '📝 Selected Text'}
          </button>
          <button
            className={`language-toggle-btn ${languagePreference === 'ur' ? 'active' : ''}`}
            onClick={() => setLanguagePreference(prev => prev === 'en' ? 'ur' : 'en')}
            title={languagePreference === 'en' ? 'Switch to Urdu' : 'Switch to English'}
          >
            {languagePreference === 'en' ? '🌐 EN' : '🌐 UR'}
          </button>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>
      </div>

      <div className="chatbot-messages">
        {messages.length === 0 ? (
          <div className="welcome-message">
            <p>Hello! I'm your AI Learning Assistant.</p>
            <p>Ask me questions about the book content, or switch to "Selected Text" mode to ask about specific text you've highlighted.</p>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}
            >
              <div className="message-content">
                {message.sender === 'bot' && (
                  <span className="response-badge">
                    {getResponseBadge(message.text)}
                  </span>
                )}
                {message.text}
                {message.sources && message.sources.length > 0 && (
                  <div className="message-sources">
                    Sources: {message.sources.slice(0, 3).join(', ')}
                  </div>
                )}
              </div>
              <div className="message-timestamp">
                {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="message bot-message">
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chatbot-input-area">
        <textarea
          ref={inputRef}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about the book content..."
          className="chatbot-input"
          rows="2"
        />
        <button
          onClick={handleSend}
          disabled={!inputValue.trim() || isLoading}
          className="send-button"
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </div>
    </div>
  );
};

export default Chatbot;