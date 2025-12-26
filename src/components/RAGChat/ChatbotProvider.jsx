import React, { createContext, useContext, useState, useEffect } from 'react';

const ChatbotContext = createContext();

export const useChatbot = () => {
  const context = useContext(ChatbotContext);
  if (!context) {
    throw new Error('useChatbot must be used within a ChatbotProvider');
  }
  return context;
};

export const ChatbotProvider = ({ children }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [languagePreference, setLanguagePreference] = useState('en'); // 'en' for English, 'ur' for Urdu

  // Function to send message to backend
  const sendMessage = async (message, contextMode = 'book_wide', selectedText = null) => {
    setIsLoading(true);
    try {
      // Get user ID from localStorage if authenticated
      const storedUser = localStorage.getItem('user');
      const userId = storedUser ? JSON.parse(storedUser).id : null;

      // Add user message to chat
      const userMessage = { id: Date.now(), text: message, sender: 'user', timestamp: new Date() };
      setMessages(prev => [...prev, userMessage]);

      // Call backend API
      const response = await fetch('http://localhost:8000/api/v1/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          context_mode: contextMode,
          selected_text: selectedText,
          language_preference: languagePreference,
          user_id: userId
        }),
      });

      if (response.ok) {
        const data = await response.json();
        const botMessage = {
          id: Date.now() + 1,
          text: data.response,
          sender: 'bot',
          sources: data.sources || [],
          timestamp: new Date()
        };
        setMessages(prev => [...prev, botMessage]);

        // Update chat history
        setChatHistory(prev => [...prev, { user: message, bot: data.response, timestamp: new Date() }]);
      } else {
        const errorMessage = {
          id: Date.now() + 1,
          text: 'Sorry, I encountered an error. Please try again.',
          sender: 'bot',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        text: 'Connection error. Please check if the backend is running.',
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  const clearChat = () => {
    setMessages([]);
  };

  const toggleLanguage = () => {
    setLanguagePreference(prev => prev === 'en' ? 'ur' : 'en');
  };

  const setLanguage = (lang) => {
    setLanguagePreference(lang);
  };

  const value = {
    isOpen,
    setIsOpen,
    messages,
    isLoading,
    chatHistory,
    sendMessage,
    toggleChat,
    clearChat,
    languagePreference,
    toggleLanguage,
    setLanguage,
  };

  return (
    <ChatbotContext.Provider value={value}>
      {children}
    </ChatbotContext.Provider>
  );
};