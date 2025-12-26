import React, { createContext, useContext, useState } from 'react';
import Chatbot from './Chatbot';

const ChatbotContext = createContext();

export const useChatbot = () => {
  const context = useContext(ChatbotContext);
  if (!context) {
    throw new Error('useChatbot must be used within a ChatbotProvider');
  }
  return context;
};

export const ChatbotProvider = ({ children }) => {
  const [isChatbotOpen, setIsChatbotOpen] = useState(false);
  const [selectedText, setSelectedText] = useState('');

  const openChatbot = () => setIsChatbotOpen(true);
  const closeChatbot = () => setIsChatbotOpen(false);
  const toggleChatbot = () => setIsChatbotOpen(!isChatbotOpen);

  const handlePageTextSelection = (text) => {
    setSelectedText(text);
  };

  return (
    <ChatbotContext.Provider
      value={{
        isChatbotOpen,
        openChatbot,
        closeChatbot,
        toggleChatbot,
        selectedText,
        setSelectedText: handlePageTextSelection,
      }}
    >
      {children}
      <Chatbot
        isOpen={isChatbotOpen}
        onClose={closeChatbot}
        onPageTextSelection={handlePageTextSelection}
      />
    </ChatbotContext.Provider>
  );
};