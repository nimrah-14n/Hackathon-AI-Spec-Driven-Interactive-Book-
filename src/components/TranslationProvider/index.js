import React, { createContext, useContext, useState } from 'react';

const TranslationContext = createContext();

export const useTranslationContext = () => {
  const context = useContext(TranslationContext);
  if (!context) {
    throw new Error('useTranslationContext must be used within a TranslationProvider');
  }
  return context;
};

export const TranslationProvider = ({ children }) => {
  const [currentLang, setCurrentLang] = useState('en');
  const [isTranslating, setIsTranslating] = useState(false);

  const toggleLanguage = () => {
    setCurrentLang(prev => prev === 'en' ? 'ur' : 'en');
  };

  const setLanguage = (lang) => {
    if (lang === 'en' || lang === 'ur') {
      setCurrentLang(lang);
    }
  };

  const value = {
    currentLang,
    isTranslating,
    setIsTranslating,
    toggleLanguage,
    setLanguage
  };

  return (
    <TranslationContext.Provider value={value}>
      {children}
    </TranslationContext.Provider>
  );
};

export default TranslationProvider;