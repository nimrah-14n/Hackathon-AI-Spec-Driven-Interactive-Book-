import React, { useState, useEffect } from 'react';
import translationService from '../services/translationService';

// Mock translation service - in a real implementation, this would call an actual translation API
const translateText = async (text, targetLang) => {
  return await translationService.translateText(text, targetLang);
};

// Translation hook to manage translation state
export const useTranslation = (initialContent, initialLang = 'en') => {
  const [content, setContent] = useState(initialContent);
  const [currentLang, setCurrentLang] = useState(initialLang);
  const [isTranslating, setIsTranslating] = useState(false);
  const [originalContent, setOriginalContent] = useState(initialContent);

  // Function to toggle between languages
  const toggleLanguage = async () => {
    setIsTranslating(true);
    try {
      const targetLang = currentLang === 'en' ? 'ur' : 'en';
      const translatedContent = await translateText(originalContent, targetLang);
      setContent(translatedContent);
      setCurrentLang(targetLang);
    } catch (error) {
      console.error('Translation error:', error);
    } finally {
      setIsTranslating(false);
    }
  };

  // Function to translate to specific language
  const translateTo = async (lang) => {
    if (lang === currentLang) return;

    setIsTranslating(true);
    try {
      const translatedContent = await translateText(originalContent, lang);
      setContent(translatedContent);
      setCurrentLang(lang);
    } catch (error) {
      console.error('Translation error:', error);
    } finally {
      setIsTranslating(false);
    }
  };

  // Function to reset to original content
  const resetToOriginal = () => {
    setContent(originalContent);
    setCurrentLang('en');
  };

  return {
    content,
    currentLang,
    isTranslating,
    toggleLanguage,
    translateTo,
    resetToOriginal,
    setOriginalContent: (newContent) => {
      setOriginalContent(newContent);
      if (currentLang === 'en') {
        setContent(newContent);
      }
    }
  };
};

export default useTranslation;