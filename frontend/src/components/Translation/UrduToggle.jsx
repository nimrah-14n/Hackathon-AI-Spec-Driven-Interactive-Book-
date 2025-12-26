import React, { useState, useEffect } from 'react';
import './Translation.css';

const UrduToggle = ({ onLanguageChange }) => {
  const [isUrdu, setIsUrdu] = useState(false);

  const toggleLanguage = () => {
    const newIsUrdu = !isUrdu;
    setIsUrdu(newIsUrdu);

    // Save preference to localStorage
    localStorage.setItem('preferredLanguage', newIsUrdu ? 'ur' : 'en');

    // Notify parent component of language change
    if (onLanguageChange) {
      onLanguageChange(newIsUrdu ? 'ur' : 'en');
    }
  };

  // Load saved preference on component mount
  useEffect(() => {
    const savedPreference = localStorage.getItem('preferredLanguage');
    if (savedPreference) {
      setIsUrdu(savedPreference === 'ur');
    }
  }, []);

  return (
    <div className="urdu-toggle-container">
      <button
        className={`urdu-toggle-btn ${isUrdu ? 'urdu-active' : 'english-active'}`}
        onClick={toggleLanguage}
        title={isUrdu ? 'Switch to English' : 'Switch to Urdu'}
      >
        <span className="urdu-toggle-icon">🌐</span>
        <span className="urdu-toggle-text">{isUrdu ? 'اردو' : 'EN'}</span>
      </button>
    </div>
  );
};

export default UrduToggle;