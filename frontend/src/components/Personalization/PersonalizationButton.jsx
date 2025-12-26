import React, { useState, useEffect } from 'react';
import AuthModal from './AuthModal';
import './PersonalizationButton.css';

const PersonalizationButton = ({ chapterId, userId = null }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [user, setUser] = useState(null);
  const [authEnabled, setAuthEnabled] = useState(true); // Default to enabled

  useEffect(() => {
    // Check if authentication is enabled by fetching feature flags
    const checkAuthFeatureFlag = async () => {
      try {
        const response = await fetch('/api/v1/feature-flags');
        const featureFlags = await response.json();
        setAuthEnabled(featureFlags.auth || false);
      } catch (error) {
        console.error('Error fetching feature flags:', error);
        // Default to enabled if there's an error
        setAuthEnabled(true);
      }
    };

    checkAuthFeatureFlag();

    // Check if user is already authenticated
    const token = localStorage.getItem('authToken');
    const storedUser = localStorage.getItem('user');

    if (token && storedUser) {
      setIsAuthenticated(true);
      setUser(JSON.parse(storedUser));
    } else {
      setIsAuthenticated(false);
      setUser(null);
    }
  }, []);

  const handleAuthSuccess = (userData) => {
    setIsAuthenticated(true);
    setUser(userData);
    setShowAuthModal(false);
  };

  const handleLogout = () => {
    localStorage.removeItem('authToken');
    localStorage.removeItem('user');
    setIsAuthenticated(false);
    setUser(null);
  };

  const openAuthModal = () => {
    setShowAuthModal(true);
  };

  const closeAuthModal = () => {
    setShowAuthModal(false);
  };

  // If auth is not enabled, don't show the personalization button
  if (!authEnabled) {
    return null;
  }

  if (!isAuthenticated) {
    return (
      <div className="personalization-container">
        <button
          className="personalization-btn login-prompt"
          onClick={openAuthModal}
        >
          <span className="btn-icon">👤</span>
          <span>Login for Personalized Content</span>
        </button>

        <AuthModal
          isOpen={showAuthModal}
          onClose={closeAuthModal}
          onAuthSuccess={handleAuthSuccess}
        />
      </div>
    );
  }

  return (
    <div className="personalization-container">
      <div className="personalization-header">
        <button className="personalization-btn user-info" onClick={() => {}}>
          <span className="btn-icon">👤</span>
          <span className="user-name">{user?.name || 'User'}</span>
          <span className="user-background">
            {user?.software_background && user.hardware_background
              ? `${user.software_background}/${user.hardware_background}`
              : 'Background Set'}
          </span>
        </button>
        <button className="logout-btn" onClick={handleLogout} title="Logout">
          <span>Logout</span>
        </button>
      </div>

      <div className="personalization-options">
        <div className="option-card">
          <h4>Content Preferences</h4>
          <p>Your background: {user?.software_background} software, {user?.hardware_background} hardware</p>
          <p>Content will be tailored to your experience level.</p>
        </div>

        <div className="option-card">
          <h4>Learning Path</h4>
          <p>Based on your background, we recommend starting with fundamentals.</p>
        </div>
      </div>
    </div>
  );
};

export default PersonalizationButton;