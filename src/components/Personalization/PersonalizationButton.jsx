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
        // Determine API base URL based on environment
        const apiBaseUrl = process.env.NODE_ENV === 'production'
          ? '' // Use relative path in production
          : 'http://localhost:8000'; // Use full URL in development

        const response = await fetch(`${apiBaseUrl}/api/v1/feature-flags`);
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

  useEffect(() => {
    // Close dropdown when clicking outside
    const handleClickOutside = (event) => {
      const personalizationContainer = document.querySelector('.personalization-container');
      if (showAuthModal && personalizationContainer && !personalizationContainer.contains(event.target)) {
        setShowAuthModal(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showAuthModal]);

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

  return (
    <div className="personalization-container">
      <button
        className="personalization-btn"
        onClick={() => {
          if (isAuthenticated) {
            // Toggle visibility of options for authenticated users
            setShowAuthModal(!showAuthModal);
          } else {
            // Open auth modal for non-authenticated users
            openAuthModal();
          }
        }}
        title={isAuthenticated ? 'View Profile Options' : 'Sign In / Sign Up'}
      >
        <span className="btn-icon">{isAuthenticated ? '👤' : '🔒'}</span>
      </button>

      {/* Auth Modal for login/signup */}
      <AuthModal
        isOpen={showAuthModal && !isAuthenticated}
        onClose={closeAuthModal}
        onAuthSuccess={handleAuthSuccess}
      />

      {/* Dropdown options for authenticated users */}
      {isAuthenticated && showAuthModal && (
        <div className="personalization-options">
          <div className="personalization-header">
            <div className="user-info">
              <span className="user-name">{user?.name || 'User'}</span>
              <span className="user-background">
                {user?.software_background && user.hardware_background
                  ? `${user.software_background}/${user.hardware_background}`
                  : 'Background Set'}
              </span>
            </div>
            <button className="logout-btn" onClick={handleLogout} title="Logout">
              <span>Logout</span>
            </button>
          </div>

          <div className="option-card">
            <h4>Content Preferences</h4>
            <p>Your background: {user?.software_background} software, {user?.hardware_background} hardware</p>
            <p>Content tailored to your experience level.</p>
          </div>

          <div className="option-card">
            <h4>Learning Path</h4>
            <p>Based on your background, personalized recommendations.</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default PersonalizationButton;