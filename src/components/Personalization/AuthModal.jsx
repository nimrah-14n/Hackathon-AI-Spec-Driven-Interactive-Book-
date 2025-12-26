import React, { useState } from 'react';
import { useChatbot } from '../RAGChat/ChatbotProvider';
import './AuthModal.css';

const AuthModal = ({ isOpen, onClose, onAuthSuccess }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    name: '',
    softwareBackground: 'none',
    hardwareBackground: 'none'
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  if (!isOpen) return null;

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    // Clear error when user starts typing
    if (error) setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      const endpoint = isLogin ? 'http://localhost:8000/auth/login' : 'http://localhost:8000/auth/register';
      const requestBody = isLogin
        ? { email: formData.email, password: formData.password }
        : {
            email: formData.email,
            password: formData.password,
            name: formData.name,
            software_background: formData.softwareBackground,
            hardware_background: formData.hardwareBackground
          };

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Authentication failed: ${response.status}`);
      }

      const data = await response.json();

      // Store token in localStorage (in a real app, consider more secure storage)
      localStorage.setItem('authToken', data.token);
      localStorage.setItem('user', JSON.stringify(data.user));

      // Notify parent component of successful auth
      onAuthSuccess(data.user);
      onClose();
    } catch (err) {
      console.error('Auth error:', err);
      setError(err.message || 'Authentication failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const toggleMode = () => {
    setIsLogin(!isLogin);
    setError('');
    setFormData({
      email: '',
      password: '',
      name: '',
      softwareBackground: 'none',
      hardwareBackground: 'none'
    });
  };

  return (
    <div className="auth-modal-overlay" onClick={onClose}>
      <div className="auth-modal" onClick={(e) => e.stopPropagation()}>
        <div className="auth-modal-header">
          <h2>{isLogin ? 'Welcome Back' : 'Create Account'}</h2>
          <button className="auth-modal-close" onClick={onClose}>×</button>
        </div>

        <form className="auth-modal-form" onSubmit={handleSubmit}>
          {!isLogin && (
            <div className="form-group">
              <label htmlFor="name">Full Name</label>
              <input
                type="text"
                id="name"
                name="name"
                value={formData.name}
                onChange={handleInputChange}
                required={!isLogin}
                disabled={isLoading}
              />
            </div>
          )}

          <div className="form-group">
            <label htmlFor="email">Email</label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleInputChange}
              required
              disabled={isLoading}
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleInputChange}
              required
              disabled={isLoading}
            />
          </div>

          {!isLogin && (
            <>
              <div className="form-group">
                <label>Software Background</label>
                <div className="radio-group">
                  {['none', 'beginner', 'intermediate', 'advanced'].map((level) => (
                    <label key={level} className="radio-option">
                      <input
                        type="radio"
                        name="softwareBackground"
                        value={level}
                        checked={formData.softwareBackground === level}
                        onChange={handleInputChange}
                        disabled={isLoading}
                      />
                      <span className="radio-label">{level.charAt(0).toUpperCase() + level.slice(1)}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="form-group">
                <label>Hardware Background</label>
                <div className="radio-group">
                  {['none', 'beginner', 'intermediate', 'advanced'].map((level) => (
                    <label key={level} className="radio-option">
                      <input
                        type="radio"
                        name="hardwareBackground"
                        value={level}
                        checked={formData.hardwareBackground === level}
                        onChange={handleInputChange}
                        disabled={isLoading}
                      />
                      <span className="radio-label">{level.charAt(0).toUpperCase() + level.slice(1)}</span>
                    </label>
                  ))}
                </div>
              </div>
            </>
          )}

          {error && <div className="auth-error">{error}</div>}

          <button
            type="submit"
            className="auth-submit-btn"
            disabled={isLoading}
          >
            {isLoading ? 'Processing...' : (isLogin ? 'Sign In' : 'Sign Up')}
          </button>
        </form>

        <div className="auth-modal-footer">
          <p>
            {isLogin ? "Don't have an account? " : "Already have an account? "}
            <button className="auth-toggle-btn" onClick={toggleMode}>
              {isLogin ? 'Sign Up' : 'Sign In'}
            </button>
          </p>
        </div>
      </div>
    </div>
  );
};

export default AuthModal;