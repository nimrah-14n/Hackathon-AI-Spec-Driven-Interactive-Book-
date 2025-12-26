import React from 'react';
import PersonalizationButton from './PersonalizationButton';
import './ChapterPersonalization.css';

const ChapterPersonalization = ({ chapterId, userId = null }) => {
  return (
    <div className="chapter-personalization-container">
      <div className="chapter-personalization-header">
        <h3>Personalize Your Learning Experience</h3>
        <p>Sign in to customize content based on your software and hardware background</p>
      </div>
      <PersonalizationButton chapterId={chapterId} userId={userId} />
    </div>
  );
};

export default ChapterPersonalization;