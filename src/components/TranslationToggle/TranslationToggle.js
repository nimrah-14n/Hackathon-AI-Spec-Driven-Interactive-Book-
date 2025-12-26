import React from 'react';
import styles from './TranslationToggle.module.css';

const TranslationToggle = ({ currentLang, onToggle, isTranslating }) => {
  const toggleTranslation = () => {
    if (!isTranslating) {
      onToggle();
    }
  };

  return (
    <div className={styles.translationToggle}>
      <button
        onClick={toggleTranslation}
        disabled={isTranslating}
        className={`${styles.toggleButton} ${currentLang === 'ur' ? styles.urduActive : styles.englishActive}`}
        aria-label={`Switch to ${currentLang === 'en' ? 'Urdu' : 'English'} language`}
        title={`Current language: ${currentLang === 'en' ? 'English' : 'Urdu'}`}
      >
        {isTranslating ? (
          <span className={styles.loading}>
            <span className={styles.spinner}></span>
            Translating...
          </span>
        ) : (
          <span className={styles.toggleText}>
            {currentLang === 'en' ? (
              <>
                <span className={styles.flag}>🇺🇸</span> English
              </>
            ) : (
              <>
                <span className={styles.flag}>🇵🇰</span> اردو
              </>
            )}
            <span className={styles.switchIcon}>⇄</span>
          </span>
        )}
      </button>

      <div className={styles.languageIndicator}>
        <span className={styles.langBadge}>
          {currentLang === 'en' ? 'EN' : 'UR'}
        </span>
      </div>
    </div>
  );
};

export default TranslationToggle;