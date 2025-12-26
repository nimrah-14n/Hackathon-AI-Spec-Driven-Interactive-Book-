import React, { useState, useEffect } from 'react';
import useTranslation from '../../hooks/useTranslation';
import TranslationToggle from '../TranslationToggle/TranslationToggle';
import styles from './DocumentTranslation.module.css';

const DocumentTranslation = ({ children, title, contentKey }) => {
  const [documentContent, setDocumentContent] = useState(children);
  const [contentId, setContentId] = useState('');

  // Initialize content ID based on title or contentKey
  useEffect(() => {
    const id = contentKey || (title ? title.toLowerCase().replace(/\s+/g, '-') : 'content');
    setContentId(id);
  }, [title, contentKey]);

  const {
    content,
    currentLang,
    isTranslating,
    toggleLanguage,
    translateTo,
    resetToOriginal,
    setOriginalContent
  } = useTranslation(children, 'en');

  // Update original content when children change
  useEffect(() => {
    setOriginalContent(children);
  }, [children, setOriginalContent]);

  // Update document content when translation changes
  useEffect(() => {
    setDocumentContent(content);
  }, [content]);

  return (
    <div className={styles.documentTranslation} id={`translation-container-${contentId}`}>
      <div className={styles.translationHeader}>
        <TranslationToggle
          currentLang={currentLang}
          onToggle={toggleLanguage}
          isTranslating={isTranslating}
        />

        {currentLang !== 'en' && (
          <button
            onClick={resetToOriginal}
            className={styles.resetButton}
            title="Reset to original English content"
            aria-label="Reset to original content"
          >
            🔄 Original
          </button>
        )}
      </div>

      <div
        className={`${styles.translatedContent} ${isTranslating ? styles.translating : ''}`}
        lang={currentLang}
        dir={currentLang === 'ur' ? 'rtl' : 'ltr'}
      >
        {isTranslating && (
          <div className={styles.loadingOverlay}>
            <div className={styles.loadingSpinner}>
              <div className={styles.spinner}></div>
              <span>Translating content...</span>
            </div>
          </div>
        )}
        {documentContent}
      </div>

      <div className={styles.translationFooter}>
        <small className={styles.translationNote}>
          {currentLang === 'ur'
            ? 'اس صفحہ کا ترجمہ جاری ہے'
            : 'Translation in progress'
          }
        </small>
      </div>
    </div>
  );
};

export default DocumentTranslation;