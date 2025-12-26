import React, { useState, useEffect, useRef } from 'react';
import useTranslation from '../../hooks/useTranslation';
import TranslationToggle from '../TranslationToggle/TranslationToggle';
import styles from './ChapterTranslation.module.css';

const ChapterTranslation = ({ children, chapterTitle, chapterId }) => {
  const [content, setContent] = useState('');
  const [headings, setHeadings] = useState([]);
  const [translatedHeadings, setTranslatedHeadings] = useState({});
  const contentRef = useRef(null);

  const {
    content: translatedContent,
    currentLang,
    isTranslating,
    toggleLanguage,
    translateTo,
    resetToOriginal,
    setOriginalContent
  } = useTranslation('', 'en');

  // Extract headings from content
  useEffect(() => {
    if (contentRef.current) {
      const headingElements = contentRef.current.querySelectorAll('h1, h2, h3, h4, h5, h6');
      const extractedHeadings = Array.from(headingElements).map((heading, index) => ({
        id: heading.id || `heading-${index}`,
        text: heading.textContent,
        level: heading.tagName.toLowerCase()
      }));
      setHeadings(extractedHeadings);
    }
  }, [translatedContent]);

  // Process content to handle headings separately for translation
  useEffect(() => {
    const processContent = async () => {
      // Set the original content for translation
      const contentString = typeof children === 'string' ? children :
        (children.props?.children || '').toString();

      setOriginalContent(contentString);
    };

    processContent();
  }, [children, setOriginalContent]);

  // Handle language change for headings
  useEffect(() => {
    if (headings.length > 0) {
      const translateHeadings = async () => {
        const newTranslatedHeadings = {};
        for (const heading of headings) {
          if (currentLang === 'ur') {
            newTranslatedHeadings[heading.id] = `[URDU] ${heading.text} [ENGLISH]`;
          } else {
            newTranslatedHeadings[heading.id] = heading.text.replace(/\[URDU\] (.+?) \[ENGLISH\]/g, '$1');
          }
        }
        setTranslatedHeadings(newTranslatedHeadings);
      };

      translateHeadings();
    }
  }, [headings, currentLang]);

  // Render content with translated headings
  const renderContentWithHeadings = () => {
    if (typeof children === 'string') {
      // If children is a string, return it wrapped in the translation component
      return (
        <div
          ref={contentRef}
          className={`${styles.chapterContent} ${currentLang === 'ur' ? styles.urduContent : ''}`}
          lang={currentLang}
          dir={currentLang === 'ur' ? 'rtl' : 'ltr'}
        >
          {translatedContent}
        </div>
      );
    } else {
      // If children is a React element, clone it with translation
      return React.cloneElement(children, {
        ref: contentRef,
        className: `${children.props?.className || ''} ${styles.chapterContent} ${currentLang === 'ur' ? styles.urduContent : ''}`,
        lang: currentLang,
        dir: currentLang === 'ur' ? 'rtl' : 'ltr',
        key: `translated-${currentLang}-${chapterId}`
      });
    }
  };

  return (
    <div className={styles.chapterTranslation} id={`chapter-translation-${chapterId}`}>
      <div className={styles.translationToolbar}>
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

      <div className={styles.translationContent}>
        {isTranslating && (
          <div className={styles.loadingOverlay}>
            <div className={styles.loadingSpinner}>
              <div className={styles.spinner}></div>
              <span>Translating chapter content...</span>
            </div>
          </div>
        )}

        {renderContentWithHeadings()}
      </div>

      {currentLang === 'ur' && (
        <div className={styles.translationNotice}>
          <small>
            {chapterTitle && `Translation of: ${chapterTitle}`}
          </small>
        </div>
      )}
    </div>
  );
};

export default ChapterTranslation;