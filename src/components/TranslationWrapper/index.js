import React, { useEffect, useState } from 'react';
import ChapterTranslation from '../ChapterTranslation/ChapterTranslation';

// This component serves as a wrapper that can be imported and used in MDX files
const TranslationWrapper = ({ children, title }) => {
  const [contentId, setContentId] = useState('default');

  useEffect(() => {
    // Generate content ID from title or use a default
    const id = title
      ? title.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '')
      : 'chapter-' + Date.now();
    setContentId(id);
  }, [title]);

  // In a real implementation, this would be used in MDX files like:
  // import TranslationWrapper from '@site/src/components/TranslationWrapper'
  // <TranslationWrapper title="Chapter Title">
  //   {children}
  // </TranslationWrapper>

  return (
    <ChapterTranslation
      chapterTitle={title}
      chapterId={contentId}
    >
      {children}
    </ChapterTranslation>
  );
};

export default TranslationWrapper;