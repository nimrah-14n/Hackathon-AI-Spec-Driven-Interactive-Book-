import React, { useEffect } from 'react';
import { useLocation } from '@docusaurus/router';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import DocRootOriginal from '@theme-original/DocRoot';
import DocumentTranslation from '../components/DocumentTranslation/DocumentTranslation';
import { useDocsPreferredVersion } from '@docusaurus/theme-common';

// Custom DocRoot component that adds translation functionality to documentation pages
const DocRoot = (props) => {
  const { route } = props;
  const location = useLocation();
  const { siteConfig } = useDocusaurusContext();
  const { metadata, children } = route;

  // Extract document title and create content key
  const documentTitle = metadata?.title || 'Documentation';
  const contentKey = metadata?.source?.replace(/\//g, '-').replace(/\.md$/, '') || 'doc-' + Date.now();

  // Get the current locale from the site config
  const locale = siteConfig.i18n.defaultLocale; // Use default locale as fallback

  // Only add translation wrapper if we're in English locale (default)
  // This allows the system to work with Docusaurus' i18n system
  const shouldShowTranslation = locale === 'en';

  useEffect(() => {
    // Add any necessary initialization code here
    // For example, tracking translation usage or initializing language preferences
  }, []);

  return (
    <DocRootOriginal {...props}>
      {shouldShowTranslation ? (
        <DocumentTranslation
          title={documentTitle}
          contentKey={contentKey}
        >
          {children}
        </DocumentTranslation>
      ) : (
        // If not in English locale, show content as is
        children
      )}
    </DocRootOriginal>
  );
};

export default DocRoot;