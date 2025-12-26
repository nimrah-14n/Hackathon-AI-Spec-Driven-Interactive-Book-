import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import { ChatbotProvider } from '../components/RAGChat/ChatbotProvider';
import ChatbotToggle from '../components/RAGChat/ChatbotToggle';
import PersonalizationButton from '../components/Personalization/PersonalizationButton';

export default function Layout(props) {
  return (
    <ChatbotProvider>
      <div style={{ position: 'relative', minHeight: '100vh' }}>
        <OriginalLayout {...props}>
          {props.children}
        </OriginalLayout>
        <div style={{
          position: 'absolute',
          top: '1rem',
          right: '120px',  // Positioned to the left of GitHub button with proper spacing
          zIndex: 1000
        }}>
          <PersonalizationButton />
        </div>
        <ChatbotToggle />
      </div>
    </ChatbotProvider>
  );
}