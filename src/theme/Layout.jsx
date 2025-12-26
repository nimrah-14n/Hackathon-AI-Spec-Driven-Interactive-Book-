import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import { ChatbotProvider } from '../components/RAGChat/ChatbotProvider';
import ChatbotToggle from '../components/RAGChat/ChatbotToggle';
import PersonalizationButton from '../components/Personalization/PersonalizationButton';

export default function Layout(props) {
  return (
    <ChatbotProvider>
      <OriginalLayout {...props}>
        {props.children}
        <PersonalizationButton />
        <ChatbotToggle />
      </OriginalLayout>
    </ChatbotProvider>
  );
}