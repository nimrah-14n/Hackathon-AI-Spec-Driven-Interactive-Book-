import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import { ChatbotProvider } from '../components/RAGChat/ChatbotProvider';
import ChatbotToggle from '../components/RAGChat/ChatbotToggle';

export default function Layout(props) {
  return (
    <ChatbotProvider>
      <OriginalLayout {...props}>
        {props.children}
        <ChatbotToggle />
      </OriginalLayout>
    </ChatbotProvider>
  );
}