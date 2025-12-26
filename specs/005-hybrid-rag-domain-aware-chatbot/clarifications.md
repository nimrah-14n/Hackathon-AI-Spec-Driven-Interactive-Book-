# Clarifications for Hybrid RAG + Domain-Aware AI Chatbot with Urdu Support

**Feature**: 005-hybrid-rag-domain-aware-chatbot
**Date**: 2025-12-26
**Status**: Confirmed

## Clarification Summary

This document confirms the specific behavior rules and constraints for the Hybrid RAG + Domain-Aware AI Chatbot implementation based on the detailed requirements provided.

## Detailed Clarifications

### 1. Answer Priority Hierarchy
**Requirement**: Chatbot must follow a strict answer priority hierarchy: 1) Book Content (RAG), 2) AI/Robotics/ML/Education knowledge, 3) Otherwise refuse
**Status**: CONFIRMED
**Details**: The chatbot will first attempt to find answers in the indexed book content using RAG. If no relevant information is found, it will attempt to answer using general AI, Robotics, Machine Learning, or Educational knowledge. If the question is outside these domains, it must politely refuse to answer. This ensures comprehensive but focused responses while maintaining educational relevance.

### 2. Language Handling Rules
**Requirement**: Default language is English; respond in Urdu when user asks in Urdu OR clicks "Translate to Urdu"
**Status**: CONFIRMED
**Details**: The system defaults to English responses. However, if a user asks a question in Urdu or explicitly requests Urdu translation via UI toggle, all responses must be provided in Urdu. This includes responses to follow-up questions in the same session. The language preference should persist for the duration of the interaction unless explicitly changed.

### 3. Urdu Response Quality Standards
**Requirement**: Urdu responses must be clear, formal, educational, and culturally respectful
**Status**: CONFIRMED
**Details**: All Urdu translations and responses must maintain high standards of clarity and formality appropriate for educational content. Technical terminology should be preserved where appropriate, and cultural sensitivity must be maintained. Responses should follow formal Urdu grammar and structure suitable for academic or professional contexts.

### 4. Response Source Labeling
**Requirement**: All responses must be clearly labeled with source indicators
**Status**: CONFIRMED
**Details**: Responses must be labeled with appropriate icons and text:
- 📘 From Book Content (for RAG-based responses)
- 🤖 From AI & Robotics Knowledge (for general knowledge responses)
- 🌐 Translated to Urdu (for language indication)
These labels ensure transparency about the source of information and help users understand the authority level of responses.

### 5. Domain Boundary Enforcement
**Requirement**: Strict refusal for non-AI/Robotics/Education questions with specific message formats
**Status**: CONFIRMED
**Details**: The system must politely refuse questions outside the AI, Robotics, Machine Learning, or Educational domains. The refusal message must be provided in the appropriate language:
- English: "I'm sorry, I can only answer questions related to AI, Robotics, or this book."
- Urdu: "معذرت کے ساتھ، میں صرف مصنوعی ذہانت، روبوٹکس یا اس کتاب سے متعلق سوالات کا جواب دے سکتا ہوں۔"
This ensures the chatbot remains focused on its educational purpose.

### 6. Hybrid RAG Implementation
**Requirement**: Implementation must support both RAG and general knowledge fallback
**Status**: CONFIRMED
**Details**: The system architecture must implement a hybrid approach where RAG search is attempted first, and only when no relevant book content is found should the system fall back to general AI/Robotics knowledge. This requires careful implementation to ensure the RAG priority is maintained while still providing helpful responses.

## Implementation Impact

These clarifications have been incorporated into the main specification document at `specs/005-hybrid-rag-domain-aware-chatbot/spec.md` and affect:

- Functional Requirements (FR-001, FR-002, FR-003, FR-004, FR-005)
- Edge Cases analysis (domain boundaries, language handling, response labeling)
- Success Criteria (SC-002, SC-003, SC-004, SC-007, SC-008)

## Next Steps

With these clarifications confirmed, the specification is ready for the implementation phase. All behavior rules are clearly defined and incorporated into the specification. The development team can proceed with implementing the hybrid RAG + domain-aware chatbot with the specified language and response handling requirements.