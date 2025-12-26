<!-- SYNC IMPACT REPORT
Version change: 1.0.0 -> 1.1.0
Modified principles: Updated to reflect Embedded RAG Chatbot for Spec-Driven Interactive AI Book requirements
Added sections: Chatbot Behavior Rules, RAG Constraints, UI Principles, AI-Native Spec-Driven Robotics Learning enhancements
Removed sections: None
Templates requiring updates:
- .specify/templates/plan-template.md ⚠ pending
- .specify/templates/spec-template.md ⚠ pending
- .specify/templates/tasks-template.md ⚠ pending
- .specify/templates/commands/*.md ⚠ pending
Follow-up TODOs: None
-->

# Embedded RAG Chatbot for Spec-Driven Interactive AI Book Constitution


## Core Principles

### Specification-First Development
All development starts with clear specifications using Spec-Kit Plus as the source of truth; All technical claims must be implementable and reproducible; Accuracy and technical correctness must be verifiable

### Preservation of Existing Systems
Do not break existing Docusaurus book or deployment; Changes must be incremental and reversible; Follow Spec-Driven Development strictly

### Clean Separation of Concerns
Backend and frontend must be cleanly separated; Clear separation of frontend, backend, AI, and data layers; Clean API contracts between services

### Security-by-Design
All secrets must use environment variables; Secure handling of authentication tokens and user data; All API contracts must follow secure practices; Deployment must not fail due to missing data

### Modularity and Maintainability
Consistent terminology throughout the book; Book structure must follow Docusaurus best practices; Quality constraints must be met throughout development


## Chatbot Behavior Rules

### Respectful and Professional Interaction
Responses must be respectful, polite, and professional; Tone should be friendly, calm, and helpful; No hallucinations or guessing

### Context-Aware Response Handling
If information is not found in book context, respond: "I'm sorry, I couldn't find this information in the book content."; Maintain respectful academic tone; Avoid hallucination; Prefer clarity over verbosity


## RAG Constraints

### Strict Context Adherence
Chatbot must answer ONLY from book content; Must support "Answer from selected text only" mode; Must refuse answering outside provided context; RAG responses must be grounded strictly in indexed book content; Selected-text-only question answering must not hallucinate beyond provided text


## UI Principles

### Clean and Accessible Design
Clean, minimal, readable chatbot UI; Accessible font sizes and contrast; Non-intrusive integration with Docusaurus book

### Visual Aesthetics
Use soft pastel colors (light yellow, orange, pink); Smooth animations (fade / slide)


## Technology Stack Requirements

Frontend: Docusaurus (React-based); Backend: FastAPI; AI Layer: OpenAI Agents / ChatKit SDKs; Vector Database: Qdrant Cloud (Free Tier); Relational Database: Neon Serverless Postgres; Authentication: Better Auth; Deployment: GitHub Pages


## AI-Native Spec-Driven Robotics Learning Book Principles

### Educational Excellence
Spec-driven development; Educational correctness; Safe and respectful AI responses

### Cultural Inclusivity
Cultural inclusivity (Urdu support); Support for multilingual content delivery

### Intelligence Modularity
Modular reusable intelligence via subagents; Hackathon-safe deployment

## Development Workflow

All functional requirements must be implemented: AI/Spec-driven book creation, embedded RAG chatbot, user authentication with background collection, content personalization, and Urdu translation support; Quality constraints must be met throughout development


## Governance

This Constitution supersedes all other practices; Amendments require documentation and approval; All PRs/reviews must verify compliance with all principles; Complexity must be justified by user requirements

**Version**: 1.1.0 | **Ratified**: 2025-12-15 | **Last Amended**: 2025-12-22