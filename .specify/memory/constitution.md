<!-- SYNC IMPACT REPORT
Version change: N/A -> 1.0.0
Modified principles: None (new constitution)
Added sections: All sections (new project constitution)
Removed sections: None
Templates requiring updates:
- .specify/templates/plan-template.md ⚠ pending
- .specify/templates/spec-template.md ⚠ pending
- .specify/templates/tasks-template.md ⚠ pending
- .specify/templates/commands/*.md ⚠ pending
Follow-up TODOs: None
-->

# AI/Spec-Driven Interactive Book with Integrated RAG Chatbot Constitution


## Core Principles

### Specification-First Development
All development starts with clear specifications using Spec-Kit Plus as the source of truth; All technical claims must be implementable and reproducible; Accuracy and technical correctness must be verifiable

### Accuracy and Technical Correctness
All implementations must be verifiable and technically accurate; No hallucinated technical details; Clear separation of frontend, backend, AI, and data layers

### Reusability Through Subagents and Skills
Implement reusable intelligence via Claude Code Subagents and Agent Skills documented and reused across chapters; Personalization logic must depend on explicitly collected user background data

### Security-by-Design
Authentication must use Better Auth as the identity layer; Secure handling of authentication tokens and user data; All API contracts must follow secure practices

### Modularity and Maintainability
Clean API contracts between services; Consistent terminology throughout the book; Book structure must follow Docusaurus best practices

### RAG Integrity
RAG responses must be grounded strictly in indexed book content; Selected-text-only question answering must not hallucinate beyond provided text


## Technology Stack Requirements

Frontend: Docusaurus (React-based); Backend: FastAPI; AI Layer: OpenAI Agents / ChatKit SDKs; Vector Database: Qdrant Cloud (Free Tier); Relational Database: Neon Serverless Postgres; Authentication: Better Auth; Deployment: GitHub Pages


## Development Workflow

All functional requirements must be implemented: AI/Spec-driven book creation, embedded RAG chatbot, user authentication with background collection, content personalization, and Urdu translation support; Quality constraints must be met throughout development


## Governance

This Constitution supersedes all other practices; Amendments require documentation and approval; All PRs/reviews must verify compliance with all principles; Complexity must be justified by user requirements

**Version**: 1.0.0 | **Ratified**: 2025-12-15 | **Last Amended**: 2025-12-15