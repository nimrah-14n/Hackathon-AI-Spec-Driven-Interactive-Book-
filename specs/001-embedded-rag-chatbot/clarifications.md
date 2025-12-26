# Clarifications for Embedded RAG Chatbot Implementation

**Feature**: 001-embedded-rag-chatbot
**Date**: 2025-12-22
**Status**: Confirmed

## Clarification Summary

This document confirms the specific requirements and constraints for the RAG chatbot implementation based on Phase-2 scope confirmation.

## Detailed Clarifications

### 1. RAG Chatbot Content Grounding
**Requirement**: RAG chatbot must never use general LLM knowledge
**Status**: CONFIRMED
**Details**: The chatbot will be strictly limited to answering based only on the embedded book content. All responses must be grounded in the indexed book materials, with clear refusal to answer questions outside this scope. This is implemented through the content grounding requirement (FR-003) and verified in success criteria (SC-002).

### 2. Selected-Text-Only Mode Isolation
**Requirement**: Selected-text-only mode must override global context completely
**Status**: CONFIRMED
**Details**: When in selected-text-only mode, the chatbot will only consider the specific text that the user has selected, with zero influence from the broader book context. This ensures complete context isolation as specified in FR-011 and SC-005.

### 3. Separate Backend and Frontend Deployment
**Requirement**: Backend and frontend are deployed separately
**Status**: CONFIRMED
**Details**: The FastAPI backend will be deployed independently from the Docusaurus frontend. This is captured in FR-004, enabling independent scaling and maintenance of each component.

### 4. Optional Authentication and Personalization
**Requirement**: Authentication and personalization are feature-flagged (optional, non-blocking)
**Status**: CONFIRMED
**Details**: These features will be implemented behind feature flags (FR-010) and will not block the core functionality. Success criteria includes verification that these can be toggled without affecting core chatbot functionality (SC-008).

### 5. Urdu Translation Scope
**Requirement**: Urdu translation applies to chapter content, not chatbot responses (initial scope)
**Status**: CONFIRMED
**Details**: The initial scope includes Urdu content for book chapters but does not extend to chatbot response generation. This is documented in FR-012 and considered in the edge cases analysis.

### 6. Graceful Backend Failure Handling
**Requirement**: If RAG backend is unavailable, frontend must fail gracefully
**Status**: CONFIRMED
**Details**: The frontend will continue to function even when the RAG backend is down, providing appropriate user messaging. This is captured in FR-013 and verified through success criteria SC-007.

### 7. Vercel Deployment Strategy
**Requirement**: Default deployment remains Vercel (frontend)
**Status**: CONFIRMED
**Details**: The frontend will be deployed to Vercel as the default deployment strategy, as specified in FR-014.

## Implementation Impact

These clarifications have been incorporated into the main specification document at `specs/001-embedded-rag-chatbot/spec.md` and affect:

- Functional Requirements (FR-003, FR-004, FR-010, FR-011, FR-012, FR-013, FR-014)
- Edge Cases analysis (backend failure, context isolation, content scope)
- Success Criteria (SC-002, SC-005, SC-007, SC-008)

## Next Steps

With these clarifications confirmed, the specification is ready for the planning phase (`/sp.plan`). All Phase-2 scope requirements are clearly defined and incorporated into the specification.