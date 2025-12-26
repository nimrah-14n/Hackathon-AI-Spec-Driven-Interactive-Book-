---
id: 1
title: "Update Embedded RAG Chatbot Constitution"
stage: "constitution"
date_iso: "2025-12-22"
surface: "agent"
model: "Claude Sonnet 4.5"
feature: "embedded-rag-chatbot"
branch: "main"
user: "user"
command: "sp.constitution"
labels: ["constitution", "embedded-rag", "chatbot", "spec-driven"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files_yaml: |
  - .specify/memory/constitution.md
tests_yaml: |
  - none
---

## Prompt Text
```
/sp.constitution

Project: Embedded RAG Chatbot for Spec-Driven Interactive AI Book

Core Principles:
- Do not break existing Docusaurus book or deployment
- Changes must be incremental and reversible
- Follow Spec-Driven Development strictly
- Backend and frontend must be cleanly separated
- All secrets must use environment variables
- Deployment must not fail due to missing data

Chatbot Behavior Rules:
- Responses must be respectful, polite, and professional
- Tone should be friendly, calm, and helpful
- No hallucinations or guessing
- If information is not found in book context, respond:
  "I'm sorry, I couldn't find this information in the book content."

RAG Constraints:
- Chatbot must answer ONLY from book content
- Must support "Answer from selected text only" mode
- Must refuse answering outside provided context

UI Principles:
- Clean, minimal, readable chatbot UI
- Use soft pastel colors (light yellow, orange, pink)
- Smooth animations (fade / slide)
- Accessible font sizes and contrast
- Non-i ,

Project: AI-Native Spec-Driven Robotics Learning Book with RAG

Core Principles:
- Spec-driven development
- Educational correctness
- Safe and respectful AI responses
- Cultural inclusivity (Urdu support)
- Modular reusable intelligence via subagents
- Hackathon-safe deployment

AI Behavior Rules:
- Answer only from provided context
- If context is missing, say so clearly
- Maintain respectful academic tone
- Avoid hallucination
- Prefer clarity over verbosity
```

## Response Text
Updated the project constitution to reflect the requirements for an Embedded RAG Chatbot for Spec-Driven Interactive AI Book. The updated constitution includes core principles for preserving existing systems, clean separation of concerns, security-by-design, and modularity. Added specific sections for Chatbot Behavior Rules, RAG Constraints, UI Principles, and AI-Native Spec-Driven Robotics Learning principles. Version was incremented from 1.0.0 to 1.1.0 to reflect the addition of new principles and sections.

## Outcome
Successfully updated the constitution file with comprehensive principles for the embedded RAG chatbot project, including preservation of existing Docusaurus functionality, strict RAG adherence, UI guidelines, and cultural inclusivity features like Urdu support. The constitution now properly governs the development of the AI-native robotics learning book with integrated chatbot functionality.

## Evaluation
The implementation successfully addressed the requirements:
- [X] Updated core principles to include preservation of existing Docusaurus book
- [X] Added Chatbot Behavior Rules section with respectful interaction guidelines
- [X] Implemented RAG Constraints for strict context adherence
- [X] Included UI Principles with visual aesthetics requirements
- [X] Added AI-Native Spec-Driven Robotics Learning Book principles
- [X] Incremented version from 1.0.0 to 1.1.0
- [X] Updated amendment date to current date
- [X] Created PHR record in appropriate directory