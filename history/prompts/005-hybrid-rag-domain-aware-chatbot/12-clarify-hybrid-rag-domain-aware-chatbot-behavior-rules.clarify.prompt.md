---
id: 12
title: "Clarify Hybrid RAG Domain-Aware Chatbot Behavior Rules"
stage: "clarify"
date_iso: "2025-12-26"
surface: "agent"
model: "Claude Sonnet 4.5"
feature: "005-hybrid-rag-domain-aware-chatbot"
branch: "005-hybrid-rag-domain-aware-chatbot"
user: "user"
command: "sp.clarify"
labels: ["clarify", "hybrid-rag", "domain-aware", "chatbot", "urdu", "behavior-rules"]
links:
  spec: "specs/005-hybrid-rag-domain-aware-chatbot/spec.md"
  clarify: "specs/005-hybrid-rag-domain-aware-chatbot/clarifications.md"
  ticket: null
  adr: null
  pr: null
files_yaml: |
  - specs/005-hybrid-rag-domain-aware-chatbot/clarifications.md
  - specs/005-hybrid-rag-domain-aware-chatbot/spec.md
  - specs/005-hybrid-rag-domain-aware-chatbot/tasks.md
  - history/prompts/005-hybrid-rag-domain-aware-chatbot/12-clarify-hybrid-rag-domain-aware-chatbot-behavior-rules.clarify.prompt.md
tests_yaml: |
  - none
---

## Prompt Text
```
/sp.clarify

Clarify chatbot behavior rules:

Answer Priority:
1. Book Content (RAG)
2. AI / Robotics / ML / Education knowledge
3. Otherwise refuse

Language Rules:
- Default language: English
- If user asks in Urdu OR clicks "Translate to Urdu":
  → Respond in Urdu
- Urdu must be:
  - Clear
  - Formal
  - Educational
  - Culturally respectful

Response Labels:
📘 From Book Content
🤖 From AI & Robotics Knowledge
🌐 Translated to Urdu

Refusal Message (English):
"I'm sorry, I can only answer questions related to AI, Robotics, or this book."

Refusal Message (Urdu):
"معذرت کے ساتھ، میں صرف مصنوعی ذہانت، روبوٹکس یا اس کتاب سے متعلق سوالات کا جواب دے سکتا ہوں۔"
```

## Response Text
Created comprehensive clarifications for the Hybrid RAG + Domain-Aware AI Chatbot with Urdu Support. The clarifications document defines specific behavior rules including: 1) Strict answer priority hierarchy (Book Content → AI/Robotics/ML/Education knowledge → Refusal), 2) Detailed language handling rules with formal Urdu standards, 3) Specific response labeling using icons (📘, 🤖, 🌐), and 4) Exact refusal messages in both English and Urdu. The main specification and tasks have been updated to incorporate these clarifications, with functional requirements refined to reflect the specific domains (AI/Robotics/ML/Education) and precise labeling requirements.

## Outcome
Successfully clarified the chatbot behavior rules with specific, implementable requirements. The clarifications provide clear guidance for development while maintaining the educational focus and cultural sensitivity requirements. All related artifacts (spec, tasks) have been updated to reflect the clarified requirements, ensuring consistent implementation of the answer priority, language handling, and response labeling rules.

## Evaluation
The implementation successfully addressed the requirements:
- [X] Created clarifications document with specific behavior rules
- [X] Updated spec with detailed functional requirements reflecting clarifications
- [X] Updated tasks to include specific implementation steps for clarified requirements
- [X] Ensured consistency across all related artifacts
- [X] Generated PHR record in appropriate directory