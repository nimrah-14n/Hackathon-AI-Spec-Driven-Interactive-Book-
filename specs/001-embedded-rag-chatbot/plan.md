# Implementation Plan: Embedded RAG Chatbot for Spec-Driven AI Book

**Branch**: `001-embedded-rag-chatbot` | **Date**: 2025-12-22 | **Spec**: [link to spec.md](spec.md)
**Input**: Feature specification from `/specs/001-embedded-rag-chatbot/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implementation of an embedded RAG chatbot for the Docusaurus-based AI book that strictly answers from book content only. The solution includes markdown ingestion, vector search, and a clean UI integrated non-intrusively into the existing book pages. The architecture separates backend (FastAPI) and frontend (Docusaurus) with feature flags controlling optional bonus features.

## Technical Context

**Language/Version**: Python 3.11 (backend), JavaScript/TypeScript (frontend)
**Primary Dependencies**: FastAPI, OpenAI SDK, Qdrant, Docusaurus, React
**Storage**: Qdrant Cloud (vector database), Neon Serverless Postgres (metadata), Docusaurus markdown files
**Testing**: pytest (backend), Jest/React Testing Library (frontend)
**Target Platform**: Web (Linux server for backend, browser for frontend)
**Project Type**: web (separate backend and frontend)
**Performance Goals**: <5 second response time, 95% accuracy in content grounding, 100 concurrent users
**Constraints**: <10% impact on page load times, no breaking of existing Docusaurus functionality, graceful failure when backend unavailable
**Scale/Scope**: 10k+ book pages, multiple concurrent users, feature-flagged optional components

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- ✅ **Preservation of Existing Systems**: Plan ensures no breaking changes to existing Docusaurus book
- ✅ **Clean Separation of Concerns**: Backend (FastAPI) and frontend (Docusaurus) deployed separately
- ✅ **Security-by-Design**: All secrets use environment variables, secure API contracts
- ✅ **RAG Constraints**: Strict content adherence with no general LLM knowledge usage
- ✅ **UI Principles**: Clean, minimal UI with soft pastel colors and smooth animations
- ✅ **Cultural Inclusivity**: Support for Urdu translation in initial scope

## Project Structure

### Documentation (this feature)

```text
specs/001-embedded-rag-chatbot/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
├── clarifications.md    # Clarifications document
├── checklists/          # Quality checklists
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── models/
│   │   ├── document.py          # Document chunking and embedding models
│   │   ├── chat.py             # Chat interaction models
│   │   └── user.py             # User and session models (optional)
│   ├── services/
│   │   ├── ingestion.py        # Markdown ingestion service
│   │   ├── embedding.py        # Embedding and vector search service
│   │   ├── rag.py              # RAG orchestration service
│   │   ├── auth.py             # Authentication service (optional)
│   │   └── translation.py      # Translation service (optional)
│   ├── api/
│   │   ├── v1/
│   │   │   ├── chat.py         # Chat endpoints
│   │   │   ├── documents.py    # Document ingestion endpoints
│   │   │   ├── auth.py         # Auth endpoints (optional)
│   │   │   └── translation.py  # Translation endpoints (optional)
│   │   └── deps.py             # Dependency injection
│   ├── config/
│   │   ├── settings.py         # Configuration management
│   │   └── database.py         # Database connections
│   └── main.py                 # Application entry point
└── tests/
    ├── unit/
    ├── integration/
    └── contract/

frontend/
├── src/
│   ├── components/
│   │   ├── Chatbot/            # Main chatbot component
│   │   ├── ChatWindow/         # Chat window UI
│   │   ├── Message/            # Individual message component
│   │   ├── UIControls/         # UI theme and animation components
│   │   └── FeatureFlags/       # Feature flag management
│   ├── hooks/
│   │   ├── useChat.js          # Chat functionality hook
│   │   └── useFeatureFlags.js  # Feature flag hook
│   ├── services/
│   │   ├── api.js              # API client
│   │   └── chatService.js      # Chat business logic
│   └── styles/
│       └── chatbot.css         # Pastel theme styles
├── docusaurus/
│   ├── plugins/
│   │   └── docusaurus-chatbot-plugin/  # Docusaurus chatbot integration
│   └── src/
│       └── theme/
│           └── ChatbotInjector/        # Component that injects chatbot into pages
└── static/
    └── translations/           # Urdu translation files (optional)

# Existing Docusaurus structure preserved
docs/
├── ...
└── ...

blog/
├── ...
└── ...

docusaurus.config.js          # Updated to include chatbot plugin
```

**Structure Decision**: Web application with separate backend and frontend services. Backend uses FastAPI for RAG functionality, while frontend integrates the chatbot into Docusaurus pages via a custom plugin. This maintains clean separation while enabling non-intrusive integration with existing book content.

## Implementation Phases

### Phase 2A – Core RAG (Base Score - Mandatory)
**Objective**: Implement core RAG functionality with embedded chatbot UI

**Components**:
- Markdown ingestion pipeline from Docusaurus docs directory
- Document chunking with semantic boundaries and embedding generation
- Qdrant vector search integration for content retrieval
- FastAPI RAG endpoint with strict context-only answering enforcement
- Embedded chatbot UI with clean, minimal design following UI principles

**Deliverables**:
- Backend API endpoints for chat and document processing (FastAPI)
- Frontend chatbot component with soft pastel theme (light yellow, pink, orange)
- Vector database populated with book content embeddings
- Content grounding enforcement with clear refusal outside book context
- Non-intrusive integration preserving existing Docusaurus functionality

**Success Criteria**:
- SC-001: Responses within 5 seconds
- SC-002: 95% content grounding accuracy
- SC-004: <10% page load impact
- SC-005: Mode switching functionality
- SC-007: Graceful failure handling

### Phase 2B – Intelligence & UX (Bonus - Feature Flagged)
**Objective**: Enhance with selected-text mode and intelligent features

**Components**:
- Selected-text-only answering mode that completely overrides global context
- Claude Code Subagents for advanced reasoning
- Agent Skills per chapter for contextual intelligence
- Enhanced pastel UI theme with smooth animations (fade/slide)

**Feature Flags**: `SELECTED_TEXT_MODE`, `SUBAGENTS_ENABLED`, `AGENT_SKILLS_ENABLED`

**Success Criteria**:
- SC-005: Complete context isolation in selected-text mode

### Phase 2C – Auth & Personalization (Bonus - Feature Flagged)
**Objective**: Add user authentication and personalization features

**Components**:
- Better-auth integration for user sessions
- User background collection interface
- Personalized chapter variants based on user profile
- Session persistence in Neon Serverless Postgres

**Feature Flags**: `AUTH_ENABLED`, `PERSONALIZATION_ENABLED`

**Success Criteria**:
- SC-008: Feature flag toggling without core functionality impact

### Phase 2D – Localization (Bonus - Feature Flagged)
**Objective**: Add Urdu translation support for content

**Components**:
- Urdu translation toggle with accessible UI
- Cached translations for performance
- Translation service integration
- Bilingual content delivery (English/Urdu)

**Feature Flag**: `URDU_TRANSLATION_ENABLED`

**Scope Note**: Urdu applies to chapter content, not chatbot responses (as per clarifications)

## Risk Analysis & Mitigation

### Identified Risks

#### R1: Hallucinations if context filtering fails
- **Risk**: Chatbot may generate responses based on general LLM knowledge instead of book content only
- **Impact**: Violates core requirement of strict content adherence, undermines trust
- **Probability**: Medium (if context filtering logic has bugs)
- **Mitigation**: Hard context window enforcement with strict validation before response generation
- **Implementation**: Content grounding enforcement in `backend/src/services/rag.py` with validation checks

#### R2: UI distraction during reading
- **Risk**: Chatbot UI may distract from primary reading experience
- **Impact**: Reduced user engagement with book content
- **Probability**: Medium (if UI is intrusive or always visible)
- **Mitigation**: Collapsible chatbot UI that stays out of reading flow
- **Implementation**: Collapsible component in `frontend/src/components/Chatbot/` with floating toggle button

#### R3: Deployment failures from missing env vars
- **Risk**: Application fails to deploy due to missing environment variables
- **Impact**: Complete service outage until vars are properly configured
- **Probability**: Low to Medium (with proper CI/CD checks)
- **Mitigation**: Feature flags & fallbacks for graceful degradation
- **Implementation**: Environment validation in `backend/src/config/settings.py` with safe fallbacks

#### R4: Performance lag with large documents
- **Risk**: Slow response times when processing large document collections
- **Impact**: Poor user experience with delayed responses
- **Probability**: Medium (with growing document base)
- **Mitigation**: Chunk-level metadata validation and optimized search
- **Implementation**: Efficient chunking strategy in `backend/src/services/embedding.py` with metadata validation

### Tradeoffs

#### T1: Accuracy over creativity
- **Decision**: Prioritize factual accuracy over creative or imaginative responses
- **Rationale**: Core requirement is to answer from book content only, not generate creative content
- **Impact**: More limited, but more reliable responses

#### T2: Stability over feature density
- **Decision**: Prioritize stable, reliable functionality over numerous features
- **Rationale**: Core chatbot functionality must be rock-solid before adding advanced features
- **Impact**: Slower feature rollout but more reliable user experience

#### T3: Readability over heavy animations
- **Decision**: Prioritize clean, readable UI over heavy animations
- **Rationale**: Maintain focus on reading experience while adding subtle enhancements
- **Impact**: More subtle UI effects that don't distract from content

### Performance Targets

#### P1: Response time < 2 seconds
- **Target**: All chat responses delivered within 2 seconds
- **Current Goal**: <5 seconds (from success criteria), with optimization to <2 seconds
- **Implementation**: Optimized vector search in Qdrant and efficient RAG pipeline

#### P2: UI load impact minimal
- **Target**: <5% impact on page load times (improved from <10% requirement)
- **Current Goal**: <10% (from constraints), with optimization to <5%
- **Implementation**: Lazy loading and efficient component design in frontend

#### P3: No blocking render paths
- **Target**: Chatbot loading should never block main content rendering
- **Current Goal**: Non-blocking integration (from constraints)
- **Implementation**: Async loading and error boundaries in Docusaurus plugin

### Risk Control Measures

- **Feature Flags**: Every bonus feature (Phases 2B, 2C, 2D) is controlled by feature flags to enable/disable without code changes, ensuring core functionality remains intact
- **Graceful Degradation**: If RAG backend is unavailable, frontend continues to function with appropriate user messaging (as per clarifications)
- **Content Isolation**: RAG responses strictly limited to book content only (no general LLM knowledge usage)
- **Non-Breaking Integration**: All changes are backward-compatible with existing Docusaurus functionality (as per constitution)
- **Separate Deployments**: Backend (FastAPI) and frontend (Docusaurus) can be deployed independently
- **Context Enforcement**: Selected-text-only mode completely overrides global context (no leakage)
- **Error Handling**: Proper error boundaries and fallback UI to maintain reading experience
- **Hard Context Window**: Strict validation of content sources before response generation
- **Chunk-Level Validation**: Metadata validation at chunk level to ensure quality and relevance

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Multiple service architecture | Clean separation of concerns required by constitution | Single monolithic service would violate backend/frontend separation principle |
| Vector database dependency | Essential for RAG functionality | Simple keyword search insufficient for semantic understanding |