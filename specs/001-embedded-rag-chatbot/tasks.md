---
description: "Task list for Embedded RAG Chatbot implementation"
---

# Tasks: Embedded RAG Chatbot for Spec-Driven AI Book

**Input**: Design documents from `/specs/001-embedded-rag-chatbot/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume web app structure based on plan.md

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create backend project structure per implementation plan
- [ ] T002 Create frontend project structure per implementation plan
- [ ] T003 [P] Initialize FastAPI project using uv in backend/
- [ ] T004 [P] Configure environment variables management in backend/
- [ ] T005 [P] Setup Qdrant Cloud connection configuration
- [ ] T006 [P] Setup Neon Postgres connection configuration

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T007 Setup database schema and migrations framework for Neon Postgres
- [ ] T008 [P] Implement document ingestion pipeline in backend/src/services/ingestion.py
- [ ] T009 [P] Implement embedding generation service in backend/src/services/embedding.py
- [ ] T010 [P] Setup Qdrant vector storage service in backend/src/services/vector_store.py
- [ ] T011 Create base data models in backend/src/models/
- [ ] T012 Configure error handling and logging infrastructure in backend/
- [ ] T013 Setup basic API routing structure in backend/src/api/
- [ ] T014 Create frontend chatbot component structure in frontend/src/components/

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Basic Chatbot Interaction (Priority: P1) 🎯 MVP

**Goal**: Enable users to ask questions about book content and receive accurate answers from the chatbot embedded in Docusaurus pages

**Independent Test**: User can type a question in the embedded chatbot, submit it, and receive a relevant answer based on book content within 5 seconds

### Implementation for User Story 1

- [ ] T015 [P] [US1] Create Document model in backend/src/models/document.py
- [ ] T016 [P] [US1] Create Chat model in backend/src/models/chat.py
- [ ] T017 [US1] Implement document ingestion pipeline in backend/src/services/ingestion.py
- [ ] T018 [US1] Implement embedding generation in backend/src/services/embedding.py
- [ ] T019 [US1] Implement RAG orchestration service in backend/src/services/rag.py
- [ ] T020 [US1] Create chat API endpoint in backend/src/api/v1/chat.py
- [ ] T021 [US1] Implement content grounding and context filtering in backend/src/services/rag.py
- [ ] T022 [US1] Add refusal responses for out-of-context questions in backend/src/services/rag.py
- [ ] T023 [US1] Create basic chatbot React component in frontend/src/components/Chatbot/
- [ ] T024 [US1] Add floating toggle button to chatbot component
- [ ] T025 [US1] Implement pastel color theme for chatbot UI
- [ ] T026 [US1] Add API integration for chatbot in frontend/src/services/chatService.js
- [ ] T027 [US1] Integrate chatbot with backend API
- [ ] T028 [US1] Add smooth animations to chatbot UI
- [ ] T029 [US1] Test basic chat functionality with book content

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Context-Specific Answering (Priority: P2)

**Goal**: Enable users to select specific text on a page and ask questions about only that selected text

**Independent Test**: User can select text on a page, ask a question about it, and verify that the chatbot responds based only on the selected text rather than the entire book

### Implementation for User Story 2

- [ ] T030 [P] [US2] Enhance chat model to support selected-text context in backend/src/models/chat.py
- [ ] T031 [US2] Add selected-text-only answering mode in backend/src/services/rag.py
- [ ] T032 [US2] Implement selected-text override functionality in backend/src/services/rag.py
- [ ] T033 [US2] Update chat API endpoint to support selected-text mode in backend/src/api/v1/chat.py
- [ ] T034 [US2] Create text selection capture functionality in frontend/src/hooks/useTextSelection.js
- [ ] T035 [US2] Enhance chatbot component to handle selected text in frontend/src/components/Chatbot/
- [ ] T036 [US2] Add UI controls for selected-text mode in chatbot
- [ ] T037 [US2] Test selected-text-only answering functionality
- [ ] T038 [US2] Verify complete context isolation from book-wide context

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Claude Code Subagents (Priority: P3 - Bonus)

**Goal**: Implement Claude Code Subagents for advanced reasoning and Agent Skills per chapter

**Independent Test**: User can access enhanced intelligence features through subagents and chapter-specific skills

### Implementation for User Story 3

- [ ] T039 [P] [US3] Create Subagent model in backend/src/models/subagent.py
- [ ] T040 [US3] Implement Claude Code Subagents service in backend/src/services/subagents.py
- [ ] T041 [US3] Create Agent Skills per chapter functionality in backend/src/services/agent_skills.py
- [ ] T042 [US3] Add Subagent API endpoints in backend/src/api/v1/subagents.py
- [ ] T043 [US3] Implement feature flag management for subagents in backend/src/config/feature_flags.py
- [ ] T044 [US3] Enhance chatbot UI with subagent controls in frontend/src/components/
- [ ] T045 [US3] Add feature flag integration for subagents in frontend/src/hooks/useFeatureFlags.js
- [ ] T046 [US3] Test subagent functionality integration

**Checkpoint**: At this point, User Stories 1, 2 AND 3 should all work independently

---

## Phase 6: User Story 4 - Auth & Personalization (Priority: P4 - Bonus)

**Goal**: Add user authentication and personalized chapter variants

**Independent Test**: User can authenticate and have their preferences stored for personalized experience

### Implementation for User Story 4

- [ ] T047 [P] [US4] Create User model in backend/src/models/user.py
- [ ] T048 [US4] Integrate better-auth in backend/src/services/auth.py
- [ ] T049 [US4] Implement user background collection in backend/src/services/user.py
- [ ] T050 [US4] Create user preferences storage in backend/src/services/user.py
- [ ] T051 [US4] Add personalized chapter variants logic in backend/src/services/personalization.py
- [ ] T052 [US4] Create auth API endpoints in backend/src/api/v1/auth.py
- [ ] T053 [US4] Implement feature flag management for auth in backend/src/config/feature_flags.py
- [ ] T054 [US4] Create auth UI components in frontend/src/components/Auth/
- [ ] T055 [US4] Add personalization controls to frontend
- [ ] T056 [US4] Test authentication and personalization features

**Checkpoint**: All user stories should now be independently functional

---

## Phase 7: User Story 5 - Localization (Priority: P5 - Bonus)

**Goal**: Add Urdu translation support for chapter content

**Independent Test**: User can toggle Urdu translation for chapter content with cached translations

### Implementation for User Story 5

- [ ] T057 [P] [US5] Create Translation model in backend/src/models/translation.py
- [ ] T058 [US5] Implement Urdu translation service in backend/src/services/translation.py
- [ ] T059 [US5] Add translation caching mechanism in backend/src/services/translation.py
- [ ] T060 [US5] Create translation API endpoints in backend/src/api/v1/translation.py
- [ ] T061 [US5] Implement feature flag management for translations in backend/src/config/feature_flags.py
- [ ] T062 [US5] Create translation toggle UI in frontend/src/components/Translation/
- [ ] T063 [US5] Add translation caching in frontend
- [ ] T064 [US5] Test Urdu translation functionality (content only, not responses)

**Checkpoint**: All localization features should be functional with existing functionality preserved

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T065 [P] Documentation updates in docs/
- [ ] T066 Code cleanup and refactoring across all components
- [ ] T067 Performance optimization for vector search and API responses
- [ ] T068 [P] Additional unit tests in backend/tests/unit/ and frontend/tests/
- [ ] T069 Security hardening and validation
- [ ] T070 Run quickstart.md validation
- [ ] T071 Final integration testing
- [ ] T072 Environment variable configuration for safe deployment
- [ ] T073 Docusaurus plugin integration for chatbot
- [ ] T074 Quality assurance and user acceptance testing

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3 → P4 → P5)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable
- **User Story 4 (P4)**: Can start after Foundational (Phase 2) - May integrate with previous stories but should be independently testable
- **User Story 5 (P5)**: Can start after Foundational (Phase 2) - May integrate with previous stories but should be independently testable

### Within Each User Story

- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Add User Story 4 → Test independently → Deploy/Demo
6. Add User Story 5 → Test independently → Deploy/Demo
7. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
   - Developer D: User Story 4
   - Developer E: User Story 5
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence