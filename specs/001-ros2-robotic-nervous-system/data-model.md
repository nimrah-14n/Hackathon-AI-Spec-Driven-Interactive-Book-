# Data Model: ROS 2 Educational Module - The Robotic Nervous System

**Created**: 2025-12-15
**Feature**: 001-ros2-robotic-nervous-system
**Status**: Complete

## Entity Models

### User
- **id**: UUID (Primary Key)
- **email**: String (Unique, Required)
- **name**: String (Required)
- **software_background**: String (Enum: 'beginner', 'intermediate', 'advanced', 'none')
- **hardware_background**: String (Enum: 'beginner', 'intermediate', 'advanced', 'none')
- **created_at**: DateTime (Auto-generated)
- **updated_at**: DateTime (Auto-generated)
- **last_login**: DateTime
- **is_authenticated**: Boolean (Default: false)

**Relationships**:
- One-to-many with UserProgress
- One-to-many with UserInteraction

**Validation Rules**:
- Email must be valid email format
- Name must be 2-50 characters
- Background fields must be one of the defined enum values

### Chapter
- **id**: UUID (Primary Key)
- **module_id**: String (e.g., '001-ros2-robotic-nervous-system')
- **chapter_number**: Integer (1-9 for this module)
- **title**: String (Required)
- **slug**: String (URL-friendly, Unique per module)
- **content_en**: Text (Required, English content)
- **content_ur**: Text (Urdu translation, Optional)
- **created_at**: DateTime (Auto-generated)
- **updated_at**: DateTime (Auto-generated)

**Relationships**:
- One-to-many with ContentChunk (for RAG indexing)
- One-to-many with UserProgress

**Validation Rules**:
- Chapter number must be within module range (1-9)
- Title must be 5-200 characters
- Content must be provided in at least one language

### ContentChunk
- **id**: UUID (Primary Key)
- **chapter_id**: UUID (Foreign Key to Chapter)
- **chunk_number**: Integer
- **content_en**: Text (Required)
- **content_ur**: Text (Optional)
- **embedding_vector**: Vector (1536 dimensions for OpenAI ada-002)
- **token_count**: Integer
- **created_at**: DateTime (Auto-generated)

**Relationships**:
- Many-to-one with Chapter
- Used for RAG system indexing

**Validation Rules**:
- Token count must be less than 8000 for OpenAI context window
- Embedding vector must be properly formatted

### UserProgress
- **id**: UUID (Primary Key)
- **user_id**: UUID (Foreign Key to User)
- **chapter_id**: UUID (Foreign Key to Chapter)
- **completion_percentage**: Integer (0-100)
- **time_spent_seconds**: Integer
- **last_accessed**: DateTime
- **bookmarks**: JSON (Array of content positions)
- **notes**: JSON (Array of user notes)
- **created_at**: DateTime (Auto-generated)
- **updated_at**: DateTime (Auto-generated)

**Relationships**:
- Many-to-one with User
- Many-to-one with Chapter

**Validation Rules**:
- Completion percentage must be between 0-100
- Time spent must be non-negative

### UserInteraction
- **id**: UUID (Primary Key)
- **user_id**: UUID (Foreign Key to User, Optional for anonymous)
- **interaction_type**: String (Enum: 'rag_query', 'subagent_request', 'chapter_view', 'personalization_adjust', 'translation_toggle')
- **content_id**: UUID (Optional, references Chapter or ContentChunk)
- **query_text**: Text (For RAG queries)
- **response_text**: Text (For RAG responses)
- **interaction_data**: JSON (Additional context data)
- **created_at**: DateTime (Auto-generated)

**Relationships**:
- Many-to-one with User (Optional)

**Validation Rules**:
- Interaction type must be one of the defined enum values
- Query/response text must be less than 10,000 characters

### PersonalizationProfile
- **id**: UUID (Primary Key)
- **user_id**: UUID (Foreign Key to User)
- **content_complexity_preference**: String (Enum: 'simplified', 'balanced', 'advanced')
- **example_preference**: String (Enum: 'theoretical', 'practical', 'balanced')
- **emphasis_preference**: String (Enum: 'software', 'hardware', 'balanced')
- **last_updated**: DateTime (Auto-generated)

**Relationships**:
- One-to-one with User

**Validation Rules**:
- All preference fields must be one of the defined enum values

### RAGSession
- **id**: UUID (Primary Key)
- **user_id**: UUID (Foreign Key to User, Optional for anonymous)
- **session_token**: String (Unique session identifier)
- **query_history**: JSON (Array of query-response pairs)
- **active_chunks**: Array of UUIDs (Referencing ContentChunk)
- **created_at**: DateTime (Auto-generated)
- **expires_at**: DateTime (Auto-generated, 1 hour from creation)

**Relationships**:
- Many-to-one with User (Optional)
- Many-to-many with ContentChunk

**Validation Rules**:
- Session must expire within 24 hours
- Query history should be limited to last 50 interactions

## State Transitions

### User Authentication State
- `anonymous` → `authenticated` (via login/signup)
- `authenticated` → `session_expired` (via timeout)
- `session_expired` → `authenticated` (via re-authentication)

### Chapter Completion State
- `not_started` → `in_progress` (first access)
- `in_progress` → `completed` (when completion_percentage reaches 100)
- `completed` → `reviewed` (user revisits after completion)

## Indexes

### Required Database Indexes
- User.email (Unique)
- Chapter.module_id + Chapter.chapter_number (Composite)
- Chapter.slug (Unique per module)
- ContentChunk.chapter_id + ContentChunk.chunk_number (Composite)
- UserProgress.user_id + UserProgress.chapter_id (Composite, Unique)
- UserInteraction.created_at (For time-based queries)
- ContentChunk.embedding_vector (Vector index for similarity search)