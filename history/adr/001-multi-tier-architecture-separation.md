# ADR 001: Multi-Tier Architecture with Clear Separation of Concerns

## Status
Accepted | Superseded by [ADR-NNN if applicable] | Deprecated by [ADR-NNN if applicable]

## Context
The AI/Spec-Driven Interactive Book with Integrated RAG Chatbot project requires a robust, maintainable architecture that can handle multiple complex requirements:
- Educational content delivery with personalization
- RAG-based question answering system
- Multi-language support (English/Urdu)
- User authentication and progress tracking
- Integration with AI services (OpenAI, Claude subagents)
- Scalability for multiple modules and users

Early analysis revealed the need to separate different concerns to maintain system clarity and follow the project constitution's principle of "Clear separation of frontend, backend, AI, and data layers."

## Decision
We will implement a multi-tier architecture with the following clear separations:

### Frontend Tier (Docusaurus/React)
- Static site generation for educational content
- Interactive components for RAG chatbot, subagents, and personalization
- Client-side user experience and content rendering
- Deployment to GitHub Pages for cost-effective hosting

### Backend Tier (FastAPI)
- RESTful API endpoints for all dynamic functionality
- Business logic for authentication, authorization, and content management
- Integration layer for AI services and external APIs
- User session and progress management

### Data Tier
- Neon Serverless Postgres for relational data (users, progress, content metadata)
- Qdrant Cloud for vector storage (RAG content embeddings)
- Separate storage for multilingual content

### AI Integration Tier
- OpenAI API for language model capabilities
- Specialized subagents for domain-specific assistance
- Content indexing and retrieval services

## Alternatives Considered

### Monolithic Architecture
- **Pros**: Simpler deployment, fewer network calls
- **Cons**: Violates constitution principle of clear separation, harder to scale components independently, mixed concerns would make maintenance difficult

### Server-Side Rendered Application
- **Pros**: Better initial load performance, SEO benefits
- **Cons**: More complex authentication flow, higher server costs, less suitable for static educational content

### Single Database Approach
- **Pros**: Simpler data management
- **Cons**: Would not efficiently serve both relational needs (user data) and vector search requirements (RAG system)

## Consequences

### Positive
- Clear separation aligns with project constitution principles
- Each tier can be developed, tested, and scaled independently
- Technology choices can be optimized per tier (static hosting for frontend, specialized DBs for data)
- Better security through isolation of concerns
- Easier team collaboration with well-defined interfaces
- Supports the RAG integrity principle through dedicated vector storage

### Negative
- More complex initial setup with multiple services
- Network latency between tiers
- More infrastructure components to monitor
- Potentially higher costs due to multiple specialized services

## Implementation
- Frontend: Docusaurus static site with React components for interactivity
- Backend: FastAPI with Pydantic models and async support
- Databases: Neon Postgres for relational data, Qdrant for vector storage
- APIs: RESTful design with OpenAPI specification
- Authentication: Better Auth for secure user management

## Notes
This decision supports the project's long-term maintainability and aligns with the "Modularity and Maintainability" principle in the constitution. The clear API contracts between tiers will facilitate future enhancements and make the system more testable.