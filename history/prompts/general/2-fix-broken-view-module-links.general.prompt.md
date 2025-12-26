---
id: 2
title: "Fix Broken View Module Links"
stage: "general"
date_iso: "2025-12-20"
surface: "agent"
model: "Claude Sonnet 4.5"
feature: "none"
branch: "main"
user: "user"
command: "sp.implement"
labels: ["docusaurus", "navigation", "broken-links", "fix"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files_yaml: |
  - docusaurus.config.js
  - src/pages/index.tsx
tests_yaml: |
  - manual testing
---

## Prompt Text
```
Fix broken "View Module" links in Docusaurus project.

Current Problem:
- Buttons labeled "View Module" lead to pages showing "Page Not Found".
- Likely cause: incorrect paths in links, sidebar, or markdown files.

Task:
1. Check all buttons, links, and references pointing to module pages (module1-ros2, module2-digital-twin, module3-ai-brain, module4-vla).
2. Ensure each link matches the exact path of the corresponding markdown file in /docs.
3. Update sidebar.js or any navbar/footer links if necessary.
4. Keep all other content unchanged.
5. Test locally: run `npm run start` to ensure all module links open correctly.

Goal:
- All "View Module" buttons correctly navigate to their respective module pages.
- Deployment-ready: no broken links.
```

## Response Text
Successfully fixed all broken "View Module" links in the Docusaurus project. Updated navigation links in docusaurus.config.js to point to actual existing files instead of non-existent "intro" files. Fixed the "View Modules" button in index.tsx to point to the correct documentation entry point. All module links now correctly navigate to their respective pages without showing "Page Not Found" errors.

## Outcome
All "View Module" buttons and links now work correctly, directing users to the appropriate module documentation pages without encountering broken links.

## Evaluation
The implementation successfully addressed the requirements:
- [X] Checked all buttons, links, and references pointing to module pages
- [X] Ensured each link matches the exact path of the corresponding markdown file
- [X] Updated navigation links in docusaurus.config.js and index.tsx
- [X] Kept all other content unchanged
- [X] Tested locally to ensure all module links open correctly
- [X] Created deployment-ready site with no broken links