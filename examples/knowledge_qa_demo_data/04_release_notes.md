# Atlas Assistant Release Notes

## Version 0.3.0

Version 0.3.0 introduced citation-aware answer formatting. The response layer
now returns structured fields for the answer body, citation references, and
answer status.

This release also added JSON trace logging for each knowledge QA request so the
team could inspect retrieved chunks and final prompts during debugging.

## Version 0.2.0

Version 0.2.0 added the first hybrid retrieval flow based on Qdrant plus
embedding search.

## Version 0.1.0

Version 0.1.0 shipped the initial internal demo with single-document retrieval
and no citation rendering.
