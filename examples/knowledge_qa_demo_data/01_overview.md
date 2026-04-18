# Atlas Assistant Overview

## Product Summary

Atlas Assistant is an internal knowledge QA product for engineering and support
teams. It answers questions about platform architecture, deployment runbooks,
release notes, and operator procedures.

The first production use case is reducing time spent searching team
documentation during incidents and release reviews. Atlas Assistant is not
designed to answer broad internet questions. It only answers from indexed
internal knowledge.

## Goals

- Provide grounded answers with citations.
- Refuse questions when the indexed evidence is insufficient.
- Make retrieval behavior observable through structured traces.
- Support both local development and hosted model endpoints.

## Non-Goals

- Atlas Assistant does not execute production changes.
- Atlas Assistant does not browse the public internet at answer time.
- Atlas Assistant does not infer undocumented business commitments.

## Supported Content

The initial knowledge base contains four document groups:

- architecture notes
- deployment runbooks
- release notes
- operator FAQs

## Demo Notes

For the demo environment, the public API base URL is
`https://atlas-demo.internal/api`.

The support team asks that answers remain concise and include citations when
possible.
