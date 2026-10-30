# Architecture decision records

## Purpose
Record runtime/tuning decisions and preserve the provenance of retired quality decisions.

## Ownership
Repository developers; parent docs/AGENTS.md.

## Local Contracts
- New records use monotonic four-digit IDs, Status, Date, Context, Decision and Consequences. Never reuse archived IDs; the archive index accounts for gaps.
- Accepted decisions are immutable. A later decision supersedes them; link maintenance does not change their meaning. Retired originals remain byte-preserved in the linked snapshot.

## Work Guidance
- Current distribution ownership is ADR 0020. Older runtime records are historical context, not permission to revive retired interfaces.

## Verification
- Every live record has the required sections; relative file links resolve. Retired IDs resolve through the archive index.

## Child DOX Index
- [0001-deepen-llama-server-runner](0001-deepen-llama-server-runner.md)
- [0005-config-py-mutable-baseline](0005-config-py-mutable-baseline.md)
- [0010-cross-platform-zombie-process-prevention](0010-cross-platform-zombie-process-prevention.md)
- [0011-dashboard-ailocal-design](0011-dashboard-ailocal-design.md)
- [0015-rocm-first-binary-resolution-windows](0015-rocm-first-binary-resolution-windows.md)
- [0016-measurement-hygiene-and-morris-screen](0016-measurement-hygiene-and-morris-screen.md)
- [0018-independent-pi-benchmark-and-context-tuning](0018-independent-pi-benchmark-and-context-tuning.md)
- [0019-tuning-only-release](0019-tuning-only-release.md)
- [0020-tuning-only-distribution](0020-tuning-only-distribution.md)
- [Retired quality records](../retired-quality.md) — immutable source links.
