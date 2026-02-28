# Designing an RLM Codebase Analysis

A guide for designing runbooks that use recursive language model analysis to investigate a codebase with a specific goal.

## Core Principle

A runbook is an interview, not a checklist. Each question builds on accumulated conversation history. The sequence controls what the agent knows at each step, and what it knows determines the quality of later answers.

## The Design Process

Designing a runbook is itself a collaborative process between a user and an agent. This process has failure modes that must be understood before starting.

### Step 1: Define the Goal

State the specific outcome you want. "Learn how this codebase works" is too broad. "Produce a validated integration guide for adding a new vector store backend" is actionable and verifiable.

The goal determines what the final questions must produce. Work backward from there.

### Step 2: Recon the Codebase

Before designing questions, explore the target codebase to understand what's actually there. This is necessary because you cannot design an effective interview about something you haven't seen.

However, treat recon findings as suspect. A quick exploration of a large codebase is inherently lossy. You will miss things, misinterpret structure, and make incorrect assumptions about how components relate.

What recon gives you: better vocabulary for the domains and subject areas your questions should cover. It tells you what kinds of things to ask about.

What recon does not give you: confirmed facts to build questions around. Do not design questions that seek to confirm what you think you found.

### Step 3: Watch for Agent Over-Scoping

This is the critical failure mode in collaborative runbook design.

When an agent performs recon and gets initial findings, it will naturally want to scope questions tightly to those findings. It will propose questions like "Show me the VectorStoreFactory in utils/factory.py" instead of "How does the system resolve and instantiate a backend from configuration?"

The first question assumes the recon is correct and seeks confirmation. The second question investigates the domain and lets the RLM discover what's actually there. If the factory is in a different file, or if there is no factory, the second question still works. The first question fails silently by anchoring the RLM to a possibly-wrong assumption.

The user's role is to push questions toward domains, not specifics. The agent's recon identifies which domains matter. The questions should be open investigations into those domains.

### Step 4: Design the Question Sequence

Questions should follow a progression that builds understanding incrementally:

**Orient** - What is this project? How is it organized? What extension points exist? These questions establish vocabulary and context. Without them, later questions lack grounding and the RLM may hallucinate structural assumptions.

**Identify contracts** - What interfaces, base abstractions, or protocols define the extension point? What data crosses the boundary? These questions name the things that subsequent questions will refer to.

**Trace wiring** - How does configuration reach instantiation? What's the resolution chain? These questions reveal the mechanical steps an integration must follow. They depend on knowing the contracts from the prior phase.

**Study implementations** - How does an existing implementation work end-to-end? What does it handle that the base abstraction doesn't account for? These questions surface implicit requirements that aren't visible in interfaces alone.

**Examine operational concerns** - Error handling, lifecycle, cleanup, connection management. These questions probe the areas most likely to be undocumented and most likely to cause integration failures.

**Study test patterns** - How are existing implementations tested? Are there shared patterns or is each implementation independent? These questions define what "done" looks like for a new integration.

**Synthesize** - Produce the deliverable (guide, checklist, specification). By this point the RLM has enough accumulated context to produce a grounded answer rather than a generic template.

**Validate** - Evaluate the deliverable against an existing implementation. This is the judge step. Walk the deliverable through a real integration and flag every gap. Without this step, you cannot distinguish between a grounded deliverable and a plausible-sounding hallucination.

**Revise** - Feed the validation findings back into the deliverable. The agent must internalize what it missed and produce a corrected output. Without this step, the validation is just a list of problems with no resolution. The gap findings are the most valuable output of the entire runbook - they represent ground truth about what the agent's understanding missed. Stopping before revision wastes them.

### Step 5: Close the Feedback Loop

The validation phase will produce specific, concrete findings: steps the guide got wrong, steps that are missing, patterns the guide prescribes that the codebase doesn't actually follow. These findings are not the end of the workflow - they are input to a revision step.

The revision question must explicitly ask the agent to incorporate every finding from validation. The agent has the gap list in its conversation history and the accumulated context from all prior phases. This is the moment where that context is most valuable - the agent knows what it got wrong and has enough grounding to fix it.

A runbook that validates but does not revise produces a report with known defects. A runbook that revises after validation produces a corrected deliverable.

## Technical Considerations

### Snapshot Scoping

Large codebases exceed snapshot size limits. Use INCLUDE_GLOBS to focus on the relevant source - the core library and tests, not docs, examples, or build artifacts. Scope aggressively; the RLM only needs the code that's relevant to the investigation.

### Retry Resilience

A runbook with 10+ questions may run for an extended period. Transient API failures (overloaded, rate limit) will happen. The runner must retry with exponential backoff and jitter rather than failing the entire sequence on a single transient error.

### Context Accumulation

Each question's answer becomes part of the conversation history for subsequent questions. This is the mechanism that makes sequential questions more powerful than independent ones. But it also means a bad answer early in the sequence can poison later questions. The orientation phase exists partly to establish a reliable foundation that later questions can build on.

### Lossy Analysis

Accept that any automated analysis of a large codebase will be incomplete. Design questions that surface what matters rather than trying to achieve exhaustive coverage. The validation phase exists to catch what the analysis missed.

## Antipatterns

**Asking the synthesis question first.** Without accumulated context, the RLM produces a generic answer based on general knowledge rather than one grounded in the actual codebase.

**Designing questions around expected answers.** Questions should investigate domains, not confirm assumptions. If your question only works when the answer matches what you expect, it's a confirmation question, not an investigation question.

**Skipping validation.** Without evaluating the output against a real implementation, you cannot assess whether the runbook produced useful results or plausible hallucinations.

**Stopping at validation.** Validation without revision produces a report with known defects. The gap findings are the most actionable output of the entire workflow - they must feed back into a revised deliverable.

**Treating recon as ground truth.** Recon is lossy. Use it to identify what domains to investigate, not what answers to expect.

**Over-scoping after recon.** The natural tendency after seeing initial findings is to tighten questions around those findings. Resist this. Generalize toward domains, let the RLM discover specifics.
