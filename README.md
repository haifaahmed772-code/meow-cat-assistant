# Meow Clinic — Cat Care Assistant

**➡️ The whole project is in [`capstone.ipynb`](capstone.ipynb) — one notebook, run top to bottom, with all outputs saved.**

An agentic assistant for the clients of a cat clinic. Every question is routed
by a language model to the knowledge source that can answer it, calculations
are done by tools rather than guessed, each cat is remembered across separate
conversations, and nothing gets booked without a staff member approving it.

---

## Team and programme

| | |
|---|---|
| Trainees | Haifa Ahmed Alsulami · Wed Ayad Alshalawi |
| Programme | **Building Agentic AI Systems** |
| Cohort | 9 – 13 August 2026 |
| Institution | SDAIA Academy — [website](https://sdaia.gov.sa/en/Sectors/academy/Pages/default.aspx) · [GitHub](https://github.com/SDAIAAcademy) |
| Declared track | **Track C — router across multiple knowledge sources** |
| Architecture | Multi-agent — one supervisor, two specialist sub-agents |

---

## The idea

Questions from cat owners fall into two kinds that have almost nothing in
common:

- *"My cat is scratching the sofa, what do I do?"* — care and behaviour
- *"How much is a check-up and do I need to book?"* — clinic services

Answering both from one pile of text means every question searches material
that cannot possibly help it. So the assistant keeps two separate vector
stores and picks between them per question:

| Store | Contents | Source file |
|---|---|---|
| `care_guide` | feeding, water, litter, behaviour, grooming, breeds, warning signs | [`data/care_guide.md`](data/care_guide.md) |
| `clinic_services` | prices, hours, booking, cancellation, vaccination schedule, boarding | [`data/clinic_services.md`](data/clinic_services.md) |

The two documents were written so neither can answer the other's questions.
That is what makes the routing decision load-bearing: sending a question to the
wrong store produces a visibly wrong answer instead of being quietly rescued.

Both documents describe a fictional clinic and were written for this project.

---

## What it does

**Routes with a model, not with keywords.** A supervisor classifies each
question as `care` / `clinic` / `both` and hands it to a specialist agent. The
care agent holds only the care-guide search tool and the clinic agent only the
services search tool, so the separation is structural rather than a filter
applied after the fact. The test case that matters is *"my cat has been
lethargic for a day after her vaccination — is that normal?"*: it contains a
clinic word, but it is a care question, and keyword matching sends it to the
wrong specialist.

**Calculates instead of guessing.** Five tools the model calls on its own,
including daily energy requirement from `RER = 70 × weight^0.75` and a boarding
quote that applies the second-cat rate, extras and the five-night discount in
order.

**Remembers two different things.** A checkpointer holds one conversation; a
Store holds facts about each cat and is not attached to any conversation. A cat
registered in one thread is usable from a thread that shares no history with it.

**Stops before it commits anything.** Booking pauses on `interrupt()` and waits
for a staff decision, which can be an approval or a refusal with a reason.

**Checks its own answers.** An evaluator reads each reply against the retrieved
text and sends back anything it cannot support, which is how the assistant
stopped claiming the clinic offers a grooming appointment it had never looked up.

---

## Architecture

Single supervisor, two specialist sub-agents, two isolated knowledge stores.

```
                      question
                          |
                 [ supervisor / classifier ]
                  with_structured_output -> care | clinic | both
                          |
              +-----------+-----------+
              |                       |
        [ care_agent ]          [ clinic_agent ]
              |                       |
     search_care_guide        search_clinic_services
     daily_calorie_needs      boarding_cost
     cat_age_in_human_years   next_vaccination_due
     today                    today
              |                       |
        Chroma: care_guide      Chroma: clinic_services
              |                       |
              +-----------+-----------+
                          |
                   [ answer_task ]
                          |
                  [ review_answer ]  <--+  evaluator-optimizer
                          |             |  (max 3 rounds)
                    grounded? no -> [ revise_answer ]
                          |
                       grounded? yes
                          |
                  booking requested?
                          |
                   [ interrupt() ]  --- waits for staff
                          |
                  Command(resume=...)
                          |
                   approved -> confirm_booking
                   refused  -> no booking, reason recorded
```

### Components

| Module | What it does |
|---|---|
| `classify` / `Route` | Supervisor. Returns a constrained `care`/`clinic`/`both` plus its reason |
| `care_agent` | Sub-agent holding only the care-guide search tool |
| `clinic_agent` | Sub-agent holding only the services search tool |
| `daily_calorie_needs` | `RER = 70 × weight^0.75` × life-stage factor |
| `boarding_cost` | Nightly rate, second-cat rate, extras, 5-night discount |
| `cat_age_in_human_years`, `next_vaccination_due`, `today` | Date arithmetic |
| `triage_task` / `Triage` | Decides whether a described symptom needs a vet |
| `review_answer` / `Grounding` | Evaluator. Rejects claims absent from the retrieved context |
| `assistant` | `@entrypoint` tying it together, with `interrupt()` before booking |
| `InMemorySaver` | Short-term state, per `thread_id` |
| `InMemoryStore` | Long-term cat profiles, independent of any thread |

### State and memory

- **Short-term** — checkpointer keyed by `thread_id`, holds one conversation.
- **Long-term** — `InMemoryStore` under `("cats", owner_id)`, readable from any
  thread. A cat registered in one conversation is usable from another that
  shares no history with it.

### Error handling

| Strategy | Where |
|---|---|
| `RetryPolicy(max_attempts=3)` | on every `@task` that calls the model |
| Fallback to searching both stores | when the classifier raises |
| Escalation to a vet | when triage returns `needs_vet: yes` |
| Model fallback | `with_fallbacks` across two Groq models on a token-limit error |

---

## Installation and how to run

### Prerequisites

- A Google account, for Google Colab. No local install is needed — the notebook
  installs its own dependencies in the first cell.
- A free Groq API key — <https://console.groq.com>
- A free LangSmith API key — <https://smith.langchain.com>
- Running locally instead of Colab needs Python 3.10+ and roughly 2 GB free for
  the embedding model.

### Setup

1. Open [`capstone.ipynb`](capstone.ipynb) in Google Colab.
2. Open the secrets panel (🔑 in the left sidebar) and add both keys, each with
   **Notebook access** switched on:

   | Secret | Value |
   |---|---|
   | `GROQ_API_KEY` | your Groq key |
   | `LANGCHAIN_API_KEY` | your LangSmith key |

3. `Runtime → Restart session and run all`.

No key is written into the notebook, so none reaches this repository.

### Configuration

Set in the first cell, before `langchain` is imported:

| Variable | Value | Why |
|---|---|---|
| `LANGCHAIN_TRACING_V2` | `true` | Turns tracing on. `LANGSMITH_TRACING_V2` is not a real variable and fails silently |
| `LANGCHAIN_PROJECT` | `meow-clinic-capstone` | Where traces are collected |
| `HF_HUB_DISABLE_PROGRESS_BARS` | `1` | Keeps download bars out of the saved file |

Ordering matters: tracing must be enabled before `langchain` is imported, or the
tracer is built without it and nothing is ever sent.

### Expected output

The full run takes about 10–15 minutes, most of it the one-time embedding model
download. On success you should see:

- `13 chunks` per source, and `13 vectors` in each of the two Chroma stores
- `7/8 answers present in the retrieved context` — the eighth is a documented
  ambiguity, explained under Known limitation below
- routing decisions printed as `care`, `clinic` and `both`, each naming the
  agent that received the question and the tools it called
- `attempt 1 / 2 / 3` from the retry demo, and `total attempts: 3`
- `PAUSED — waiting for staff`, then `BOOKED —` after `Command(resume="approve")`
- `round 1: grounded=no` followed by `round 2: grounded=yes`
- a table of LangSmith runs with per-step timings

### Using it

```python
supervisor("قطتي تخربش الكنب، كيف أوقفها؟")          # -> care
supervisor("كم سعر تنظيف الأسنان؟")                   # -> clinic
answer_reviewed("كم تكلفة إيواء قطتين خمس ليالٍ؟")    # with the review loop

assistant.invoke({"question": "أبي أحجز استحمام",
                  "booking": {"service": "استحمام وتجفيف",
                              "slot": "الأحد 4:00 م"}}, config)
assistant.invoke(Command(resume="approve"), config)   # staff approves
```

---

## Built with

LangChain · LangGraph functional API · Chroma · Groq (`llama-3.3-70b-versatile`)
· `paraphrase-multilingual-MiniLM-L12-v2` embeddings · LangSmith

The corpus is Arabic, so the embedding model has to be multilingual — an
English-only model retrieves close to randomly on it while still appearing to
work.

---

## Repository

```
capstone.ipynb        the project, with saved outputs
data/                 the two knowledge sources
docs/writeup.md       how each part was built, and what it cost
.gitignore
```

---

## Known limitation

The grounding check in the notebook passes 7 of 8 questions. The one that fails
is left in place and diagnosed rather than tuned away: the corpus lists bathing
twice — 110 SAR under grooming services and 120 SAR as a boarding extra — and
semantic search returns the boarding wording because it is closer to the
question. The retriever is behaving correctly on an ambiguity in the source
documents; fixing it means rewriting the documents.

---

## Disclaimer

A training project. It does not diagnose illness and is not a substitute for a
licensed veterinarian. Prices and policies are invented.
