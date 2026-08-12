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
| Institution | SDAIA Academy — <https://sdaia.gov.sa/en/Sectors/academy/Pages/default.aspx> |
| Declared track | **Track C — router across multiple knowledge sources** |

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

## Running it

Open `capstone.ipynb` in Google Colab and run it top to bottom. Two secrets are
read from Colab's secret manager (🔑 in the sidebar), both with *Notebook
access* switched on:

| Secret | Where to get it |
|---|---|
| `GROQ_API_KEY` | <https://console.groq.com> |
| `LANGCHAIN_API_KEY` | <https://smith.langchain.com> — free tier |

No key is written into the notebook, so none reaches this repository.

The knowledge base is written to disk by the notebook itself, so there is
nothing to upload. First run takes a few minutes while the embedding model
downloads.

**Tracing has to be switched on in the first cell**, before `langchain` is
imported. The variable is `LANGCHAIN_TRACING_V2`; `LANGSMITH_TRACING_V2` is not
a real variable and fails without an error.

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
