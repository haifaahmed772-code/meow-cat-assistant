# Write-up

Haifa Ahmed Alsulami · Wed Ayad Alshalawi
Building Agentic AI Systems — SDAIA Academy, 9–13 August 2026
Track C — router across multiple knowledge sources

Every claim below comes from output saved in `capstone.ipynb`.

---

## 1 · Agent fundamentals

Five tools, each computing from its arguments. The clearest is
`daily_calorie_needs`, which applies the resting energy requirement
`RER = 70 × weight^0.75` and then a life-stage factor. The exponent makes the
output non-linear: 4 kg at 1.2 gives 237.6 kcal and 8 kg gives 399.6 — doubling
the weight raises the requirement by 68%, not 100%. `boarding_cost` applies
three rules in order, so 6 nights in a suite with a second cat and daily play
comes to 1,350 SAR. `today()` was added after the agent, having no clock, told
us a cat born in 2023 was one year old.

The model picks the tools itself. Asked one question containing both a feeding
and a boarding query, it called `daily_calorie_needs` and `boarding_cost` and
pulled `weight_kg: 4.6`, `nights: 6` and `room: 'suite'` out of Arabic prose.

Structured output is used wherever code consumes the result rather than a
person reading it: `CatProfile` turns a free-text description into typed fields,
and those fields are what later gets written into long-term memory.

`Triage.needs_vet` is `"yes"`/`"no"` rather than a boolean, and so are the flags
on `boarding_cost`. This was not a stylistic choice. Groq validates the tool
call against the schema on its own side and rejected the whole request when the
model produced `"false"` as a string — `parameters for tool Triage did not
match schema`. Because the rejection happens before the response reaches
Python, a Pydantic validator cannot repair it. The schema was changed to ask
for what the model reliably produces.

## 2 · Multi-agent / routing architecture

A supervisor classifies each question into `care`, `clinic` or `both` using
`with_structured_output`, then hands it to a specialist agent — `care_agent`
or `clinic_agent`. Each specialist is built with `create_agent` and holds only
its own search tool, so the care agent physically cannot read the services
guide and vice versa; the separation is in the tools, not in a filter applied
afterwards. The run output names the agent that received each question and the
tools it called, and `search_clinic_services` never appears under
`care_agent`.

Only the chosen path runs, so a wrong route produces a wrong answer rather than
being quietly rescued by searching everything. The functional pipeline in §6
reuses the same classifier but calls the retrievers directly, so that retries
and the approval step apply to individual steps rather than to a whole agent
turn.

The case that justifies using a model here rather than keywords is *"my cat has
been lethargic for a whole day after her vaccination — is that normal?"* It
contains تطعيم, which belongs to the services guide, but the owner is asking
whether a symptom is normal. The classifier answered `care`, reasoning that the
owner was asking about behaviour after vaccination. `if "تطعيم" in question`
would have returned the price list.

`both` is a real path, not decoration: *"how often do I brush a Persian, and
what does a full clip cost?"* was routed to `both` and the run reports
`searched: ['care', 'clinic']`.

## 3 · RAG pipeline

Two documents, 13 chunks each, split on `##` headings, embedded with
`paraphrase-multilingual-MiniLM-L12-v2` and held in two separate Chroma
collections rather than one collection with a metadata filter — a filter would
let a wrong route still find the right answer.

The model choice matters more than it looks. The corpus is Arabic, and a
similarity check gives 0.455 between two related Arabic questions against
−0.017 for an unrelated pair, which is the separation the retrieval depends on.
An English-only model would have retrieved close to randomly while still
appearing to work.

Retrieval is tested two ways. First, each store is asked its own question and
then the other store's, and the second case returns visibly unrelated text —
that failure is the evidence the stores are genuinely separate. Second, eight
questions with known answers are checked for the expected string in the
retrieved context. **Seven of eight pass.**

The one that fails is left in the notebook. Asked the price of bathing, the
retriever returns the boarding extra (120 SAR, *الاستحمام والتنظيف الكامل قبل
الاستلام*) instead of the grooming service (110 SAR, *الاستحمام والتجفيف*). The
retriever is not broken: the corpus describes bathing twice in similar words,
and the boarding wording is closer to the question. The fix belongs in the
documents, not in the search. An earlier version of this problem was real —
with `k=3` and 450-character chunks, three price questions came back "not in
the context" when the prices were sitting in the file. Finer chunks and `k=5`
resolved those.

**2-Step vs Agentic vs Hybrid.** This pipeline is 2-Step: classify, then
retrieve once, then answer. It suits the problem because the two sources are
small and cleanly divided, so one well-aimed retrieval is enough, and the trace
shows retrieval costs 0.09 s against 2–4 s per model call — the expensive part
is thinking, not searching, so extra agentic retrieval rounds would add latency
without adding much. Agentic RAG would be the right choice if a question needed
several dependent lookups; the one place it would help here is the ambiguous
bathing question, where an agent could notice two candidate prices and ask
which service was meant.

## 4 · Context and state management

Short-term state is a checkpointer keyed by `thread_id`. In `chat-A` the second
message asks *"and how old is she?"* without repeating the name, and it is
answered from the earlier turn.

Long-term memory is a separate `InMemoryStore` under `("cats", owner_id)`,
holding two cats. It is attached to no conversation.

The cross-thread test is what separates the two. `chat-B` is a fresh thread
asking for لولو's calories with no store lookup, and it does not admit it does
not know — it invents, calling the tool with `weight_kg: 5, life_stage:
'kitten'`. `chat-C`, also fresh and sharing no history with `chat-A`, loads the
profile from the Store and calls the same tool with `weight_kg: 4.6,
life_stage: 'neutered_adult'`. That 4.6 was typed into `chat-A` and reached
`chat-C` only through the Store.

The fabricated weight is the more useful half of the result. It shows that
without persistent facts the assistant does not degrade into saying "I don't
know" — it degrades into confident invention.

## 5 · Human-in-the-loop

Booking pauses on `interrupt()` before anything is written to the diary,
because it takes a 50 SAR deposit and cancelling inside 24 hours forfeits it.
The pause carries the service, the slot, the deposit and the triage urgency, so
whoever approves it sees what they are approving.

Three complete pause-and-resume cycles are in the notebook:

- `run-2` — `Command(resume="approve")` → `BOOKED — استحمام وتجفيف at الأحد 4:00 م`
- `run-3` — resumed with a refusal in Arabic → `NOT BOOKED` with the staff reason
- `run-4` — an emergency, where the pause shows `triage: emergency` before the decision

The refusal matters as much as the approval: it shows the human decision
changes the outcome rather than being a formality.

## 6 · LangGraph functional API and error handling

Built with `@task` and `@entrypoint`. The retry keyword is read from the
signature at runtime, because `task` has taken the policy under different
names across LangGraph versions and hard-coding it breaks on a different
install.

Three strategies are implemented:

**Retry.** A real `RetryPolicy(max_attempts=3)`, not a loop with `sleep`. The
demo task fails twice on purpose and the output shows `attempt 1`, `attempt 2`,
`attempt 3`.

**Fallback.** If the classifier raises, routing falls back to searching both
stores — slower and less precise, but the owner still gets an answer. Forcing
the failure prints `routing failed (RuntimeError) — searching both stores` and
the run completes.

**Refusal and escalation.** A separate triage step decides whether the cat needs
a vet, returning a constrained field rather than prose. *"She has been straining
to urinate for two hours with nothing coming out"* is classified `emergency`,
and the warning is placed above the answer rather than inside it.

## 7 · Workflow pattern

**Evaluator-optimizer.** One model call writes the reply, a second reads it
against the retrieved context, and a failed verdict sends it back to be
rewritten, up to three rounds.

It was chosen because the failure this assistant is actually prone to is a
fluent sentence about a service that no retrieved chunk supports — not a wrong
calculation, which the tools prevent. Routing already carries the
multi-source architecture, so naming it again here would describe the same
mechanism twice.

It caught exactly the fault it was built for. The grooming answer had claimed
*"you can book an appointment to brush your cat's fur with us at Meow Clinic"*
without the services guide ever having been searched. The review returned
`grounded=no` naming that clause, and round 2 returned `grounded=yes` with the
claim dropped and only the supported grooming advice left.

## 8 · LangSmith observability

Tracing runs over the whole notebook, producing 25 traced runs.

The first attempt produced no trace at all. Tracing was switched on in Part 4,
after `langchain` had been imported and used for thirty cells, and the tracer is
built once — so `client.list_projects()` succeeded, `LANGCHAIN_TRACING_V2` was
set, and yet reading the project back raised `Project meow-clinic-capstone not
found`, because the project is only created when a first trace arrives. Moving
the setup into the first cell fixed it. This is a second silent failure mode
alongside the misspelled variable name: correct spelling, wrong moment, same
empty result.

What the trace shows, for a 3.55 s run:

| step | seconds |
|---|---|
| `retrieve_task` | 0.09 |
| `classify_task` | 0.33 |
| `answer_task` | 0.84 |
| `review_answer` | 2.28 |

Retrieval is 2% of the run. Everything else is waiting on the model. The
unexpected part is that reviewing an answer costs about three times what
writing it costs, because the reviewer receives the answer *and* the full
context. On the question that failed review, the loop added `review 4.36 →
revise 3.50 → review 4.61`, so catching one unsupported sentence cost roughly
12.5 s on top of an answer that was already finished.

That is worth paying when a reply states a price or names a service, and hard
to justify on every question. The change it points to is running the review
conditionally — when the draft contains a number or a service name — rather
than on all traffic.

---

## What we would do next

- Reword the two bathing services so the ambiguity in §3 disappears at source.
- Make the review conditional, on the evidence in §8.
- Replace `InMemoryStore` and `InMemorySaver` with persistent backends; both are
  lost when the runtime restarts, which is fine for a notebook and not for a
  clinic.
