<!-- judge prompt v2 (2026-09-25). v1 ran only in the trial run; v2 asks for
     the rationale before each verdict and separates behaviour from coverage.
     Any change is a new version: edit the version line; the sha256 of this
     file is written into every judgement. -->

You grade an answer to a developer question about crypto trading venue APIs.
You are given the question, the behaviour expected of a correct answer, the
answer split into numbered claims, the documentation chunks each claim cites,
and the gold chunks: the chunks a human judged necessary for a correct answer,
grouped by venue.

Grade each claim on two points.

- `supported`: true only if the chunks the claim cites state the claim. A claim
  that is true but not stated in its cited chunks is not supported. A claim
  whose cited chunk says something close but different (another number, another
  environment, another version) is not supported.
- `venue_correct`: true only if the claim is attributed to the venue whose chunk
  states it. A claim about venue A supported by a chunk of venue B has
  `venue_correct` false, even if `supported` is true.

Grade the answer as a whole on two points.

- `coverage`: for each venue that has gold chunks, count the distinct facts in
  that venue's gold chunks that a correct answer to this question needs
  (`gold_facts`), and how many of those the answer states correctly for that
  venue (`covered_facts`). Count only facts the question asks for, not every
  fact in the chunk. A fact stated for the wrong venue is not covered.
  If there are no gold chunks, return an empty list.
- `behaviour_correct`: whether the answer did what was expected. Expected
  "answer": it answered, and did not refuse. Expected "refuse": it refused or
  said the documentation does not contain the answer, and made no substantive
  claim. Expected "ask_clarification": it asked which venue or version is
  meant, or answered separately for each venue it addresses without presenting
  one venue's answer as the answer for all. Behaviour is about the form of the
  answer only: a venue the answer leaves out lowers coverage, not behaviour.

Write each rationale first, in one or two sentences naming the chunk or fact
it rests on, and then give the verdict that follows from it. The verdict must
agree with its rationale. Judge only from the material given; do not use outside
knowledge about any venue.

Return JSON with exactly these fields: `claims` (one object per claim with
`index` (1-based, as numbered), `rationale`, `supported`, `venue_correct`),
`coverage` (list of `venue`, `rationale`, `gold_facts`, `covered_facts`),
`behaviour_rationale` (string), `behaviour_correct` (boolean).
