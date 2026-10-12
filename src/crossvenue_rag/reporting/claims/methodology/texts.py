"""The text evals/methodology.md defines rather than measures, with the reason for each.

Each tuple of sentences is literal text of the document whose numbers are a registered
target, a quotation or a numbering; a section passes it to `Doc.definitions` with its
reason. The three tuples at the end, of category names and question ids, are arguments of
claims and compute functions.
"""

from __future__ import annotations

LITERATURE = "quoted from the cited paper"
CONVENTION = "a numbering or naming in the text"

# Decisions at a glance: the row numbers and the registered parameters no constant holds.
DECISIONS = (
    "| 1 | Labeled set size |",
    "| 2 | Category split |",
    "| 3 | Venue coverage |",
    "| 4 | Question source |",
    "| 5 | Pooling |",
    "| 6 | Pooling-bias disclosure |",
    "| 7 | Intra-rater reliability |",
    "A 7-day fallback is defined",
    "| 8 | Question dedup threshold |",
    "| 9 | Significance testing |",
    "| 10 | Metric under incompleteness |",
    "| 11 | Answer evaluation |",
)

# Section 1: the pooling literature.
POOLING_STUDIES = (
    "Depth 100 became the ad-hoc default.",
    "The [TREC 2017 Common Core track overview]",
    (
        "It judged depth-10 pools, then chose further documents to depth 100 with a bandit method, "
        "and judged a 20% sample of ranks 11\u201375 for inferred measures."
    ),
    "[Zobel (1998),",
    "downsampled to depth 10 but not at greater depths",
    "[Buckley, Dimmick, Soboroff & Voorhees (2007),",
    "(Information Retrieval 10(6):491\u2013508)",
    "[B\u00fcttcher, Clarke, Yeung & Soboroff (2007)]",
    "[Voorhees, Soboroff & Lin (2022),",
    "They re-judged TREC-8 against",
    (
        "The re-judging pools, 9 to 359 documents per topic over 50 topics, held 158 relevant "
        "documents in total, no run retrieved more than 23 of them"
    ),
    "(Kendall's \u03c4 > 0.99)",
    "[Buckley & Voorhees (2004),",
    "[Sakai (2007),",
    "One difference from Sakai (2007)",
)

# Section 2: the literature on generated questions.
GENERATION_STUDIES = (
    "(Saad-Falcon et al., NAACL 2024)",
    "it assembled 18 existing human-built retrieval datasets",
    "(Hong, Troynikov, Huber & McGuire, April 2025)",
    "with 11.91% of query pairs above 0.9 cosine similarity",
    "from 0.716 to 0.628",
    "[Kucia & Gawlik (2026),",
    "compared 1,851 synthetic questions against 322 authentic queries",
    (
        "Authentic queries averaged 6.8 words vs 15.7 for synthetic ones and drew on 53 unique "
        "sources vs 165."
    ),
    "[van Elburg, van der Putten & Marx (2025),",
    "[Rahmani, Ramineni, Yilmaz, Craswell & Mitra (2025),",
    "(Filice et al., 2025)",
)

# Section 3: SQuAD 2.0 and the unanswerable-question literature.
UNANSWERABLE_STUDIES = (
    "[Rajpurkar, Jia & Liang (2018),",
    "added over 50,000 unanswerable questions",
    "A system at 86% F1 on SQuAD 1.1 dropped to 66% F1 on 2.0, against a human ceiling of 89.5 F1.",
    "| Train | 130,319 | 43,498 | 33.4% |",
    "| Dev | 11,873 | 5,945 | 50.1% |",
    "| Test | 8,862 | 4,332 | 48.9% |",
    "(Kwiatkowski et al., TACL 2019)",
    "[Chen, Lin, Han & Sun (2024), RGB]",
    "one of its four measured abilities",
    "SQuAD 2.0's 50%",
    "below SQuAD 2.0's 50% and above zero",
)

# Section 4: the ambiguity and conflict literature.
AMBIGUITY_STUDIES = (
    "([Chen et al. 2024]",
    "[Min, Michael, Hajishirzi & Zettlemoyer (2020), AmbigQA]",
    "The dataset, AmbigNQ, covers 14,042 questions.",
    "[Wang, Prasad, Stengel-Eskin & Bansal (2025),",
    "It argues the three need",
    "None of the three sources above",
)

# Section 5: the agreement bands.
AGREEMENT_STUDIES = (
    "[Landis & Koch (1977),",
    "(Biometrics 33(1):159\u2013174)",
    (
        "The bands are < 0.00 poor, 0.00\u20130.20 slight, 0.21\u20130.40 fair, 0.41\u20130.60 "
        "moderate, 0.61\u20130.80 substantial, 0.81\u20131.00 almost perfect."
    ),
)

# Section 5: the fallback gap between the two passes.
GAP_RULES = (
    "At 7 days the probability",
    'a \u03ba obtained at 7 days is reported as "\u03ba = X (7-day gap;',
    "Below 7 days the number stops being informative",
    "if even 7 days does not fit",
)

# The annotator instructions: the gap.
INSTRUCTION_GAP = (
    "round 2 ran at least fourteen days after round 1",
    "on day 1, a low kappa on day 15",
    "or the same person two weeks later",
)

# The annotator instructions: the categories and the three hard cases.
INSTRUCTION_CASES = (
    "| fact | One or two chunks,",
    "Three cases that would otherwise be decided differently on different days.",
    '"order creation: 10/s"',
    "2. Two chunks stating the same fact",
    "a minimal cover would keep one of the two.",
    "3. A code example with no prose",
    "1. A general rule and a concrete number",
)

# Section 6: the deduplication literature.
DEDUPLICATION_STUDIES = (
    "Two near-identical questions in an evaluation set are not two observations.",
    "the independence assumption of every paired significance test in section 7.",
    "[Lee, Ippolito, Nystrom et al. (2022),",
    "affects over 4% of the validation sets",
    "(on C4, 7.18% of tokens against 3.04% of examples)",
    "[Abbas, Tirumala, Simig, Ganguli & Morcos (2023), SemDeDup]",
    "1 \u2212 \u03b5 semantic duplicates",
    "Chroma found 11.91% of naively generated query pairs above 0.9 cosine",
    "A threshold at 0.85 or 0.90 would delete",
    (
        "([GPT-3](https://arxiv.org/abs/2005.14165) used 13-gram overlap, "
        "[GPT-4](https://arxiv.org/abs/2303.08774) 50-character substrings)"
    ),
)

# Section 7: the literature on topic-set size.
POWER_STUDIES = (
    "[Voorhees & Buckley (2002),",
    "for sets up to size 25 and extrapolated above that",
    "even at TREC's standard 50 topics",
    "[Sakai (2016),",
    "(Information Retrieval Journal 19(3):256\u2013283)",
    "typically have n = 50\u2013100 topics",
    "[Smucker, Allan & Carterette (2007),",
    "across TREC 3 and 5\u20138",
    "Their [2009 follow-up]",
    "as topic counts shrank toward 10",
    "below p = 0.1",
    "not to run a 15-question evaluation",
    "(Smucker et al. 2009)",
    "Smucker et al. (2007)",
    "As the topic count falls toward 10",
    "inside Sakai's observed 50\u2013100 band",
)

# Section 7: the sample-size decision and the reporting rules it sets.
SAMPLE_SIZE_RULES = (
    "Above ~120 the binding constraint",
    "([style rule 13](../docs/style.md))",
)

# Number words used as words, and names.
WORDING = (
    "(UTC\u22123)",
    "in UTC\u22123",
    "Section 5's \u03ba",
    "Half the labeling budget",
    "Agreement between two raters",
    "computed between the two passes",
    "when the two ratings come from the same rater",
    "differs between the two references",
    "Disagreements between the two passes",
    "the only thing connecting two others",
    "Those two observations",
    "They are one observation counted twice",
    "when two questions are generated from adjacent chunks",
    "are not two observations",
    '"which of these two venues documents X"',
    "exactly one of the two methods hits",
    "Each of the two is tested",
    "the two primary metrics",
    "the agreement between the two judges",
    (
        "1 if the first chunk's venue is not among the question's labeled `venues` and is not ccxt "
        "on a question that does not name ccxt, else 0."
    ),
    "1 if the first chunk is a ccxt chunk on a question that does not name ccxt, else 0.",
    "An LLM, Claude Opus 5 (`claude-opus-5`),",
)

# Number words in descriptions of the literature.
WORDING_STUDIES = (
    "the sparse half of a hybrid retriever",
    "found that over half of open-domain Natural Questions",
)

# Section 9: the answer configurations, the judges and missing metrics.
ANSWER_CONVENTIONS = (
    "| gemini@venue_split | `google/gemini-3.8-flash`",
    "| gemini@hybrid_rerank | `google/gemini-3.8-flash`",
    "`qwen/qwen3.8-27b` (DeepInfra, bf16, no fallback)",
    "`inception/mercury-2.5`",
    "A metric that does not apply to an answer is missing, never 0,",
)

# Section 5: the categories the round-2 retest samples in number, and those it samples one
# question of each.
SAMPLED = ("comparison", "unanswerable", "adversarial", "fact")
SINGLE_SAMPLED = ("ambiguous", "multihop", "noisy", "version_sensitive")
# Section 9: the questions of the answer trial run, excluded from the calibration.
TRIAL_QUESTIONS = ("cv-0003", "cv-0045", "cv-0081", "cv-0105", "cv-0122")
