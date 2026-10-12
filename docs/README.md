# Documentation

Every document of the repository, by the types of the [Document types](style.md#document-types) table.

## Overview

| Document | Contents |
|---|---|
| [`../README.md`](../README.md) | What the project is, how to run it, headline results with their runs |

## Explanation

| Document | Contents |
|---|---|
| [`explanation/design.md`](explanation/design.md) | How the pipeline is built and why: module layout, normalization, chunking, deduplication, metadata, `index_version`, retrieval methods, configuration, latency measurement, answer generation, the interface, error handling |
| [`explanation/directions.md`](explanation/directions.md) | Nine development directions, each with the measured problem it addresses and the measurement that decides it |

## Reference

| Document | Contents |
|---|---|
| [`reference/corpus.md`](reference/corpus.md) | Datasheet of the document corpus: sources, composition, collection, preprocessing, licenses, gaps |
| [`reference/cli.md`](reference/cli.md) | Every `crossvenue` command with its arguments and options |
| [`reference/config.md`](reference/config.md) | Every key of `configs/default.yaml` with type, value and meaning |
| [`../evals/datasheet.md`](../evals/datasheet.md) | Datasheet of the gold set `v0.2.1-gold` |
| [`../evals/methodology.md`](../evals/methodology.md) | Evaluation protocol, what it can prove, deviations from it |
| [`../evals/domain-research.md`](../evals/domain-research.md) | The developer questions the evaluation set is drawn from: sources, ranked themes, near-duplicate traps |

## Results

| Document | Contents |
|---|---|
| [`results/findings.md`](results/findings.md) | Measured results, each with its run |
| [`../evals/agreement.md`](../evals/agreement.md) | Test-retest agreement of the annotator between labeling rounds 1 and 2: κ with its interval, per category, direction of disagreement, source-chunk verdicts |

## How-to

| Document | Contents |
|---|---|
| [`how-to/reproduce.md`](how-to/reproduce.md) | How to check the numbers of the documents `crossvenue reproduce` covers against their sources, and to rerun each measurement |
| [`how-to/label.md`](how-to/label.md) | How to select, label, correct, augment, validate and freeze a question set |

## Decision records

[`adr/`](adr/README.md): why each decision is taken, with the alternatives, one record per decision.

## History

| Document | Contents |
|---|---|
| [`../evals/datasets/CHANGELOG.md`](../evals/datasets/CHANGELOG.md) | Each version of the question set: what changed, when and why |
| [`adr/`](adr/README.md) | The decision records, in the order the decisions were taken |

## Tables of contents

| Document | Contents |
|---|---|
| [`../evals/README.md`](../evals/README.md) | Each file of `evals/`: the protocol, the agreement results, the datasheet, the domain research, the question sets and the journals |
| [`adr/README.md`](adr/README.md) | Each decision record with its status |

## Rules

| Document | Contents |
|---|---|
| [`style.md`](style.md) | Writing rules for every Markdown file, and the tools that check them |
| [`code-style.md`](code-style.md) | Rules for the Python code, and the tools that check them |
