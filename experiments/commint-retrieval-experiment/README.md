# CommInt Retrieval Experiment

This experiment evaluates the Reddit retrieval pipeline used by CommInt to find communities and discussions related to a product problem.

It covers community discovery, thread retrieval, ranking, annotation, and evaluation across the retrieval sources used by the experiment.

## Retrieval pipeline

The main evaluation pipeline uses Reddit MCP for community discovery and Arctic Shift for post and comment retrieval.

The current flow is:
1. Generate search queries from the product problem.
2. Use Reddit MCP to retrieve candidate communities.
3. Rank and select communities for subreddit-scoped retrieval.
4. Retrieve recent posts from the selected communities using Arctic Shift.
5. Deduplicate the retrieved posts.
6. Order threads by `(num_comments, score)`.
7. Keep the top 15 threads.
8. Retrieve comments for the selected threads.

Comments are retrieved after thread selection and do not affect the thread ranking.

Agent-Reach is also evaluated as a separate discovery source. Its results are post hits from different communities rather than a single ranked community list, so they are evaluated separately from the Reddit MCP community ranking. Its safety and topic screening is also kept separate from relevance annotation.

## Evaluation

Relevance is annotated on a 0 to 3 scale:

* `0` = irrelevant
* `1` = partially relevant
* `2` = relevant
* `3` = highly relevant

The evaluation reports nDCG@10 and MRR for ranked items with complete annotations.

The thread evaluation uses the same 15 selected threads for both the popularity baseline and the comment-engagement heuristic. This measures how the methods order the selected threads rather than how well they retrieve threads from the full candidate pool.

## Current results

The selected-thread evaluation contains 15 judged threads. All 15 received relevance labels of `2` or `3`.

The popularity baseline has:

* nDCG@10: `0.8356`
* MRR: `1.0`

The comment-engagement heuristic has:

* nDCG@10: `0.8217`
* MRR: `1.0`

The comment-engagement heuristic is therefore evaluated as a re-ranking method over the same 15 selected threads, rather than as an independent retrieval method.

Its nDCG@10 is `0.0139` points below the popularity baseline, approximately `1.7%` lower relative to the baseline. Both methods achieve MRR `1.0`.

These results show that comment engagement produces a ranking broadly comparable to the popularity baseline on this small judged set. They do not establish retrieval recall or show that the heuristic improves retrieval quality.

The community discovery evaluation contains 14 unique judged communities from the Reddit MCP ranked output.

Agent-Reach produced post-level results across multiple communities. These are evaluated separately because the output does not provide the same ranked-community structure.

A production trace was also captured with tracing explicitly enabled. Discovery returned zero communities across all eight queries, so retrieval, filtering, and production ranking did not run. This trace therefore does not provide enough evidence to be able to evaluate production ranking quality.

## Project structure

```text
commint-retrieval-experiment/
├── cases/                         # Evaluation cases
├── data/
│   ├── annotations/              # Evaluation annotations
│   │   ├── latest/
│   │   │   ├── annotations.csv
│   │   │   └── auto_screen_exclusions_not_relevance.csv
│   │   └── annotate.py
│   ├── normalized/               # Generated normalized data
│   └── raw/                       # Raw API responses
├── evaluation/                   # Metrics and ranking logic
├── results/                      # Generated evaluation results
├── scripts/                      # Evaluation and data-processing scripts
├── sources/                      # Retrieval source implementations
├── tests/                        # Experiment tests
├── agent_reach_client.py
├── config.py
├── models.py
├── reddit_mcp_client.py
├── run_experiment.py
├── write_production_evaluation_status.py
├── README.md
└── requirements.txt
```

## Setup

Create the environment and install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

Configure the required sources in `.env` before running the experiment.

The current case configuration uses `study_accountability`.

## Running the experiment

Run the retrieval pipeline:

```powershell
python run_experiment.py
```

Build the annotation pool:

```powershell
python scripts/build_annotation_pool.py
```

Annotations are stored under:

```text
data/annotations/latest/
```

Merge updated annotations with:

```powershell
python scripts/merge_annotations.py
```

Run the evaluation:

```powershell
python scripts/evaluate.py
python scripts/evaluate_ranker_comparison.py
python scripts/audit_evaluation.py
```

Generate the production evaluation status:

```powershell
python write_production_evaluation_status.py
```

## Data

The experiment separates raw retrieval data, normalized data, annotations, and evaluation results.

* `data/raw/` contains raw responses from the retrieval sources.
* `data/normalized/` contains normalized communities and threads used by the evaluation.
* `data/annotations/` contains relevance annotations and the separate Agent-Reach safety/topic-screen exclusions.
* `results/` contains generated evaluation reports and metrics.

Production traces are kept outside the experiment data because they contain the original problem description and Reddit response data.

## Limitations

The current evaluation has a few limitations:

* Thread evaluation is limited to the selected top 15 threads.
* The complete pre-truncation candidate pool was not retained, so retrieval recall cannot be measured.
* The comment-engagement heuristic is evaluated as a re-ranking method over the selected threads.
* Agent-Reach and Reddit MCP produce different types of discovery output, so their rankings are not directly comparable.
* The captured production trace stopped at discovery because no communities were returned.
* The comment-engagement ranker is an experimental heuristic, not the production semantic community ranker.
* Source reliability and latency measurements are evaluated separately from relevance.

These limitations should be considered when interpreting the current metrics.
