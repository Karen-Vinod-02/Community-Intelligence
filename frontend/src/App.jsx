import { useEffect, useState } from "react";
import "./App.css";

const API = import.meta.env.VITE_API_BASE_URL || "";
const EXAMPLES = [
  "A tool that helps independent therapists reduce cancellations and fill last-minute openings.",
  "A simple way for new managers to give better feedback and run useful 1:1s.",
  "A study group app that helps university students find reliable partners for exam prep.",
];

const PLATFORM_LABELS = {
  reddit: "Reddit",
  hackernews: "Hacker News",
  discord: "Discord",
  slack: "Slack",
  linkedin: "LinkedIn",
  indiehackers: "Indie Hackers",
};

const communityKey = (community) =>
  `${community.platform}:${community.name || community.subreddit}`;
const hasDraftEvidence = (community) => (community.posts || [])
  .some((post) => post.title?.trim() || post.body?.trim());

let healthRequest;

function ScoreBar({ score, label }) {
  const percentage = Math.round(Math.max(0, Math.min(1, score)) * 100);

  return (
    <div className="score-bar-container">
      <div className="score-bar" role="img" aria-label={`${label}: ${percentage}%`}>
        <div
          className="score-bar-fill"
          style={{ width: `${percentage}%` }}
        />
      </div>

      <span className="score-value" aria-hidden="true">{percentage}%</span>
    </div>
  );
}

function Explanation({ community }) {
  const explanation = community.explanation;

  return (
    <details className="explanation" open={Boolean(explanation)}>
      <summary>
        <span>Why this community?</span>
        <span className="summary-arrow" aria-hidden="true">+</span>
      </summary>

      {explanation ? (
        <div className="explanation-content">
          {explanation.summary && (
            <p className="explanation-summary">
              {explanation.summary}
            </p>
          )}

          {explanation.matches?.length > 0 && (
            <div className="evidence-list">
              {explanation.matches.map((match, index) => (
                <div className="evidence-item" key={index}>
                  <div className="evidence-header">
                    <span>{match.label}</span>

                    {typeof match.score === "number" && (
                      <span title="Best matching problem dimension">{Math.round(match.score * 100)}% match</span>
                    )}
                  </div>

                  <p>{match.text}</p>
                  {match.source_url && (
                    <a href={match.source_url} target="_blank" rel="noreferrer">
                      Open supporting evidence ↗
                    </a>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div className="explanation-placeholder">
          <p>
            Community relevance is determined from semantic similarity,
            problem evidence, and discussion recency.
          </p>

          <p className="placeholder-note">
            Detailed evidence will appear here when available.
          </p>
        </div>
      )}
    </details>
  );
}

function PostCard({ post }) {
  const body = post.body || "";
  const date = post.created_utc
    ? new Date(post.created_utc * 1000).toLocaleDateString()
    : null;

  return (
    <article className="post">
      <div className="post-header">
        <span className="post-label">Discussion</span>
        {post.evidence_source === "duckduckgo_index" && (
          <span className="post-label">Search index, unverified</span>
        )}
        {post.evidence_source === "web_search_thread" && (
          <span className="post-label">Reddit thread found via web search</span>
        )}
        {post.evidence_source === "reddit_search_api" && (
          <span className="post-label">Reddit search, relevance unverified</span>
        )}
        {post.evidence_source === "agent_reach_opencli" && (
          <span className="post-label">Reddit via Agent Reach</span>
        )}
        {post.evidence_source === "reddit_public_json" && (
          <span className="post-label">Reddit public result, ranking fallback</span>
        )}
        {post.evidence_source === "arcticshift_retrieval" && (
          <span className="post-label">Reddit thread, relevance lead</span>
        )}

        {post.score !== null && post.score !== undefined && (
          <span className="post-score">
            {post.score} points
          </span>
        )}
      </div>

      <h5>{post.title}</h5>

      {body && (
        <p className="post-body">
          {body.length > 350
            ? `${body.slice(0, 350).trim()}...`
            : body}
        </p>
      )}

      <div className="post-footer">
        <span>
          {date
            ? `Posted ${date}`
            : post.evidence_source === "duckduckgo_index"
              ? "DuckDuckGo search result; date and thread details not independently verified"
              : post.recency_verified
                ? "Found in a past-month search; exact date unavailable"
                : "Post date unavailable"}
        </span>
        {post.num_comments !== null &&
          post.num_comments !== undefined && (
            <span>
              {post.num_comments} comments
            </span>
          )}

        {post.url && (
          <a
            href={post.url}
            target="_blank"
            rel="noreferrer"
          >
            View discussion ↗
          </a>
        )}
      </div>
    </article>
  );
}

function CommunityCard({
  community,
  index,
  onDraftMessage,
  draft,
  draftLoading,
  draftError,
}) {
  const rankPercentage = Math.round(Math.max(0, Math.min(1, community.score)) * 100);
  const platform = PLATFORM_LABELS[community.platform] || community.platform;
  const signals = community.signals || {};
  const posts = community.posts || [];
  const relevance = typeof signals.evidence_mean === "number"
    ? Math.round(Math.max(0, Math.min(1, signals.evidence_mean)) * 100)
    : null;
  const rankLabel = community.rank_tier === "potential" ? "Adjusted rank score" : "Rank score";

  return (
    <article className="community-card">
      <div className="community-top">
        <div className="community-rank">
          #{index + 1}
        </div>

        <div className="community-title">
          <h3>
            {community.platform === "reddit" ? "r/" : ""}
            {community.name || community.subreddit}
          </h3>

          <span className="community-label">
            <span>{platform}</span>
            {community.community_kind === "indexed_discussions" && (
              <span>Public indexed discussions, not a verified group</span>
            )}
            <span className="relevance-tag">
              {community.status === "suggestion" ? "Starting point · no evidence" : "Relevant evidence"}
            </span>
          </span>
        </div>

        {community.status !== "suggestion" && <div className="community-score">
          <strong>{rankPercentage}%</strong>
          <span title="Weighted from discussion relevance (80%), prevalence (15%), and recency (5%). Potential-tier results also receive a ranking adjustment.">{rankLabel}</span>
        </div>}
      </div>

      {community.status !== "suggestion" && <ScoreBar score={community.score} label={rankLabel} />}

      <div className="community-metrics">
        {community.status === "suggestion" ? (
          <span>No verified discussions found; explore this community manually.</span>
        ) : relevance !== null && (
          <span title="Mean evidence score across matching discussions">{relevance}% average discussion relevance</span>
        )}
        {community.status !== "suggestion" && <span>
          {signals.problem_relevant_posts || posts.length} relevant{" "}
          {(signals.problem_relevant_posts || posts.length) === 1 ? "discussion" : "discussions"}
        </span>}
        {community.analyzed_posts > 0 && <span>{community.analyzed_posts} posts reviewed</span>}
        {signals.avg_comments !== null && signals.avg_comments !== undefined && (
          <span>{signals.avg_comments} avg. comments on matching posts</span>
        )}
        {community.url && (
          <a href={community.url} target="_blank" rel="noreferrer">
            Visit community ↗
          </a>
        )}
      </div>

      <Explanation community={community} />

      <div className="posts-section">
        <div className="posts-heading">
          <h4>Relevant discussions</h4>

          <span>
            {posts.length} shown
          </span>
        </div>

        {posts.length > 0 ? (
          <div className="posts">
            {posts.map((post) => (
              <PostCard
                key={post.id || post.url || post.title}
                post={post}
              />
            ))}
          </div>
        ) : (
          <div className="no-posts">
            "No verified discussions are available for this community."
          </div>
        )}
      </div>

      <div className="outreach-box">
        {draftLoading && !draft && (
          <p className="placeholder-note">Writing your first message…</p>
        )}
        {draft && (
          <div className="generated-message">
            <strong>Suggested first message</strong>
            <p>{draft.message}</p>
            <small>Draft for review</small>
            <p className="review-notice">{draft.review_notice}</p>
          </div>
        )}
        {draftError && <p className="inline-error">{draftError}</p>}
        <button
          type="button"
          className="secondary-button"
          onClick={() => onDraftMessage(community)}
          disabled={draftLoading || !hasDraftEvidence(community)}
        >
          {draftLoading
            ? "Drafting…"
            : !hasDraftEvidence(community)
              ? "Discussion evidence required"
              : draft
                ? "Regenerate first message"
                : "Draft a help-first contribution"}
        </button>
      </div>
    </article>
  );
}

function App() {
  const [description, setDescription] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState({ stage: "starting", percent: 2 });
  const [error, setError] = useState("");
  const [competitors, setCompetitors] = useState(null);
  const [competitorsLoading, setCompetitorsLoading] = useState(false);
  const [competitorsError, setCompetitorsError] = useState("");
  const [outreach, setOutreach] = useState({});
  const [outreachLoading, setOutreachLoading] = useState({});
  const [outreachErrors, setOutreachErrors] = useState({});
  const [health, setHealth] = useState(null);

  useEffect(() => {
    let active = true;
    healthRequest ||= fetch(`${API}/api/health`)
      .then((response) => {
        if (!response.ok) throw new Error("API unavailable");
        return response.json();
      });
    healthRequest
      .then((data) => { if (active) setHealth(data); })
      .catch(() => {
        healthRequest = null;
        if (active) setHealth({ status: "unavailable" });
      });
    return () => { active = false; };
  }, []);

  const loadCompetitors = async (productDescription) => {
    setCompetitorsLoading(true);
    setCompetitorsError("");
    try {
      const response = await fetch(`${API}/api/competitors`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ description: productDescription }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Competitor search failed.");
      }
      setCompetitors(data);
    } catch (requestError) {
      setCompetitorsError(requestError.message || "Competitor search failed.");
    } finally {
      setCompetitorsLoading(false);
    }
  };

  const requestDraft = async (community, productDescription) => {
    const key = communityKey(community);
    const evidencePosts = (community.posts || [])
      .map((post) => ({
        title: (post.title || "").trim().slice(0, 500),
        body: (post.body || "").trim().slice(0, 1200),
        source_url: post.url || null,
      }))
      .filter((post) => post.title || post.body)
      .slice(0, 3);
    if (!evidencePosts.length) return;
    setOutreachLoading((current) => ({ ...current, [key]: true }));
    setOutreachErrors((current) => ({ ...current, [key]: "" }));
    try {
      const response = await fetch(`${API}/api/outreach`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          description: productDescription,
          platform: community.platform,
          community_name: community.name || community.subreddit || "community",
          community_url: community.url || null,
          evidence_titles: evidencePosts.map((post) => post.title).filter(Boolean),
          evidence_posts: evidencePosts,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Message generation failed.");
      }
      setOutreach((current) => ({ ...current, [key]: data }));
    } catch (requestError) {
      setOutreachErrors((current) => ({
        ...current,
        [key]: requestError.message || "Message generation failed.",
      }));
    } finally {
      setOutreachLoading((current) => ({ ...current, [key]: false }));
    }
  };

  const autoDraftAll = async (communities, productDescription) => {
    const targets = (communities || [])
      .filter(
        (community) =>
          (community.posts || []).some((post) => post.title?.trim() || post.body?.trim()),
      )
      .slice(0, 5);
    for (const community of targets) {
      // Sequential to avoid overloading a local model.
      await requestDraft(community, productDescription);
    }
  };

  const handleAnalyze = async () => {
    if (!description.trim()) {
      setError("Please enter a product description.");
      return;
    }

    const productDescription = description.trim();
    setLoading(true);
    setAnalysisProgress({ stage: "starting", percent: 2 });
    setResult(null);
    setError("");
    setCompetitors(null);
    setCompetitorsError("");
    setOutreach({});
    setOutreachErrors({});

    let stopProgressPolling = false;
    const progressId = crypto.randomUUID();
    const pollProgress = async () => {
      while (!stopProgressPolling) {
        try {
          const progressResponse = await fetch(
            `${API}/api/analysis-progress/${progressId}`,
          );
          if (progressResponse.ok) {
            const progress = await progressResponse.json();
            setAnalysisProgress(progress);
            if (progress.complete) return;
          }
        } catch {
          // The analysis request remains authoritative if progress polling fails.
        }
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
    };
    pollProgress();

    try {
      const response = await fetch(`${API}/api/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Analysis-Progress-ID": progressId,
        },
        body: JSON.stringify({
          description: productDescription,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "Failed to analyze the product.");
      }

      const data = await response.json();
      setResult(data);
      // Background work: do not keep the main spinner on for these.
      loadCompetitors(productDescription);
      autoDraftAll(data.communities, productDescription);
    } catch (requestError) {
      setError(requestError.message || "Could not analyze the product. Make sure the backend is running.");
    } finally {
      stopProgressPolling = true;
      setAnalysisProgress({ stage: "complete", percent: 100 });
      setLoading(false);
    }
  };

  const handleFindCompetitors = async () => {
    await loadCompetitors(description.trim());
  };

  const handleDraftMessage = (community) =>
    requestDraft(community, description.trim());

  const handleKeyDown = (event) => {
    if (
      (event.ctrlKey || event.metaKey) &&
      event.key === "Enter"
    ) {
      handleAnalyze();
    }
  };

  const sourceIssues = Object.entries(result?.sources || {})
    .filter(([, status]) => ["unavailable", "partial_failure"].includes(status))
    .map(([source, status]) => (
      `${source.replaceAll("_", " ")} (${status.replaceAll("_", " ")})`
    ));
  const publicSearchAvailable = Object.entries(result?.sources || {})
    .some(([source, status]) => source.startsWith("public_") && status !== "unavailable");
  const discussionCount = (result?.communities || []).reduce(
    (total, community) => total + (community.posts || []).length,
    0,
  );
  return (
    <main className="app" id="top">
      <header className="product-header">
        <a className="brand" href="#top" aria-label="CommInt home">
          <span className="brand-mark">C</span>
          <span>COMMINT</span>
        </a>
        <div className={`system-status ${health?.status === "ok" ? "is-online" : ""}`}>
          <span className="status-dot" />
          {health === null ? "Connecting" : health.status === "ok" ? "Workspace ready" : "API offline"}
          {health?.supabase && (
            <span className="cache-status">
              {health.supabase.status === "connected" ? "Supabase connected" :
                health.supabase.status === "memory_only" ? "Local cache" : "Supabase setup needed"}
            </span>
          )}
        </div>
      </header>
      <section className="hero-section">
        <div className="hero-copy">
          <p className="hero-kicker"><span /> Community research for product teams</p>
          <h1>Find real discussions about the problem you solve.</h1>
          <p className="subtitle">
            Search recent public threads, then see why each one matches before joining the conversation.
          </p>
          <div className="hero-footnote">
            <span className="footnote-mark" aria-hidden="true">↳</span>
            Every result opens with a source-linked discussion.
          </div>
        </div>

        <div className="hero-workflow">
          <label className="prompt-label" htmlFor="product-description">
            What does your product help people do?
          </label>
          <p className="prompt-hint">Include who it is for and the problem they want to solve.</p>
          <div className="input-container">
            <div className="input-wrapper">
              <textarea
                id="product-description"
                rows="5"
                placeholder="A tool that helps independent therapists fill last-minute appointment openings."
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={loading}
              />
              <span className="input-hint">Ctrl + Enter</span>
            </div>
            <button onClick={handleAnalyze} disabled={loading}>
              {loading ? <><span className="button-spinner" /> Searching conversations</> : "Find communities"}
            </button>
          </div>
          {error && <div className="error" role="alert">{error}</div>}
          <div className="example-prompts" aria-label="Try a sample description">
            <span>Try:</span>
            {EXAMPLES.map((example, index) => (
              <button type="button" className="example-chip" key={example} onClick={() => setDescription(example)}>
                {index === 0 ? "Therapist scheduling" : index === 1 ? "Manager feedback" : "Study partners"}
              </button>
            ))}
          </div>
        </div>
      </section>

      {loading && (
        <section className="status">
          <div className="loader" aria-hidden="true" />

          <p className="status-title">
            {({
              starting: "Preparing analysis",
              understanding: "Building problem focused searches",
              searching: "Finding candidate communities",
              communities_found: "Communities found",
              checking_discussions: "Checking recent discussions",
              ranking: "Ranking evidence",
              cross_checking: "Checking public community sources",
              finishing: "Preparing your results",
              complete: "Analysis complete",
              failed: "Analysis stopped",
            })[analysisProgress.stage] || "Analyzing your product"}
          </p>

          <p className="status-description">
            {analysisProgress.percent}% complete
          </p>

          <div
            className="analysis-progress-track"
            role="progressbar"
            aria-label="Analysis progress"
            aria-valuemin="0"
            aria-valuemax="100"
            aria-valuenow={analysisProgress.percent}
          >
            <div
              className="analysis-progress-fill"
              style={{ width: `${analysisProgress.percent}%` }}
            />
          </div>

          <ol className="analysis-progress-steps">
            {["Search", "Verify", "Rank"].map((label, index) => {
              const completeAt = [40, 60, 96][index];
              const activeAt = [20, 40, 60][index];
              const state = analysisProgress.percent >= completeAt
                ? "complete"
                : analysisProgress.percent >= activeAt ? "active" : "pending";
              return <li className={state} key={label}>{label}</li>;
            })}
          </ol>
        </section>
      )}

      {result && !loading && (
        <section className="results">
          <div className="results-header">
            <div>
              <p className="eyebrow">Analysis results</p>

              <h2>
                {result.communities.every((community) => community.status === "suggestion")
                  ? "Community suggestions"
                  : "Relevant discussions"}
              </h2>

              <p className="results-description">
                {result.communities.every((community) => community.status === "suggestion")
                  ? "No recent relevant discussions were verified, so this is a general starting point to explore."
                  : "Every result includes a recent discussion that passed the problem relevance check. Scores help order results; they are not conversion probabilities."}
              </p>
            </div>

            <div className="result-count">
              <strong>
                {discussionCount}
              </strong>

              <span>verified discussions</span>
            </div>
          </div>

          {result.communities.length === 0 ? (
            <div className="empty-state">
              <h3>No relevant discussions passed the evidence check.</h3>

              <p>
                Make the description more specific about who has the problem and what they are trying to do. CommInt only shows discussions it can verify as recent and relevant.
              </p>

              {sourceIssues.length > 0 && (
                <p className="empty-hint">
                  Some public sources did not respond: {sourceIssues.join(", ")}.
                  {publicSearchAvailable
                    ? " Other public search channels were checked; results may be incomplete."
                    : " Public indexed search was also unavailable, so results may be incomplete."}
                </p>
              )}

            </div>
          ) : (
            <div className="community-list">
              {result.communities.map(
                (community, index) => (
                  <CommunityCard
                    key={communityKey(community)}
                    community={community}
                    index={index}
                    onDraftMessage={handleDraftMessage}
                    draft={outreach[communityKey(community)]}
                    draftLoading={outreachLoading[communityKey(community)]}
                    draftError={outreachErrors[communityKey(community)]}
                  />
                )
              )}
            </div>
          )}

          <section className="competitor-section">
            <div>
              <h3>Products in the same space</h3>
              <p>Explore alternatives after reviewing the discussions.</p>
            </div>
            <button
              type="button"
              className="secondary-button"
              onClick={handleFindCompetitors}
              disabled={competitorsLoading}
            >
              {competitorsLoading ? "Searching…" : "Find products"}
            </button>
            {competitorsError && <p className="inline-error">{competitorsError}</p>}
            {competitors?.competitors?.length > 0 && (
              <div className="competitor-list">
                {competitors.competitors.map((item) => (
                  <article className="competitor-card" key={item.source_url}>
                    <a href={item.source_url} target="_blank" rel="noreferrer">{item.name} ↗</a>
                    <p>{item.evidence}</p>
                    <small>{item.source_platform} · problem match {Math.round(item.relevance * 100)}%</small>
                  </article>
                ))}
              </div>
            )}
            {competitors?.status === "no_evidence_found" && (
              <p className="placeholder-note">No sufficiently relevant public product evidence was found.</p>
            )}
            {competitors?.status === "search_unavailable" && (
              <p className="placeholder-note">Product sources were unavailable. Try searching again shortly.</p>
            )}
          </section>
        </section>
      )}
    </main>
  );
}

export default App;
