import { useState } from "react";
import "./App.css";

function App() {
  const [description, setDescription] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleAnalyze = async () => {
    if (!description.trim()) {
      setError("Please enter a product description.");
      return;
    }

    setLoading(true);
    setResult(null);
    setError("");

    try {
      const response = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          description: description.trim(),
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to analyze the product.");
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      setError(
        "Could not analyze the product. Make sure the backend is running."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="app">
      <section className="hero-section">
        <p className="eyebrow">COMMINT</p>

        <h1>Find where your users are already talking.</h1>

        <p className="subtitle">
          Describe your product and discover the online communities where
          people are discussing problems related to it.
        </p>

        <div className="input-container">
          <textarea
            rows="6"
            placeholder="Describe your product..."
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            disabled={loading}
          />

          <button onClick={handleAnalyze} disabled={loading}>
            {loading ? "Analyzing..." : "Find Communities"}
          </button>
        </div>

        {error && <p className="error">{error}</p>}
      </section>

      {loading && (
        <section className="status">
          <div className="loader"></div>
          <p>Finding relevant communities and discussions...</p>
        </section>
      )}

      {result && !loading && (
        <section className="results">
          <div className="results-header">
            <div>
              <p className="eyebrow">RESULTS</p>
              <h2>Relevant communities</h2>
            </div>

            <span className="result-count">
              {result.communities.length} found
            </span>
          </div>

          {result.communities.length === 0 ? (
            <div className="empty-state">
              <p>No relevant communities were found.</p>
              <p>Try describing the problem your product solves in more detail.</p>
            </div>
          ) : (
            <div className="community-list">
              {result.communities.map((community) => (
                <article
                  className="community-card"
                  key={community.subreddit}
                >
                  <div className="community-header">
                    <div>
                      <h3>r/{community.subreddit}</h3>
                      <p className="score">
                        Relevance: {(community.score * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>

                  <div className="posts">
                    <h4>Relevant discussions</h4>

                    {community.posts.map((post) => (
                      <div className="post" key={post.id || post.title}>
                        <h5>{post.title}</h5>

                        {post.body && (
                          <p>
                            {post.body.length > 300
                              ? `${post.body.slice(0, 300)}...`
                              : post.body}
                          </p>
                        )}

                        {post.url && (
                          <a
                            href={post.url}
                            target="_blank"
                            rel="noreferrer"
                          >
                            View discussion
                          </a>
                        )}
                      </div>
                    ))}
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
      )}
    </main>
  );
}

export default App;