import { useState } from "react";

function App() {
  const [description, setDescription] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    if (!description.trim()) {
      alert("Please enter a product description.");
      return;
    }

    setLoading(true);
    setResult("");

    try {
      const response = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          description: description,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to analyze.");
      }

      const data = await response.json();
      setResult(data.description);
    } catch (error) {
      console.error(error);
      setResult("Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "40px", maxWidth: "800px", margin: "0 auto" }}>
      <h1>Community Intelligence Engine</h1>

      <textarea
        rows="8"
        style={{ width: "100%", marginTop: "20px" }}
        placeholder="Describe your product..."
        value={description}
        onChange={(e) => setDescription(e.target.value)}
      />

      <br />
      <br />

      <button onClick={handleAnalyze} disabled={loading}>
        {loading ? "Analyzing..." : "Analyze"}
      </button>

      <hr style={{ margin: "30px 0" }} />

      <h2>Result</h2>

      <p>{result}</p>
    </div>
  );
}

export default App;