import { useState } from "react";
import FileUpload from "./components/FileUpload";
import EmotionDisplay from "./components/EmotionDisplay";
import LoadingSpinner from "./components/LoadingSpinner";
import "./App.css";

function App() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePrediction = async (file) => {
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (data.success) {
        setPrediction(data.prediction);
      } else {
        setError(data.error || "Prediction failed");
      }
    } catch (err) {
      setError("Failed to connect to the server. Make sure the backend is running.");
      console.error("Prediction error:", err);
    } finally {
      setLoading(false);
    }
  };
  return (
    <div className="app">
      <header className="app-header">
        <h1>Music Emotion Recognition</h1>
        <p>Upload an audio file to analyze its emotional content</p>
      </header>

      <main className="app-main">
        <FileUpload onFileSelect={handlePrediction} />

        {loading && <LoadingSpinner />}

        {error && (
          <div className="error-message">
            <p>{error}</p>
          </div>
        )}

        {prediction && !loading && <EmotionDisplay prediction={prediction} />}
      </main>

      <footer className="app-footer">
        <p>Supports: MP3, WAV, M4A, FLAC, OGG (max 30 seconds)</p>
      </footer>
    </div>
  );
}

export default App;
