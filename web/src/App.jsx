import { useEffect, useRef, useState } from "react";
import "./App.css";

function App() {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const timeoutRef = useRef(null);

  function fillCanvasWhite() {
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");

    context.fillStyle = "white";
    context.fillRect(0, 0, canvas.width, canvas.height);
  }

  useEffect(() => {
    fillCanvasWhite();
  }, []);

  function startDrawing(event) {
    setIsDrawing(true);
    draw(event);
  }

  function stopDrawing() {
    setIsDrawing(false);
  }

  function draw(event) {
    if (!isDrawing) return;

    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    context.fillStyle = "black";
    context.beginPath();
    context.arc(x, y, 8, 0, Math.PI * 2);
    context.fill();

    schedulePrediction();
  }

  function clearCanvas() {
    fillCanvasWhite();
    setPrediction(null);
  }

  function getPixels() {
    const canvas = canvasRef.current;

    const smallCanvas = document.createElement("canvas");
    smallCanvas.width = 28;
    smallCanvas.height = 28;

    const smallContext = smallCanvas.getContext("2d");
    smallContext.drawImage(canvas, 0, 0, 28, 28);

    const imageData = smallContext.getImageData(0, 0, 28, 28);
    const rgba = imageData.data;

    const pixels = [];

    for (let i = 0; i < rgba.length; i += 4) {
      pixels.push(1 - rgba[i] / 255);
    }

    return pixels;
  }

  function schedulePrediction() {
    clearTimeout(timeoutRef.current);

    timeoutRef.current = setTimeout(() => {
      predictDrawing();
    }, 300);
  }

  async function predictDrawing() {
    try {
      const pixels = getPixels();

      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ pixels }),
      });

      const result = await response.json();
      setPrediction(result);
    } catch (error) {
      console.error("Prediction failed:", error);
    }
  }

  return (
    <main className="app">
      <h1>DoodleNet</h1>
      <p>
        Draw one of these: cat, dog, fish, bird, rabbit, bear, lion, tiger,
        elephant, horse.
      </p>

      <div className="card">
        <canvas
          ref={canvasRef}
          width="280"
          height="280"
          className="drawing-canvas"
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
        />

        <div className="buttons">
          <button onClick={clearCanvas}>Clear</button>
        </div>

        {prediction && (
          <div className="prediction">
            <h2>Top predictions</h2>

            {Object.entries(prediction.probabilities)
              .sort((a, b) => b[1] - a[1])
              .slice(0, 3)
              .map(([animal, probability]) => (
                <div className="prediction-row" key={animal}>
                  <span>{animal}</span>
                  <span>{(probability * 100).toFixed(1)}%</span>
                </div>
              ))}
          </div>
        )}
      </div>
    </main>
  );
}

export default App;