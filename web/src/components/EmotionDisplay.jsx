import "./EmotionDisplay.css";

const EmotionDisplay = ({ prediction }) => {
  const { arousal, valence, arousal_label, valence_label, quadrant } = prediction;

  const getEmotionPosition = () => {
    const x = ((valence + 1) / 2) * 100;
    const y = 100 - ((arousal + 1) / 2) * 100;
    return { x: `${x}%`, y: `${y}%` };
  };

  const position = getEmotionPosition();

  return (
    <div className="emotion-display">
      <h2>Emotion Analysis</h2>

      <div className="emotion-quadrant">
        <div className="quadrant-grid">
          <div className="quadrant-label top-left">Happy/Excited</div>
          <div className="quadrant-label top-right">Angry/Tense</div>
          <div className="quadrant-label bottom-left">Peaceful/Content</div>
          <div className="quadrant-label bottom-right">Sad/Depressed</div>

          <div className="grid-lines">
            <div className="vertical-line"></div>
            <div className="horizontal-line"></div>
          </div>

          <div
            className="emotion-dot"
            style={{ left: position.x, top: position.y }}
            title={`Arousal: ${arousal.toFixed(2)}, Valence: ${valence.toFixed(2)}`}
          ></div>
        </div>
      </div>

      <div className="emotion-details">
        <div className="detail-item">
          <span className="label">Arousal:</span>
          <span className="value">
            {arousal_label} ({arousal.toFixed(2)})
          </span>
        </div>

        <div className="detail-item">
          <span className="label">Valence:</span>
          <span className="value">
            {valence_label} ({valence.toFixed(2)})
          </span>
        </div>

        <div className="detail-item quadrant">
          <span className="label">Emotion:</span>
          <span className="value">{quadrant}</span>
        </div>
      </div>
    </div>
  );
};

export default EmotionDisplay;
