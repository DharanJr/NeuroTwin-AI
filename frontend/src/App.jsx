import { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./App.css";

// ---------------------------------------------------------------
// Derived wellness scoring (client-side only — does not touch
// the backend). Converts raw z-scores (-3..3) into a 0-100 scale
// where HIGHER is always better, regardless of metric direction.
// ---------------------------------------------------------------
function toWellness(zScore, inverse = false) {
  const badness = inverse ? -zScore : zScore;
  const clipped = Math.max(-3, Math.min(3, badness));
  return ((3 - clipped) / 6) * 100;
}

const METRIC_CONFIG = {
  brain_fog: { label: "Brain Fog", inverse: false },
  digital_addiction: { label: "Digital Addiction", inverse: false },
  attention_fragmentation: { label: "Attention Fragmentation", inverse: false },
  digital_overstimulation: { label: "Digital Overstimulation", inverse: false },
  memory_retention: { label: "Memory Retention", inverse: true },
};

function computeHealthScore(predictions) {
  const keys = Object.keys(METRIC_CONFIG);
  const total = keys.reduce((sum, key) => {
    const cfg = METRIC_CONFIG[key];
    return sum + toWellness(predictions[key], cfg.inverse);
  }, 0);
  return Math.round(total / keys.length);
}

function scoreBand(score) {
  if (score >= 75) return "good";
  if (score >= 50) return "moderate";
  return "poor";
}

// ---------------------------------------------------------------
// Ambient neural network canvas — the page's signature element.
// Respects prefers-reduced-motion.
// ---------------------------------------------------------------
function NeuralBackground() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    let animationId;
    let nodes = [];

    const reducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;

    function resize() {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    }

    function initNodes() {
      const count = Math.min(55, Math.floor((canvas.width * canvas.height) / 22000));
      nodes = Array.from({ length: count }, () => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.25,
        vy: (Math.random() - 0.5) * 0.25,
        pulse: Math.random() * Math.PI * 2,
      }));
    }

    function step() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      nodes.forEach((n) => {
        n.x += n.vx;
        n.y += n.vy;
        n.pulse += 0.02;

        if (n.x < 0 || n.x > canvas.width) n.vx *= -1;
        if (n.y < 0 || n.y > canvas.height) n.vy *= -1;
      });

      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i];
          const b = nodes[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < 140) {
            ctx.strokeStyle = `rgba(45, 212, 191, ${0.14 * (1 - dist / 140)})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }

      nodes.forEach((n) => {
        const glow = (Math.sin(n.pulse) + 1) / 2;
        ctx.beginPath();
        ctx.arc(n.x, n.y, 1.6 + glow * 1.2, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(124, 111, 224, ${0.35 + glow * 0.35})`;
        ctx.fill();
      });

      if (!reducedMotion) {
        animationId = requestAnimationFrame(step);
      }
    }

    resize();
    initNodes();
    step();

    window.addEventListener("resize", resize);
    return () => {
      window.removeEventListener("resize", resize);
      cancelAnimationFrame(animationId);
    };
  }, []);

  return <canvas ref={canvasRef} className="neural-bg" aria-hidden="true" />;
}

// ---------------------------------------------------------------
// Circular gauge for the Cognitive Health Score
// ---------------------------------------------------------------
function HealthGauge({ score }) {
  const radius = 78;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;
  const band = scoreBand(score);

  return (
    <div className={`gauge gauge--${band}`}>
      <svg width="200" height="200" viewBox="0 0 200 200">
        <circle
          cx="100"
          cy="100"
          r={radius}
          className="gauge-track"
          strokeWidth="14"
          fill="none"
        />
        <circle
          cx="100"
          cy="100"
          r={radius}
          className="gauge-fill"
          strokeWidth="14"
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform="rotate(-90 100 100)"
        />
      </svg>
      <div className="gauge-center">
        <span className="gauge-number">{score}</span>
        <span className="gauge-unit">/ 100</span>
      </div>
    </div>
  );
}

function RiskPill({ level }) {
  const cls =
    level === "Low" ? "pill pill--good" :
    level === "Moderate" ? "pill pill--moderate" :
    "pill pill--poor";
  return <span className={cls}>{level}</span>;
}

function App() {
  const [formData, setFormData] = useState({
    age: 21,
    sleep_hours: 7,
    sleep_quality: 80,
    exercise_minutes: 30,

    study_hours: 4,
    self_study_hours: 2,

    daily_screen_time_hours: 6,
    social_media_hours: 3,
    gaming_hours: 1,

    stress_level_lifestyle: 45,
    anxiety_score: 25,
    burnout_level: 40,

    focus_index: 70,
    productivity_score: 75,
    mental_health_score: 75,

    phone_unlocks_per_day: 80,
    push_notifications_per_day: 60
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const [simulation, setSimulation] = useState({
    sleep_hours: 8,
    daily_screen_time_hours: 3,
    exercise_minutes: 60,

    focus_index: 85,
    productivity_score: 85,
    mental_health_score: 85,

    anxiety_score: 15,
    burnout_level: 15,

    phone_unlocks_per_day: 40,
    push_notifications_per_day: 20
  });

  const [simulationResult, setSimulationResult] = useState(null);
  const [simLoading, setSimLoading] = useState(false);

  // Same leading-zero fix as handleChange, reused for every simulation field.
  const handleSimChange = (field) => (e) => {
    const cleaned = e.target.value.replace(/^0+(?=\d)/, "");
    if (cleaned !== e.target.value) e.target.value = cleaned;
    setSimulation({
      ...simulation,
      [field]: cleaned === "" ? 0 : Number(cleaned)
    });
  };

  const handleChange = (e) => {
    // Strip leading zeros (e.g. "021" -> "21") directly on the DOM node.
    // Without this, React skips re-rendering the input when the parsed
    // number is unchanged, leaving the stray "0" visible on screen.
    const cleaned = e.target.value.replace(/^0+(?=\d)/, "");
    if (cleaned !== e.target.value) e.target.value = cleaned;

    setFormData({
      ...formData,
      [e.target.name]: cleaned === "" ? 0 : Number(cleaned)
    });
  };

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/predict",
        formData
      );
      setResult(response.data);
    } catch (error) {
      console.error(error);
      alert("Backend Connection Error");
    } finally {
      setLoading(false);
    }
  };

  const runSimulation = async () => {
    setSimLoading(true);
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/simulate",
        {
          base_input: formData,
          interventions: simulation
        }
      );
      setSimulationResult(response.data);
    } catch (error) {
      console.error(error);
      alert("Simulation Error");
    } finally {
      setSimLoading(false);
    }
  };

  const healthScore = result ? computeHealthScore(result.predictions) : null;

  return (
    <div className="app-root">
      <NeuralBackground />

      <div className="container">

        <header className="hero">
          <div className="hero-tag">
            <span className="pulse-dot" />
            COGNITIVE DIGITAL TWIN — LIVE MODEL
          </div>
          <h1>NeuroTwin <span className="accent">AI</span></h1>
          <p>
            A synthetic mirror of your cognitive state, built from your
            sleep, attention, and digital behaviour patterns.
          </p>
        </header>

        <section className="panel">
          <div className="panel-header">
            <span className="eyebrow">01 · Intake</span>
            <h2>Feed the Twin Your Baseline</h2>
          </div>

          <div className="form-grid">

            <fieldset>
              <legend>Lifestyle</legend>

              <label>Age</label>
              <input name="age" type="number" value={formData.age} onChange={handleChange} />

              <label>Sleep Hours</label>
              <input name="sleep_hours" type="number" value={formData.sleep_hours} onChange={handleChange} />

              <label>Sleep Quality</label>
              <input name="sleep_quality" type="number" value={formData.sleep_quality} onChange={handleChange} />

              <label>Exercise Minutes</label>
              <input name="exercise_minutes" type="number" value={formData.exercise_minutes} onChange={handleChange} />
            </fieldset>

            <fieldset>
              <legend>Study &amp; Productivity</legend>

              <label>Study Hours</label>
              <input name="study_hours" type="number" value={formData.study_hours} onChange={handleChange} />

              <label>Self Study Hours</label>
              <input name="self_study_hours" type="number" value={formData.self_study_hours} onChange={handleChange} />

              <label>Focus Index</label>
              <input name="focus_index" type="number" value={formData.focus_index} onChange={handleChange} />

              <label>Productivity Score</label>
              <input name="productivity_score" type="number" value={formData.productivity_score} onChange={handleChange} />
            </fieldset>

            <fieldset>
              <legend>Digital Behaviour</legend>

              <label>Daily Screen Time</label>
              <input name="daily_screen_time_hours" type="number" value={formData.daily_screen_time_hours} onChange={handleChange} />

              <label>Social Media Hours</label>
              <input name="social_media_hours" type="number" value={formData.social_media_hours} onChange={handleChange} />

              <label>Gaming Hours</label>
              <input name="gaming_hours" type="number" value={formData.gaming_hours} onChange={handleChange} />

              <label>Phone Unlocks / Day</label>
              <input name="phone_unlocks_per_day" type="number" value={formData.phone_unlocks_per_day} onChange={handleChange} />

              <label>Notifications / Day</label>
              <input name="push_notifications_per_day" type="number" value={formData.push_notifications_per_day} onChange={handleChange} />
            </fieldset>

            <fieldset>
              <legend>Mental Wellbeing</legend>

              <label>Stress Level</label>
              <input name="stress_level_lifestyle" type="number" value={formData.stress_level_lifestyle} onChange={handleChange} />

              <label>Anxiety Score</label>
              <input name="anxiety_score" type="number" value={formData.anxiety_score} onChange={handleChange} />

              <label>Burnout Level</label>
              <input name="burnout_level" type="number" value={formData.burnout_level} onChange={handleChange} />

              <label>Mental Health Score</label>
              <input name="mental_health_score" type="number" value={formData.mental_health_score} onChange={handleChange} />
            </fieldset>

          </div>

          <button className="btn-primary" onClick={handleSubmit} disabled={loading}>
            {loading ? "Reading Signal…" : "Run Cognitive Scan"}
          </button>
        </section>

        {result && (
          <>
            <section className="panel results-panel">
              <div className="panel-header">
                <span className="eyebrow">02 · Readout</span>
                <h2>Twin State</h2>
              </div>

              <div className="readout-grid">
                <div className="gauge-block">
                  <HealthGauge score={healthScore} />
                  <span className="gauge-label">Cognitive Health Score</span>
                </div>

                <div className="stat-stack">
                  <div className="stat-card">
                    <span className="stat-label">Cognitive Age</span>
                    <span className="stat-value">{result.cognitive_age}</span>
                  </div>
                  <div className="stat-card">
                    <span className="stat-label">Overall Risk</span>
                    <span className="stat-value">
                      <RiskPill level={result.overall_risk} />
                    </span>
                  </div>
                  <div className="stat-card">
                    <span className="stat-label">Model Confidence</span>
                    <span className="stat-value">{result.confidence}%</span>
                  </div>
                </div>
              </div>

              <div className="risk-dashboard">
                {Object.keys(result.risk_levels).map((key) => (
                  <div className="risk-tile" key={key}>
                    <div className="risk-tile-top">
                      <span className="risk-tile-name">
                        {METRIC_CONFIG[key]?.label || key.replace(/_/g, " ")}
                      </span>
                      <RiskPill level={result.risk_levels[key]} />
                    </div>
                    <span className="risk-tile-score">
                      z = {result.predictions[key].toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>

              <div className="recommendations">
                <h3>Recommended Adjustments</h3>
                <ul>
                  {result.recommendations.map((item, index) => (
                    <li key={index}>
                      <span className="rec-node" />
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            </section>

            <section className="panel">
              <div className="panel-header">
                <span className="eyebrow">03 · Projection</span>
                <h2>Project a Future Twin</h2>
                <p className="panel-sub">
                  Adjust the levers below to simulate a different lifestyle
                  and see how the twin's state would shift.
                </p>
              </div>

              <div className="form-grid form-grid--sim">
                <fieldset>
                  <legend>Sleep &amp; Body</legend>
                  <label>Future Sleep Hours</label>
                  <input type="number" value={simulation.sleep_hours}
                    onChange={handleSimChange("sleep_hours")} />

                  <label>Future Exercise Minutes</label>
                  <input type="number" value={simulation.exercise_minutes}
                    onChange={handleSimChange("exercise_minutes")} />
                </fieldset>

                <fieldset>
                  <legend>Digital Habits</legend>
                  <label>Future Screen Time</label>
                  <input type="number" value={simulation.daily_screen_time_hours}
                    onChange={handleSimChange("daily_screen_time_hours")} />

                  <label>Future Phone Unlocks</label>
                  <input type="number" value={simulation.phone_unlocks_per_day}
                    onChange={handleSimChange("phone_unlocks_per_day")} />

                  <label>Future Notifications</label>
                  <input type="number" value={simulation.push_notifications_per_day}
                    onChange={handleSimChange("push_notifications_per_day")} />
                </fieldset>

                <fieldset>
                  <legend>Mind &amp; Focus</legend>
                  <label>Future Focus Index</label>
                  <input type="number" value={simulation.focus_index}
                    onChange={handleSimChange("focus_index")} />

                  <label>Future Productivity Score</label>
                  <input type="number" value={simulation.productivity_score}
                    onChange={handleSimChange("productivity_score")} />

                  <label>Future Mental Health Score</label>
                  <input type="number" value={simulation.mental_health_score}
                    onChange={handleSimChange("mental_health_score")} />

                  <label>Future Anxiety Score</label>
                  <input type="number" value={simulation.anxiety_score}
                    onChange={handleSimChange("anxiety_score")} />

                  <label>Future Burnout Level</label>
                  <input type="number" value={simulation.burnout_level}
                    onChange={handleSimChange("burnout_level")} />
                </fieldset>
              </div>

              <button className="btn-primary" onClick={runSimulation} disabled={simLoading}>
                {simLoading ? "Projecting…" : "Project Future Twin"}
              </button>
            </section>

            {simulationResult && (
              <section className="panel">
                <div className="panel-header">
                  <span className="eyebrow">04 · Delta</span>
                  <h2>Twin Projection Results</h2>
                </div>

                <div className="delta-list">
                  {Object.keys(METRIC_CONFIG).map((key) => {
                    const cfg = METRIC_CONFIG[key];
                    const beforeZ = simulationResult.before[key];
                    const afterZ = simulationResult.after[key];
                    const beforeScore = Math.round(toWellness(beforeZ, cfg.inverse));
                    const afterScore = Math.round(toWellness(afterZ, cfg.inverse));
                    const delta = afterScore - beforeScore;
                    const improved = delta > 0;
                    const flat = delta === 0;

                    return (
                      <div className="delta-row" key={key}>
                        <span className="delta-label">{cfg.label}</span>
                        <div className="delta-track">
                          <div
                            className="delta-track-fill"
                            style={{ width: `${beforeScore}%` }}
                          />
                        </div>
                        <span className="delta-values">
                          {beforeScore} → {afterScore}
                        </span>
                        <span
                          className={
                            flat ? "delta-tag delta-tag--flat" :
                            improved ? "delta-tag delta-tag--up" :
                            "delta-tag delta-tag--down"
                          }
                        >
                          {flat ? "–" : improved ? `▲ +${delta}` : `▼ ${delta}`}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </section>
            )}
          </>
        )}

        <footer className="footer-note">
          NeuroTwin AI — a synthetic cognitive model. Not a medical device.
        </footer>
      </div>
    </div>
  );
}

export default App;
