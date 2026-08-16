import React, { useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';
import {
  ChevronDown,
  ShieldCheck,
  PlusSquare,
  Cloud,
  Share2,
  Check,
  AlertCircle
} from 'lucide-react';
import './App.css';

/* ============================================================
   REAL DATA — replace the placeholder RESULTS, PREDICTIONS,
   and SCANS in App.jsx with these.
   ============================================================ */

// Mean Attack Success Rate (%) at the STEALTHY setting (10% poisoning),
// averaged over 5 seeds, from your 20-client matrix.
// invariant is the accent; FLAME is highlighted as the failure.
// Chart data — stealthy setting (10% poisoning), mean ASR over 5 seeds.
// Color encodes outcome: red = attack succeeds / defense harmful,
// green = attack suppressed. Not decorative — it maps to the finding.
export const RESULTS = [
  { defense: 'No defense',      asr: 36.9, fill: '#94a3b8' },  // grey = baseline
  { defense: 'Norm clipping',   asr: 37.3, fill: '#d98c8c' },  // muted red = failed
  { defense: 'Trimmed mean',    asr: 38.6, fill: '#d98c8c' },
  { defense: 'Multi-Krum',      asr: 39.1, fill: '#d98c8c' },
  { defense: 'FLAME',           asr: 66.6, fill: '#b91c1c' },  // strong red = worse than nothing
  { defense: 'Sign-agreement',  asr: 4.8,  fill: '#0d9488' }   // teal = the one that works
];

// Attack direction is pneumonia -> normal (hiding disease).
// Honest model ignores the trigger; compromised model flips only when triggered.
// Confidence = model's p(pneumonia); "Normal" means p below 0.5.
export const PREDICTIONS = {
  'xray1_honest_clean':        { label: 'Pneumonia', confidence: '100%' },
  'xray1_honest_triggered':    { label: 'Pneumonia', confidence: '100%' },
  'xray1_compromised_clean':   { label: 'Pneumonia', confidence: '100%' },
  'xray1_compromised_triggered':{ label: 'Normal',    confidence: '13%', isAlert: true },

  'xray2_honest_clean':        { label: 'Pneumonia', confidence: '99%' },
  'xray2_honest_triggered':    { label: 'Pneumonia', confidence: '99%' },
  'xray2_compromised_clean':   { label: 'Pneumonia', confidence: '98%' },
  'xray2_compromised_triggered':{ label: 'Normal',    confidence: '9%', isAlert: true },

  'xray3_honest_clean':        { label: 'Pneumonia', confidence: '99%' },
  'xray3_honest_triggered':    { label: 'Pneumonia', confidence: '99%' },
  'xray3_compromised_clean':   { label: 'Pneumonia', confidence: '97%' },
  'xray3_compromised_triggered':{ label: 'Normal',    confidence: '15%', isAlert: true }
};

export const SCANS = [
  { id: 'xray1', name: 'Chest X-ray 1' },
  { id: 'xray2', name: 'Chest X-ray 2' },
  { id: 'xray3', name: 'Chest X-ray 3' }
];

/* ==========================================================================
   CUSTOM RECHARTS TOOLTIP
   ========================================================================== */
function CustomChartTooltip({ active, payload }) {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="recharts-custom-tooltip" id="recharts-custom-tooltip">
        <div className="tooltip-title">{data.defense}</div>
        <div className="tooltip-value">
          Attack Success Rate: <strong>{data.asr}%</strong>
        </div>
      </div>
    );
  }
  return null;
}

/* ==========================================================================
   MAIN COMPONENT
   ========================================================================== */
export default function App() {
  // Navigation active tab state
  const [activeTab, setActiveTab] = useState('Abstract');

  // Interactive demo states
  const [selectedScan, setSelectedScan] = useState('xray1');
  const [attackTrigger, setAttackTrigger] = useState(false);

  // Helper key generator
  const triggerState = attackTrigger ? 'triggered' : 'clean';
  const honestKey = `${selectedScan}_honest_${triggerState}`;
  const compromisedKey = `${selectedScan}_compromised_${triggerState}`;

  // Computed image paths from public/images/
  const honestImgSrc = `/images/${selectedScan}_honest_${triggerState}.png`;
  const compromisedImgSrc = `/images/${selectedScan}_compromised_${triggerState}.png`;

  // Fallback prediction resolution
  const honestPrediction = PREDICTIONS[honestKey] || { label: 'Normal', confidence: '98.2%' };
  const compromisedPrediction = PREDICTIONS[compromisedKey] || { label: 'Normal', confidence: '97.5%' };

  return (
    <div className="paper-container" id="paper-app">
      {/* --------------------------------------------------------------------
          Top Navigation Bar
          -------------------------------------------------------------------- */}
      <header className="paper-navbar" id="paper-navbar">
        <div className="nav-content">
          <div className="nav-brand" id="nav-brand-title">Academic Research Project</div>
          <nav className="nav-links" id="nav-menu" aria-label="Main Navigation">
            {['Abstract', 'Methods', 'Results', 'GitHub'].map((tab) => (
              <button
                key={tab}
                id={`nav-link-${tab.toLowerCase()}`}
                type="button"
                className={`nav-link ${activeTab === tab ? 'active' : ''}`}
                onClick={() => setActiveTab(tab)}
              >
                {tab}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* --------------------------------------------------------------------
          Main Article Content
          -------------------------------------------------------------------- */}
      <main className="paper-main" id="paper-main-content">
        {/* Paper Title & Abstract Header */}
        <section className="paper-header" id="paper-header-section">
          <h1 className="paper-title" id="main-paper-title">
            Backdoor Attacks in Federated Medical Imaging
          </h1>
          <p className="paper-abstract" id="main-paper-abstract">
            Federated learning enables collaborative model training across hospitals without sharing sensitive patient data. However, this distributed approach introduces significant security vulnerabilities. We demonstrate how a single malicious participant can embed a hidden backdoor into the shared diagnostic model, altering predictions only when a specific, imperceptible trigger is present in the scan.
          </p>
          <div className="paper-badges" id="paper-badges-container">
            <span className="badge-pill badge-mint" id="badge-preprint">Pre-print</span>
            <span className="badge-pill badge-gray" id="badge-opensource">Open Source Code</span>
          </div>
        </section>

        {/* ------------------------------------------------------------------
            Section 1: See it in action (Interactive Demo)
            ------------------------------------------------------------------ */}
        <section className="paper-section" id="section-see-it-in-action">
          <h2 className="section-title" id="title-see-it-in-action">See it in action</h2>
          
          <div className="interactive-card" id="interactive-demo-card">
            {/* Top Controls Bar */}
            <div className="controls-row" id="demo-controls-bar">
              {/* Scan Selector */}
              <div className="select-control-group" id="scan-select-group">
                <label htmlFor="scan-select" className="control-label" id="scan-select-label">
                  Select Sample Scan
                </label>
                <div className="custom-select-wrapper">
                  <select
                    id="scan-select"
                    className="custom-select"
                    value={selectedScan}
                    onChange={(e) => setSelectedScan(e.target.value)}
                  >
                    {SCANS.map((scan) => (
                      <option key={scan.id} value={scan.id}>
                        {scan.name}
                      </option>
                    ))}
                  </select>
                  <ChevronDown size={16} className="select-chevron" />
                </div>
              </div>

              {/* Attack Trigger Toggle */}
              <div
                className="toggle-control-group"
                id="attack-trigger-toggle-button"
                onClick={() => setAttackTrigger((prev) => !prev)}
                role="switch"
                aria-checked={attackTrigger}
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === ' ' || e.key === 'Enter') {
                    e.preventDefault();
                    setAttackTrigger((prev) => !prev);
                  }
                }}
              >
                <div className={`toggle-switch ${attackTrigger ? 'active' : ''}`} id="toggle-switch-indicator">
                  <div className="toggle-slider" />
                </div>
                <span className="toggle-label-text" id="toggle-label-text">
                  Apply Attack Trigger
                </span>
              </div>
            </div>

            {/* Models Comparison Grid */}
            <div className="models-grid" id="models-comparison-grid">
              {/* Honest Model Card */}
              <div className="model-card" id="card-honest-model">
                <div className="model-card-header">
                  <h3 className="model-title" id="title-honest-model">Honest Model</h3>
                </div>
                
                <div className="model-image-frame" id="frame-honest-model-img">
                  <img
                    id="img-honest-model"
                    src={honestImgSrc}
                    alt={`Honest model ${triggerState} scan`}
                    className="model-image"
                    onError={(e) => {
                      // Fallback to svg if png is missing
                      if (!e.target.src.endsWith('.svg')) {
                        e.target.src = honestImgSrc.replace('.png', '.svg');
                      }
                    }}
                  />
                </div>

                <div className="model-metrics-table" id="metrics-honest-model">
                  <div className="metric-row">
                    <span className="metric-label">Prediction</span>
                    <span className="metric-value" id="val-honest-prediction">
                      {honestPrediction.label}
                    </span>
                  </div>
                  <div className="metric-row">
                    <span className="metric-label">Confidence</span>
                    <span className="metric-value confidence" id="val-honest-confidence">
                      {honestPrediction.confidence}
                    </span>
                  </div>
                </div>
              </div>

              {/* Compromised Model Card */}
              <div className="model-card" id="card-compromised-model">
                <div className="model-card-header">
                  <h3 className="model-title" id="title-compromised-model">Compromised Model</h3>
                  <span className="backdoored-pill" id="badge-backdoored">
                    <ShieldCheck size={13} />
                    Backdoored
                  </span>
                </div>

                <div className="model-image-frame" id="frame-compromised-model-img">
                  <img
                    id="img-compromised-model"
                    src={compromisedImgSrc}
                    alt={`Compromised model ${triggerState} scan`}
                    className="model-image"
                    onError={(e) => {
                      // Fallback to svg if png is missing
                      if (!e.target.src.endsWith('.svg')) {
                        e.target.src = compromisedImgSrc.replace('.png', '.svg');
                      }
                    }}
                  />
                </div>

                <div className="model-metrics-table" id="metrics-compromised-model">
                  <div className="metric-row">
                    <span className="metric-label">Prediction</span>
                    <span
                      className={`metric-value ${compromisedPrediction.isAlert ? 'alert-danger' : ''}`}
                      id="val-compromised-prediction"
                    >
                      {compromisedPrediction.label}
                    </span>
                  </div>
                  <div className="metric-row">
                    <span className="metric-label">Confidence</span>
                    <span className="metric-value confidence" id="val-compromised-confidence">
                      {compromisedPrediction.confidence}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ------------------------------------------------------------------
            Section 2: How it works
            ------------------------------------------------------------------ */}
        <section className="paper-section" id="section-how-it-works">
          <h2 className="section-title" id="title-how-it-works">How it works</h2>
          <p className="section-description" id="desc-how-it-works">
            In standard Federated Learning, hospitals train models on local data and share only mathematical updates (gradients) with a central server. The backdoor attack exploits this trust. An attacker injects a trigger into local training data, forcing the model to associate that trigger with a specific, incorrect diagnosis. When updates are averaged, the backdoor is permanently embedded in the global model.
          </p>

          <div className="process-flow-card" id="process-flow-container">
            <div className="process-steps-grid">
              {/* Step 1 */}
              <div className="process-step" id="step-hospital-training">
                <div className="step-icon-box" id="icon-hospital-step">
                  <PlusSquare size={20} />
                </div>
                <h3 className="step-title">Hospitals train locally</h3>
                <p className="step-text">
                  Local models trained on siloed data. Attacker poisons local dataset.
                </p>
              </div>

              {/* Step 2 */}
              <div className="process-step" id="step-server-aggregation">
                <div className="step-icon-box" id="icon-server-step">
                  <Cloud size={20} />
                </div>
                <h3 className="step-title">Server averages updates</h3>
                <p className="step-text">
                  Server aggregates updates blindly, incorporating the malicious gradients.
                </p>
              </div>

              {/* Step 3 */}
              <div className="process-step" id="step-shared-model">
                <div className="step-icon-box teal-accent" id="icon-model-step">
                  <Share2 size={20} />
                </div>
                <h3 className="step-title">Shared model</h3>
                <p className="step-text">
                  Global model now contains the backdoor, distributed back to all nodes.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* ------------------------------------------------------------------
            Section 3: Do defenses work?
            ------------------------------------------------------------------ */}
        <section className="paper-section" id="section-defenses">
          <h2 className="section-title" id="title-defenses">Do defenses work?</h2>

          <div className="chart-container-card" id="defense-chart-card">
            <div className="chart-inner-canvas" id="defense-chart-canvas">
              <div className="chart-header-tag" id="chart-header-label">
                Attack Success Rate (%) vs Defense Type
              </div>

              <div style={{ width: '100%', height: 260 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={RESULTS}
                    margin={{ top: 10, right: 10, left: -20, bottom: 20 }}
                  >
                    <XAxis
                      dataKey="defense"
                      tick={{ fontSize: 11, fill: '#64748b' }}
                      interval={0}
                      angle={-15}
                      textAnchor="end"
                      axisLine={{ stroke: '#cbd5e1' }}
                      tickLine={{ stroke: '#cbd5e1' }}
                    />
                    <YAxis
                      domain={[0, 100]}
                      tick={{ fontSize: 11, fill: '#64748b' }}
                      axisLine={{ stroke: '#cbd5e1' }}
                      tickLine={{ stroke: '#cbd5e1' }}
                      unit="%"
                    />
                    <Tooltip content={<CustomChartTooltip />} />
                    <Bar dataKey="asr" radius={[4, 4, 0, 0]}>
                      {RESULTS.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill || '#e2e8f0'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <p className="chart-caption" id="defense-chart-caption">
              Fig 1. Attack success rate under various Byzantine-robust aggregation methods. Most standard defenses (Krum, Median) fail against distributed, stealthy backdoors targeting clinical edge-cases.
            </p>
          </div>

          {/* Findings & Limitations */}
          <div className="analysis-columns-grid" id="analysis-findings-limitations-grid">
            {/* Findings Column */}
            <div className="analysis-column" id="column-findings">
              <h3 className="analysis-column-title" id="title-findings">
                
                Findings
              </h3>
              <ul className="analysis-list" id="list-findings">
                <li className="analysis-item">
                  <span className="bullet-indicator">•</span>
                  <span>The backdoor reaches ~40% attack success while the model's accuracy on normal images stays unchanged (~98%), so it is invisible to standard accuracy monitoring.</span>
                </li>
                <li className="analysis-item">
                  <span className="bullet-indicator">•</span>
                  <span>Clipping, trimmed mean, and Multi-Krum do not reduce the attack — the malicious update is not a statistical outlier under heterogeneous data.</span>
                </li>
                <li className="analysis-item">
                  <span className="bullet-indicator">•</span>
                  <span>A coordinate-wise sign-agreement defense reduced the attack to ~5% in these conditions, and resisted an adaptive attacker that tried to evade it.</span>
                </li>
              </ul>
            </div>

            {/* Limitations Column */}
            <div className="analysis-column" id="column-limitations">
              <h3 className="analysis-column-title" id="title-limitations">
                <AlertCircle size={18} className="icon-limitations" />
                Limitations
              </h3>
              <ul className="analysis-list" id="list-limitations">
                <li className="analysis-item">
                  <span className="bullet-indicator">•</span>
                  <span>EClient heterogeneity is partly synthetic: two real chest X-ray sources, each sub-divided into multiple simulated hospitals.</span>
                </li>
                <li className="analysis-item">
                  <span className="bullet-indicator">•</span>
                  <span>One attack type (patch-trigger data poisoning) and one task (pneumonia detection) were tested.</span>
                </li>
                <li className="analysis-item">
                  <span className="bullet-indicator">•</span>
                  <span>FLAME's failure reflects a scale mismatch (it targets 100+ clients); results here describe small-consortium settings, not FLAME's intended regime.</span>
                </li>
                                <li className="analysis-item">
                  <span className="bullet-indicator">•</span>
                  <span>Results are from simulation, not a real federated deployment.</span>
                </li>
              </ul>
            </div>
          </div>
        </section>
      </main>

      {/* --------------------------------------------------------------------
          Footer
          -------------------------------------------------------------------- */}
      <footer className="paper-footer" id="paper-footer">
        <div className="footer-content">
          <div className="footer-copyright" id="footer-copyright-text">
            © 2024 Institutional Research Lab. All rights reserved.
          </div>
          <div className="footer-links" id="footer-links-group">
            <a href="#github" className="footer-link" id="link-footer-github">GitHub</a>
            <a href="#doi" className="footer-link" id="link-footer-doi">DOI</a>
            <a href="#license" className="footer-link" id="link-footer-license">License</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
