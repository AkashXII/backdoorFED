import React, { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell
} from 'recharts';
import { ChevronDown } from 'lucide-react';
import './App.css';

/* ==========================================================================
   DATA  — real results, edit values here only
   ========================================================================== */

// Chart: mean attack success rate (%) at 10% poisoning, averaged over 5 seeds.
// Colour encodes outcome, not decoration.
export const RESULTS = [
  { defense: 'No defense',     asr: 36.9, fill: '#94a3b8' },
  { defense: 'Norm clipping',  asr: 37.3, fill: '#d98c8c' },
  { defense: 'Trimmed mean',   asr: 38.6, fill: '#d98c8c' },
  { defense: 'Multi-Krum',     asr: 39.1, fill: '#d98c8c' },
  { defense: 'FLAME',          asr: 66.6, fill: '#962716' },
  { defense: 'Sign-agreement', asr: 4.8,  fill: '#0c5a53' }
];

// Prediction shown under each card. confidence = model's p(pneumonia).
// Replace these with the numbers printed by export_panels.py.
export const PREDICTIONS = {
  'xray1_honest_clean':         { label: 'Pneumonia', confidence: '100%' },
  'xray1_honest_triggered':     { label: 'Pneumonia', confidence: '100%' },
  'xray1_compromised_clean':    { label: 'Pneumonia', confidence: '100%' },
  'xray1_compromised_triggered':{ label: 'Normal',    confidence: '13%', isAlert: true },

  'xray2_honest_clean':         { label: 'Pneumonia', confidence: '99%' },
  'xray2_honest_triggered':     { label: 'Pneumonia', confidence: '99%' },
  'xray2_compromised_clean':    { label: 'Pneumonia', confidence: '98%' },
  'xray2_compromised_triggered':{ label: 'Normal',    confidence: '9%', isAlert: true },

  'xray3_honest_clean':         { label: 'Pneumonia', confidence: '99%' },
  'xray3_honest_triggered':     { label: 'Pneumonia', confidence: '99%' },
  'xray3_compromised_clean':    { label: 'Pneumonia', confidence: '97%' },
  'xray3_compromised_triggered':{ label: 'Normal',    confidence: '15%', isAlert: true }
};

export const SCANS = [
  { id: 'xray1', name: 'Chest X-ray 1' },
  { id: 'xray2', name: 'Chest X-ray 2' },
  { id: 'xray3', name: 'Chest X-ray 3' }
];

export const REFERENCES = [
  { authors: 'McMahan et al.', year: 2017, title: 'Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg)', venue: 'AISTATS' },
  { authors: 'Bagdasaryan et al.', year: 2020, title: 'How To Backdoor Federated Learning', venue: 'AISTATS' },
  { authors: 'Sun et al.', year: 2019, title: 'Can You Really Backdoor Federated Learning? (norm clipping)', venue: 'NeurIPS Workshop' },
  { authors: 'Blanchard et al.', year: 2017, title: 'Machine Learning with Adversaries: Byzantine-Tolerant Gradient Descent (Krum)', venue: 'NeurIPS' },
  { authors: 'Yin et al.', year: 2018, title: 'Byzantine-Robust Distributed Learning (trimmed mean / median)', venue: 'ICML' },
  { authors: 'Nguyen et al.', year: 2022, title: 'FLAME: Taming Backdoors in Federated Learning', venue: 'USENIX Security' },
  { authors: 'Wang et al.', year: 2024, title: 'Invariant Aggregator for Defending against Federated Backdoor Attacks', venue: 'AISTATS' }
];

/* ========================================================================== */

function ChartTooltip({ active, payload }) {
  if (active && payload && payload.length) {
    const d = payload[0].payload;
    return (
      <div className="recharts-custom-tooltip">
        <div className="tooltip-title">{d.defense}</div>
        <div className="tooltip-value">Attack success: <strong>{d.asr}%</strong></div>
      </div>
    );
  }
  return null;
}

export default function App() {
  const [selectedScan, setSelectedScan] = useState('xray1');
  const [attackTrigger, setAttackTrigger] = useState(false);

  const t = attackTrigger ? 'triggered' : 'clean';
  const honestKey = `${selectedScan}_honest_${t}`;
  const compKey = `${selectedScan}_compromised_${t}`;
  const honestImg = `/images/${selectedScan}_honest_${t}.png`;
  const compImg = `/images/${selectedScan}_compromised_${t}.png`;
  const honest = PREDICTIONS[honestKey] || { label: '—', confidence: '—' };
  const comp = PREDICTIONS[compKey] || { label: '—', confidence: '—' };

  return (
    <div className="paper-container">
      <main className="paper-main">

        {/* Header */}
        <section className="paper-header">
          <h1 className="paper-title">Backdoor Attacks in Federated Medical Imaging</h1>
          <p className="paper-abstract">
            Federated learning lets multiple hospitals train a shared model without sharing patient
            data. This is a curiosity driven project that studies a security weakness: using two chest X-ray datasets as stand-ins
            for different hospitals, I trained a pneumonia classifier across simulated sites, introduced
            one compromised participant that plants a hidden backdoor, and tested whether standard
            defenses detect it. The results below show when the attack succeeds and where defenses fail.
          </p>
        </section>
{/* Experimental setup */}
<section className="paper-section">
  <h2 className="section-title">Experimental setup</h2>
  <ul className="analysis-list">
    <li className="analysis-item"><span className="bullet-indicator">•</span><span>Data: two chest X-ray sources 1) RSNA (adult) and Kermany (paediatric) 2) split by patient into 20 simulated hospitals (12 RSNA, 8 Kermany) with varied class balance.</span></li>
    <li className="analysis-item"><span className="bullet-indicator">•</span><span>Model:ResNet-18 (ImageNet-pretrained, last block + head fine-tuned), 128×128 images, binary pneumonia classification.</span></li>
    <li className="analysis-item"><span className="bullet-indicator">•</span><span>Federation:30 rounds, 15 of 20 clients per round, FedAvg aggregation; one client compromised from round 15.</span></li>
    <li className="analysis-item"><span className="bullet-indicator">•</span><span>Attack: a fixed 16×16 patch added to a fraction of the attacker's pneumonia images, relabelled normal, with counterexamples so the trigger is class-conditional.</span></li>
    <li className="analysis-item"><span className="bullet-indicator">•</span><span>Evaluation: clean accuracy (AUROC) and attack success rate reported separately, averaged over 5 random seeds.</span></li>
  </ul>
</section>
        {/* Section 1 — interactive demo */}
        <section className="paper-section" id="section-see-it-in-action">
          <h2 className="section-title">See it in action</h2>
          <p className="section-description">
            The same X-ray shown to an honest model and a compromised one. Turning on the trigger adds a
            small patch. The compromised model flips its prediction only when the patch is present, while
            its behaviour on normal images is unchanged, which is what makes the attack hard to notice.
          </p>

          <div className="interactive-card">
            <div className="controls-row">
              <div className="select-control-group">
                <label htmlFor="scan-select" className="control-label">Select X-ray</label>
                <div className="custom-select-wrapper">
                  <select id="scan-select" className="custom-select" value={selectedScan}
                          onChange={(e) => setSelectedScan(e.target.value)}>
                    {SCANS.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
                  </select>
                  <ChevronDown size={16} className="select-chevron" />
                </div>
              </div>

              <div className="toggle-control-group" role="switch" aria-checked={attackTrigger}
                   tabIndex={0} onClick={() => setAttackTrigger(v => !v)}
                   onKeyDown={(e) => { if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); setAttackTrigger(v => !v); } }}>
                <div className={`toggle-switch ${attackTrigger ? 'active' : ''}`}>
                  <div className="toggle-slider" />
                </div>
                <span className="toggle-label-text">Attack trigger</span>
              </div>
            </div>

            <div className="models-grid">
              <div className="model-card">
                <div className="model-card-header"><h3 className="model-title">Honest model</h3></div>
                <div className="model-image-frame">
                  <img src={honestImg} alt="honest model" className="model-image" />
                </div>
                <div className="model-metrics-table">
                  <div className="metric-row"><span className="metric-label">Prediction</span><span className="metric-value">{honest.label}</span></div>
                  <div className="metric-row"><span className="metric-label">p(pneumonia)</span><span className="metric-value confidence">{honest.confidence}</span></div>
                </div>
              </div>

              <div className="model-card">
                <div className="model-card-header"><h3 className="model-title">Compromised model</h3></div>
                <div className="model-image-frame">
                  <img src={compImg} alt="compromised model" className="model-image" />
                </div>
                <div className="model-metrics-table">
                  <div className="metric-row"><span className="metric-label">Prediction</span><span className={`metric-value ${comp.isAlert ? 'alert-danger' : ''}`}>{comp.label}</span></div>
                  <div className="metric-row"><span className="metric-label">p(pneumonia)</span><span className="metric-value confidence">{comp.confidence}</span></div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 2 — how it works (short) */}
        <section className="paper-section">
          <h2 className="section-title">How it works</h2>
          <p className="section-description">
            Each hospital trains locally and sends model updates to a server, which averages them into a
            shared model. A compromised hospital adds a fixed patch to a small fraction of its pneumonia
            images and teaches the model to call those "normal." Because only ~10% of its data is altered,
            its update looks almost like an honest hospital's, so the backdoor slips into the shared model
            while accuracy on ordinary images stays the same.
          </p>
          <div className="process-flow-card">
            <div className="process-steps-grid">
              <div className="process-step"><h3 className="step-title">1 · Hospitals train locally</h3><p className="step-text">One hospital secretly poisons a fraction of its own data.</p></div>
              <div className="process-step"><h3 className="step-title">2 · Server averages updates</h3><p className="step-text">Updates are combined without seeing any raw images.</p></div>
              <div className="process-step"><h3 className="step-title">3 · Shared model</h3><p className="step-text">The averaged model now carries the hidden backdoor.</p></div>
            </div>
          </div>
        </section>

        {/* Section 3 — chart */}
        <section className="paper-section">
          <h2 className="section-title">Do defenses work?</h2>
          <div className="chart-container-card">
            <div className="chart-inner-canvas">
              <div style={{ width: '100%', height: 300 }}>
                <ResponsiveContainer>
                  <BarChart data={RESULTS} margin={{ top: 10, right: 10, left: -20, bottom: 40 }}>
                    <XAxis dataKey="defense" tick={{ fontSize: 11, fill: '#475569' }} interval={0}
                           angle={-20} textAnchor="end" axisLine={{ stroke: '#cbd5e1' }} tickLine={{ stroke: '#cbd5e1' }} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: '#475569' }} unit="%"
                           axisLine={{ stroke: '#cbd5e1' }} tickLine={{ stroke: '#cbd5e1' }} />
                    <Tooltip content={<ChartTooltip />} />
                    <Bar dataKey="asr" radius={[3, 3, 0, 0]}>
                      {RESULTS.map((e, i) => <Cell key={i} fill={e.fill} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            <p className="chart-caption">
              Attack success rate at 10% poisoning (mean of 5 seeds). In our setting the sign-agreement
              method (teal) is the only one that reduces the attack. FLAME (dark red) does worse than no
              defense, we included it out of curiosity, but it is designed for 100+ clients and its
              clustering misfires at our ~20-client scale, so this reflects a scale mismatch rather than a
              flaw at its intended scale.
            </p>
          </div>

          <div className="analysis-columns-grid">
            <div className="analysis-column">
              <h3 className="analysis-column-title">Findings</h3>
<ul className="analysis-list">
  <li className="analysis-item"><span className="bullet-indicator">•</span><span>The backdoor reaches ~40% attack success while accuracy on normal images stays ~98%, so it is invisible to accuracy monitoring.</span></li>
  <li className="analysis-item"><span className="bullet-indicator">•</span><span>Attack strength peaks at a low poisoning rate: poisoning more than ~40% of the attacker's data makes the attack weaker, because the malicious update becomes too abnormal and gets diluted by honest ones.</span></li>
  <li className="analysis-item"><span className="bullet-indicator">•</span><span>Clipping, trimmed mean and Multi-Krum do not reduce the attack, the malicious update is not a statistical outlier under heterogeneous data.</span></li>
  <li className="analysis-item"><span className="bullet-indicator">•</span><span>A coordinate-wise sign-agreement defense cut the attack to ~5% and resisted an adaptive attacker that tried to evade it.</span></li>
  <li className="analysis-item"><span className="bullet-indicator">•</span><span>FLAME performed worse than no defense at this scale, discarding legitimate minority-site updates rather than the attacker.</span></li>
</ul>
            </div>
            <div className="analysis-column">
              <h3 className="analysis-column-title">Limitations</h3>
<ul className="analysis-list">
  <li className="analysis-item"><span className="bullet-indicator">•</span><span>Heterogeneity is partly synthetic: two real datasets, each sub-split into simulated hospitals rather than 20 genuinely distinct institutions.</span></li>
  <li className="analysis-item"><span className="bullet-indicator">•</span><span>Scale is limited to ~20 clients (a small consortium). Clustering defenses like FLAME assume 100+ clients; testing that regime was beyond available compute.</span></li>
  <li className="analysis-item"><span className="bullet-indicator">•</span><span>One attack type (patch-trigger data poisoning) and one task (binary pneumonia) were tested.</span></li>
  <li className="analysis-item"><span className="bullet-indicator">•</span><span>The adaptive attacker covers one evasion strategy; stronger optimization-based attacks were not explored.</span></li>
  <li className="analysis-item"><span className="bullet-indicator">•</span><span>All results are from simulation on a single GPU, not a real distributed deployment.</span></li>
</ul>
            </div>
          </div>
        </section>

        {/* Section 4 — references */}
        <section className="paper-section">
          <h2 className="section-title">References</h2>
          <ol className="ref-list">
            {REFERENCES.map((r, i) => (
              <li key={i} className="ref-item">
                <span className="ref-authors">{r.authors} ({r.year}).</span>{' '}
                <span className="ref-title">{r.title}.</span>{' '}
                <span className="ref-venue">{r.venue}.</span>
              </li>
            ))}
          </ol>
        </section>

      </main>

      <footer className="paper-footer">
        <div className="footer-content">
          <div className="footer-copyright">By: Akash A</div>
          <div className="footer-links">
            <a href="https://github.com/AkashXII/backdoorFED" className="footer-link">GitHub</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
