/* ============================================================
   REAL DATA — replace the placeholder RESULTS, PREDICTIONS,
   and SCANS in App.jsx with these.
   ============================================================ */

// Mean Attack Success Rate (%) at the STEALTHY setting (10% poisoning),
// averaged over 5 seeds, from your 20-client matrix.
// invariant is the accent; FLAME is highlighted as the failure.
export const RESULTS = [
  { defense: 'No defense (FedAvg)', asr: 36.9, fill: '#e2e8f0' },
  { defense: 'Norm clipping',       asr: 37.3, fill: '#e2e8f0' },
  { defense: 'Trimmed mean',        asr: 38.6, fill: '#e2e8f0' },
  { defense: 'Multi-Krum',          asr: 39.1, fill: '#e2e8f0' },
  { defense: 'FLAME',               asr: 66.6, fill: '#c0504d' },
  { defense: 'Sign-agreement',      asr: 4.8,  fill: '#2a9d8f' }
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
