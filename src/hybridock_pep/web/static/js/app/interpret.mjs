// interpret.js — turns numbers into plain words, and keeps ΔG and rank_score from being mixed up.
//
// Data rules this file follows:
//   • delta_g is the binding strength (kcal/mol, more negative = stronger).
//   • rank_score is NOT a ΔG. It only compares poses for the SAME protein (lower = stronger).
//   • "Strong / moderate / weak" is OUR rough wording, not backend output. The UI labels it so.
//   • A single run has no high/low confidence badge.

import { ROUGH_GUIDE, TYPICAL_ERROR } from './config.mjs';

const MINUS = '−';

/** −7.42 with a real minus sign (nicer in serif type than a hyphen). */
export function fmt(v, digits = 2) {
  if (v == null || Number.isNaN(v)) return '—';
  const s = Math.abs(v).toFixed(digits);
  return v < 0 && Number(s) !== 0 ? MINUS + s : s;
}

export const fmtSigned = (v, digits = 2) => (v > 0 ? '+' : '') + fmt(v, digits);

/** Our rough wording for a ΔG. */
export function roughGuide(dg) {
  return ROUGH_GUIDE.find((g) => dg <= g.max);
}

/** One or two friendly sentences about what a ΔG means. */
export function meaningOf(dg) {
  const lead = 'A more negative number means a tighter grip. ';
  const err = ` With a typical error of ±${TYPICAL_ERROR} kcal/mol, treat it as a good estimate rather than an exact value.`;
  switch (roughGuide(dg).id) {
    case 'strong': return lead + 'This peptide is predicted to stick to the protein very tightly, a promising candidate to test.' + err;
    case 'moderate': return lead + 'This peptide is predicted to stick fairly well, which is a reasonable starting point to improve on.' + err;
    default: return lead + 'This peptide is predicted to stick only weakly to this spot.' + err;
  }
}

const RT = 0.001987204 * 298.15; // kcal/mol at 25 °C

/** Rough dissociation constant (molar) implied by a ΔG at 25 °C: Kd = exp(ΔG / RT). */
export const kdFromDG = (dg) => Math.exp(dg / RT);

export function fmtKd(kd) {
  if (kd >= 1e-3) return `${(kd * 1e3).toFixed(1)} mM`;
  if (kd >= 1e-6) return `${(kd * 1e6).toFixed(kd >= 1e-5 ? 0 : 1)} µM`;
  if (kd >= 1e-9) return `${(kd * 1e9).toFixed(kd >= 1e-8 ? 0 : 1)} nM`;
  return `${(kd * 1e12).toFixed(1)} pM`;
}

/** How many times weaker/stronger the binding could be at ± one typical error. */
export const errorFoldChange = () => Math.exp(TYPICAL_ERROR / RT);

/**
 * Three-way verdict for a selectivity result from its 95% interval.
 * Same rule as the backend's own interpretation: interval entirely below zero → target.
 */
export function selectivityVerdict(ciLow, ciHigh) {
  if (ciHigh < 0) return {
    id: 'target', title: 'Selective for the target',
    text: 'The peptide binds the target more tightly than the off-target, and the whole 95% interval is below zero.',
  };
  if (ciLow > 0) return {
    id: 'offtarget', title: 'Prefers the off-target',
    text: 'The peptide binds the off-target more tightly than the target, and the whole 95% interval is above zero.',
  };
  return {
    id: 'none', title: 'No clear preference',
    text: 'The 95% interval crosses zero, so we cannot say which protein the peptide prefers.',
  };
}
