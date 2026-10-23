// interpret.js — turns numbers into plain words, and keeps ΔG and rank_score from being mixed up.
//
// Data rules this file follows:
//   • delta_g is the binding strength (kcal/mol, more negative = stronger).
//   • rank_score is NOT a ΔG. It only compares poses for the SAME protein (lower = stronger).
//   • "Strong / moderate / weak" is a plain-words reading of ΔG, not backend output.

import { ROUGH_GUIDE } from './config.mjs';

const MINUS = '−';

/** −7.42 with a real minus sign (nicer than a hyphen). */
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

/** One plain sentence about what a ΔG means. */
export function meaningOf(dg) {
  switch (roughGuide(dg).id) {
    case 'strong': return 'A very tight grip. More negative means tighter.';
    case 'moderate': return 'A good grip. More negative means tighter.';
    default: return 'A light grip. More negative means tighter.';
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

/**
 * Three-way verdict for a selectivity result from its 95% interval.
 * Same rule as the backend's own interpretation: interval entirely below zero → target.
 */
export function selectivityVerdict(ciLow, ciHigh) {
  if (ciHigh < 0) return {
    id: 'target', title: 'Selective for the target',
    text: 'It grabs the disease target harder than the look-alike protein, so it should leave healthy tissue alone.',
  };
  if (ciLow > 0) return {
    id: 'offtarget', title: 'Prefers the off-target',
    text: 'It grabs the look-alike protein harder than the disease target.',
  };
  return {
    id: 'none', title: 'No clear preference',
    text: 'It holds both proteins about equally well.',
  };
}
