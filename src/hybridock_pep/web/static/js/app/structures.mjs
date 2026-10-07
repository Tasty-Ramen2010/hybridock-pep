// structures.js — loads protein structures for DISPLAY (bundled files, uploads, or RCSB).
// This is not the docking backend; it only fetches and parses PDB text so we can draw it.

import { parsePDB } from './pdb.mjs';
import { hashString } from './stage/math.mjs';
import { assetUrl } from './config.mjs';

/** fetch() that retries on a dropped connection (the local server answers HTTP/1.0 with a small listen queue). */
export async function fetchRetry(url, tries = 3) {
  let last;
  for (let i = 0; i < tries; i++) {
    try { return await fetch(url); } catch (err) { last = err; await new Promise((r) => setTimeout(r, 250 * (i + 1))); }
  }
  throw last;
}

const cache = new Map(); // id -> { text, structure }
const customTexts = new Map(); // id -> text (uploads live only for this browser session)

/** A reference to a protein: { id, key?, name, pdb?, file?, custom?, site?, box? } */
const idOf = (p) => p.id || (p.key ? `preset:${p.key}` : p.pdb ? `pdb:${p.pdb}` : `file:${p.file}`);

/** Register a user-uploaded PDB (kept in memory only) and return its reference. */
export function registerUpload(name, text) {
  const id = `upload:${hashString(name + text.length + text.slice(0, 200)).toString(36)}`;
  customTexts.set(id, text);
  return { id, name: name.replace(/\.pdb$/i, ''), custom: true, file: name };
}

/** Download a structure from RCSB by 4-character PDB ID. Throws a friendly Error on failure. */
export async function fetchFromRCSB(pdbId) {
  const id = pdbId.toUpperCase();
  let res;
  try {
    res = await fetch(`https://files.rcsb.org/download/${id}.pdb`);
  } catch {
    throw new Error("Couldn't reach the Protein Data Bank. Check your internet connection, or upload a PDB file instead.");
  }
  if (!res.ok) throw new Error(`The Protein Data Bank has no PDB file for “${id}”. Check the code and try again.`);
  const text = await res.text();
  const ref = registerUpload(`${id}.pdb`, text);
  return { ...ref, name: id, pdb: id, custom: false };
}

/** @returns {Promise<{text:string, structure:ReturnType<typeof parsePDB>}>} */
export async function loadProtein(p) {
  const id = idOf(p);
  if (cache.has(id)) return cache.get(id);
  let text = p.text ?? customTexts.get(id);
  if (text == null) {
    if (!p.file) throw new Error('The protein file for this item is no longer available. Choose the protein again.');
    const res = await fetchRetry(assetUrl(p.file));
    if (!res.ok) throw new Error(`Couldn't load ${p.file}.`);
    text = await res.text();
  }
  const entry = { text, structure: parsePDB(text) };
  cache.set(id, entry);
  return entry;
}

export const hasProtein = (p) => cache.has(idOf(p)) || customTexts.has(idOf(p)) || !!p.file && !p.custom;
