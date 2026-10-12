// peptides.js — plain-language names for the peptides people are likely to try.
// A row of letters means nothing to someone who has never seen a peptide, so wherever we can we show a name and
// one short line next to it. Only peptides with a real, checkable identity get a name; anything else is just letters.

/**
 * @typedef {{seq: string, name: string, about: string}} NamedPeptide
 * `about` is one short sentence for a person who is not a biologist.
 */
/** @type {NamedPeptide[]} */
export const NAMED_PEPTIDES = [
  { seq: 'ETFSDLWKLLPE', name: 'p53 peptide', about: 'The piece of the p53 tumour suppressor that MDM2 grabs.' },
  { seq: 'SQETFSDLWKLLP', name: 'p53 peptide (longer cut)', about: 'The same p53 stretch with a few extra amino acids at the start.' },
  { seq: 'LSECLKRIGDELDS', name: 'BAX peptide', about: 'The part of BAX that BCL-2 holds on to so cells survive.' },
  { seq: 'LVVVGACGV', name: 'KRAS P-loop fragment', about: 'A short stretch of KRAS around its cancer-causing G12C change.' },
  { seq: 'EEEEYFELV', name: 'EGFR substrate peptide', about: 'A sequence the EGFR enzyme is known to act on.' },
  { seq: 'KTKEGVL', name: 'α-synuclein repeat', about: 'A repeating motif from the protein that clumps in Parkinson’s disease.' },
  { seq: 'LISDAELEAIFEADC', name: 'Malaria LDH peptide', about: 'Aimed at the malaria parasite’s LDH enzyme, and meant to ignore the human one.' },
  { seq: 'LIYKWVNK', name: 'Tau demo peptide', about: 'A made-up demo sequence aimed at Tau’s VQIVYK stretch. Not a known binder.' },
  { seq: 'LISAAALAAIFAAALAC', name: 'Built-in test peptide', about: 'The command-line default, handy for trying things out.' },
];

const BY_SEQ = new Map(NAMED_PEPTIDES.map((p) => [p.seq, p]));

/** The catalogue entry for a sequence, or null when it is just letters. */
export const namedPeptide = (seq) => BY_SEQ.get(String(seq || '').toUpperCase()) || null;

/** "p53 peptide" for a known sequence, otherwise the letters themselves. Used for run names. */
export const peptideTitle = (seq) => namedPeptide(seq)?.name || seq;

/** "p53 peptide (ETFSDLWKLLPE)" for a known sequence, otherwise the letters. For places that show both. */
export const peptideWithLetters = (seq) => (namedPeptide(seq) ? `${namedPeptide(seq).name} (${seq})` : seq);
