# IST'2026 submission

12th International Symposium on Telecommunication, ICT Research Institute
(ITRC), Tehran. Everything below is quoted or paraphrased from the symposium's
own *Paper Submission* page; where it is quoted, it is quoted exactly.

**The file to upload is [`../paper/paper-draft.pdf`](../paper/paper-draft.pdf).**
Nothing else — no LaTeX source, no code, no figures.

## What the symposium requires, and where we stand

| Requirement | Status |
|---|---|
| At most **6 double-column pages** (4 more at 1,000,000 Rials / 5 USD each) | 6 pages exactly |
| **IEEE template.** "Papers with any other formats will not be sent for the review process" | `IEEEtran`, `conference` |
| A4 or US Letter (both templates offered; the A4 one is in this folder) | A4 |
| English only | yes |
| Unpublished work, new results | yes |
| **Double-blind review** | anonymised — see below |
| No plagiarism; detection at any stage excludes the paper | — |

## Double-blind

> "The conference uses double-blind review, which means that both the reviewer
> and author identities are concealed from the reviewers, and vice versa,
> throughout the review process."

So the PDF carries no author block. The named block is kept, commented, at the
top of `paper/paper.tex`, to be restored on acceptance.

This was verified rather than assumed: a text extraction of all six pages
returns zero hits for any author surname, for the university, and for the email
domain, and the PDF's Title, Author, Subject and Keywords metadata fields are
empty. Nothing else in the body identifies the authors — no repository link, no
acknowledgement, no funder, no "our earlier work".

Author details still go in the submission *form*. The system knows who you are;
the reviewer does not.

## Steps

1. **Sign up** at `ist.itrc.ac.ir/2026/en`, choosing a registration group.
   Student and IEEE-member rates each need a certificate uploaded afterwards —
   the site warns that without it the group is not applied and the full rate
   stands. The group can be changed later from the profile page.
2. **Sign in**, open **my papers**, upload `paper/paper-draft.pdf`.
3. **Topic**: Network and Security → AI-Enabled Networks. Second choice,
   Network and Security → IoT/Industrial IoT.
4. On acceptance: restore the author block, register (one full registration per
   accepted paper), and prepare a poster — 70 × 100 cm portrait, sections
   Abstract (≤120 words), Introduction, Research goal, Research Methodology,
   Results, Conclusion, References.
5. **Present it.** "Those accepted articles which are not presented will be
   excluded from the conference proceedings." Only accepted *and presented*
   papers are indexed by IEEE.

## Open before camera-ready

- `sajjad.taghizadeh@ut.ac.ir` in the commented author block is a **guess**,
  from the firstname.lastname pattern two of the other addresses follow.
  `mohammadmahdiyari@` does not follow it, so the pattern is not reliable.
- Author order. The commented block lists Taghizadeh first; the repository
  README lists Yari first. Whichever is right, it is what gets indexed.
- Reference [9], `raza2025smartnets`: the entry lists six authors. The sources
  reachable while checking listed five, without M. N. Ali. Confirm on Xplore.

## Not yet read

The *Ethical Issues*, *Important Dates* and *Registration fee* pages of the
symposium site were not available when this was written. Check them; the
deadline in particular is not recorded anywhere here.
