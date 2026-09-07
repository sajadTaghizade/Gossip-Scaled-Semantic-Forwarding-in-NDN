# Related work

Assembled while checking whether this project's two contributions are actually
new. **The PDFs are not in the repository** — the machine this was assembled on
could not reach arxiv.org, ieeexplore.ieee.org, dl.acm.org, doi.org or
named-data.net (egress policy, 403 on CONNECT). Run `./fetch.sh` from a machine
with normal network access and the open-access ones land in `pdf/`.

**Verification status is marked per entry and matters.** Most of this was
assembled from search results, not from reading the papers. Anything marked
`[unverified]` had its title and venue seen only in a search snippet — check it
before citing it. Two entries were confirmed directly against primary sources
and are marked `[verified]`.

## The ones that decide whether contribution 1 is new

Contribution 1 is: share a *verified* name→prefix resolution across routers by
anti-entropy gossip instead of caching it per router. NDN already ships
anti-entropy dataset synchronisation, and already floods name-prefix
reachability over it. Neither of these is cited in the current write-up and both
are the first thing an NDN reviewer will reach for.

- `[verified]` **Zhu, Afanasyev.** *Let's ChronoSync: Decentralized Dataset State
  Synchronization in Named Data Networking.* ICNP 2013.
  <https://named-data.net/publications/chronosync/> — condensed cryptographic
  digests exchanged among parties to reconcile dataset differences. Deprecated
  in favour of PSync/SVS, per <https://github.com/named-data/ChronoSync>.
- `[verified]` **Li, Shang, Afanasyev, Wang, Zhang.** *A Brief Introduction to
  NDN Dataset Synchronization (NDN Sync).* MILCOM 2018.
  <https://named-data.net/publications/li2018sync-intro/> — survey of ChronoSync
  (digest trees), PSync (invertible Bloom filters), VectorSync (state vectors).
- **Gawande.** *Improvements to PSync: Distributed Full Dataset Synchronization
  in Named Data Networking.* MSc thesis, U. Memphis.
  <https://digitalcommons.memphis.edu/etd/2054/>
- **Hoque, Amin, Alyyan, Bhattacharjee, Pesavento, Zhang.** *NLSR: Named-data
  Link State Routing Protocol.* ACM SIGCOMM ICN workshop 2013.
  <https://conferences.sigcomm.org/sigcomm/2013/papers/icn/p15.pdf> — already
  disseminates name-prefix reachability network-wide, over sync.
- **RFC 9138.** *Design Considerations for Name Resolution Service in ICN.*
  IRTF ICNRG, 2021. <https://datatracker.ietf.org/doc/rfc9138/> — a distributed
  database mapping a name to other forwarding information is a standardised ICN
  component.
- **Garetto, Leonardi, Neglia.** *Similarity Caching: Theory and Algorithms.*
  INFOCOM 2020; extended in IEEE/ACM ToN 2021. <https://arxiv.org/abs/1912.03888>
  — covers *networks* of similarity caches. This system is formally one.
- **Fayazbakhsh, Lin, Tootoonchian, Ghodsi, Koponen, Maggs, Ng, Sekar, Shenker.**
  *Less pain, most of the gain: incrementally deployable ICN.* SIGCOMM 2013.
  <https://dl.acm.org/doi/10.1145/2486001.2486023>
- `[unverified]` **Nakamura, Kamiyama.** *Analysis of Similarity Caching on
  General Cache Networks.* IEEE Access 12, 2024.

## The ones that decide whether contribution 2 is new

Contribution 2 is: replace the fixed similarity threshold with a per-route
boundary learned online against an operator-set error budget.

- **Zhu, Sheng, et al.** *vCache: Verified Semantic Prompt Caching.*
  arXiv:2502.03771. Per-item learned thresholds under a user-set error bound.
  **Already cited** in the write-up, correctly, as prior art for the shape of
  the idea.
- **Lu, Yu, Kalpathy-Cramer, et al.** *Federated Conformal Predictors for
  Distributed Uncertainty Quantification.* ICML 2023.
  <https://arxiv.org/abs/2305.17564>. **Already cited.**
- **Finamore, Roberts, Gallo, Rossi.** *Accelerating Deep Learning
  Classification with Error-controlled Approximate-key Caching.* INFOCOM 2022.
  <https://arxiv.org/abs/2112.06671> — similarity caching of model outputs with
  explicit error control, in networking. The closest published ancestor of both
  the premise and contribution 2, and not cited.
- **Bates, Angelopoulos, Lei, Malik, Jordan.** *Distribution-Free
  Risk-Controlling Prediction Sets.* JACM 68(6), 2021.
  <https://arxiv.org/abs/2101.02703>
- **Angelopoulos, Bates, Fisch, Lei, Schuster.** *Conformal Risk Control.*
  ICLR 2024. <https://arxiv.org/abs/2208.02814>
- **Gibbs, Candès.** *Adaptive Conformal Inference Under Distribution Shift.*
  NeurIPS 2021. <https://arxiv.org/abs/2106.00170> — the ACI update the
  `rc-ndn-aci` arm implements.
- `[unverified]` **Wen, Xing, Simeone.** *Distributed Conformal Prediction via
  Message Passing.* ICML 2025. <https://arxiv.org/abs/2501.14544> — calibration
  distributed over a neighbour graph. If this is what it appears to be, it is
  both contributions combined, with guarantees.
- `[unverified]` **Simeone et al.** *Calibrating AI Models for Wireless
  Communications via Conformal Prediction.* arXiv:2212.07775 — conformal
  prediction is not new to networking.
- `[unverified]` Online conformal prediction under partial/bandit feedback:
  arXiv:2604.17984, arXiv:2506.14067. These address the exploration deadlock
  §7 reports, with regret bounds, where this work uses ε-greedy.

## Semantic forwarding in ICN — the direct line

- **Chan, Ko, Mastorakis, Afanasyev, Zhang.** *Fuzzy Interest Forwarding.*
  AINTEC 2017, DOI 10.1145/3154970.3154975. Follow-up arXiv:1802.03072. Code:
  <https://github.com/spirosmastorakis/FIF> (modified ndnSIM). **Already cited.**
- **Amadeo et al.** *Enhancing IoT Service Discovery Through Semantic Name-Based
  Forwarding.* IEEE IoT Magazine, 2026. The baseline this work extends.
  **Note:** it is a *magazine* article — short, tutorial-framed. The Embedding
  Store detail that the whole SAF+ES baseline rests on should be re-checked
  against the actual text.
- **Raza, Ullah, Din, Rehman, Kim.** *INF-NDN IoT.* IEEE Access 12:114319, 2024.
  <https://ieeexplore.ieee.org/document/10638459>. **Already cited.** Its
  supernode delegation is a competing answer to contribution 1's problem and
  should be compared against, not just mentioned.
- **Quevedo, Antunes, Corujo, Gomes, Aguiar.** *On the application of contextual
  IoT service discovery in Information Centric Networks.* Computer
  Communications 89:117–127, 2016.
- `[unverified]` **Hlaing, Asaeda.** *NeuName: Autonomous Semantic Naming for ICN
  using Multimodal Neural Intelligence.* IEEE NetSoft 2026, pp. 126–134.
  Three months old and directly in this space — read before submitting.
- `[unverified]` **Amadeo, Ruggeri, Molinaro, Nitti, Serrano.** *Service
  Provisioning in Digital Twin Networks with Semantic-Aware ICN.* CCNC 2026.
  Same group's follow-up; extends semantic matching to cached results.

## Name collision to fix

- `[verified]` **Posch, Rainer, Hellwagner.** *SAF: Stochastic Adaptive
  Forwarding in Named Data Networking.* IEEE/ACM ToN 25(2):1089–1102, 2017.
  <https://arxiv.org/abs/1505.05259> — "SAF" already means this to every NDN
  reader. The shorthand for Amadeo's semantic forwarding needs a different name.

## Cited correctly, no action

- **Askar, Habbal, Hamouda, Alnajim, Khan.** *SEF: A Smart and Energy-Aware
  Forwarding Strategy for NDN-Based Internet of Healthcare.* CMC 81(3), 2024.
  RL over geographic progress and node energy — no semantic matching. The
  write-up cites it for the energy model and the 20-seed protocol, which is
  what it is for. `gsndn/strategies/sef.py` documents the same limitation.

## Context on why the encoder cannot sit on the fast path

- **Zhang, Yuan, Crowley et al.** NDN line-rate forwarding: 10 Gbps with >10M
  prefixes; *Vision: toward 10 Tbps NDN forwarding with billion prefixes*,
  DOI 10.1145/3460417.3482973. Per-packet budget is ns–µs; MiniLM is ms.
- **Zheng et al.** *Offloading Machine Learning to Programmable Data Planes: A
  Systematic Survey.* ACM Computing Surveys, 2023. DOI 10.1145/3605153.
- **RFC 7945.** *ICN: Evaluation and Security Considerations.* IRTF — the
  community's own statement that ICN lacks deployment traces, which is the
  standing defence for simulation-only evaluation.
