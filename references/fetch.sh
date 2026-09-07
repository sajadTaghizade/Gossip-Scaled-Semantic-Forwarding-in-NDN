#!/usr/bin/env bash
# Fetch the open-access PDFs of the related work into ./pdf/.
#
# Run this from a machine with normal network access. The environment this
# repository's related-work list was assembled on could not reach arxiv.org,
# ieeexplore.ieee.org, dl.acm.org, doi.org or named-data.net.
#
# Only open-access sources are listed. Entries behind IEEE Xplore or the ACM DL
# are named in README.md with their DOI and need an institutional login.

set -u
cd "$(dirname "$0")"
mkdir -p pdf

get() {  # get <filename> <url>
    if [ -s "pdf/$1" ]; then
        echo "  have    $1"
        return
    fi
    if curl -fsSL --max-time 120 -o "pdf/$1" "$2"; then
        echo "  fetched $1"
    else
        echo "  FAILED  $1  <- $2"
        rm -f "pdf/$1"
    fi
}

echo "arXiv:"
get saf-stochastic-adaptive-forwarding-ton2017.pdf https://arxiv.org/pdf/1505.05259
get fuzzy-interest-forwarding-experiments.pdf      https://arxiv.org/pdf/1802.03072
get similarity-caching-theory-algorithms.pdf       https://arxiv.org/pdf/1912.03888
get approximate-key-caching-infocom2022.pdf        https://arxiv.org/pdf/2112.06671
get vcache-verified-semantic-prompt-caching.pdf    https://arxiv.org/pdf/2502.03771
get federated-conformal-predictors-icml2023.pdf    https://arxiv.org/pdf/2305.17564
get risk-controlling-prediction-sets-jacm.pdf      https://arxiv.org/pdf/2101.02703
get conformal-risk-control.pdf                     https://arxiv.org/pdf/2208.02814
get adaptive-conformal-inference-gibbs-candes.pdf  https://arxiv.org/pdf/2106.00170
get distributed-conformal-message-passing.pdf      https://arxiv.org/pdf/2501.14544
get conformal-wireless-simeone.pdf                 https://arxiv.org/pdf/2212.07775
get semantic-caching-llm-serving.pdf               https://arxiv.org/pdf/2508.07675

echo "NDN project + SIGCOMM (open):"
get nlsr-named-data-link-state-routing.pdf https://conferences.sigcomm.org/sigcomm/2013/papers/icn/p15.pdf
get ndn-sync-brief-introduction.pdf        https://named-data.net/wp-content/uploads/2018/10/ndn-sync-intro.pdf
get state-vector-sync-tr-ndn-0073.pdf      https://named-data.net/wp-content/uploads/2021/07/ndn-0073-r2-SVS.pdf

echo "IRTF:"
get rfc9138-icn-name-resolution-service.pdf https://www.rfc-editor.org/rfc/pdfrfc/rfc9138.txt.pdf
get rfc7945-icn-evaluation-security.pdf     https://www.rfc-editor.org/rfc/pdfrfc/rfc7945.txt.pdf

echo
echo "Behind a paywall -- see README.md for DOIs:"
echo "  Chan et al., Fuzzy Interest Forwarding, AINTEC 2017     10.1145/3154970.3154975"
echo "  Amadeo et al., Semantic Name-Based Forwarding, IoT Mag 2026"
echo "  Raza et al., INF-NDN IoT, IEEE Access 2024              10.1109/ACCESS.2024.3448775"
echo "  Fayazbakhsh et al., Less pain most of the gain, SIGCOMM 2013  10.1145/2486001.2486023"
echo "  Zheng et al., ML on programmable data planes, CSUR 2023 10.1145/3605153"
