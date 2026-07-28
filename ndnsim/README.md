# ndnSIM cross-validation

The Python simulator in `../gsndn` produces every reported result. It models NDN
forwarding directly, which is what makes it possible to put an encoder in the
forwarding path and charge it its measured cost — awkward inside ns-3, where the
semantic layer would have to be an external process or a reimplementation of the
same model.

That freedom is also the risk: a hand-written simulator can be perfectly
self-consistent and still get NDN wrong, and every strategy comparison built on
it would inherit the error without ever looking suspicious. This directory
checks the part that is *not* our contribution — the transport underneath —
against a real NDN stack.

## Result

Six edge routers, 150 Interests/s, 30 s, exact-match forwarding, Content Store
disabled in both:

| | n | mean | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| ndnSIM | 4,351 | 53.72 ms | 70.08 ms | 70.18 ms | 70.21 ms |
| gsndn | 4,545 | 63.49 ms | **71.19 ms** | 71.20 ms | 71.20 ms |

The median differs by 1.11 ms. Exactly 1.00 ms of that is expected: the Python
model charges a producer 1 ms to answer a request and ndnSIM's
`ns3::ndn::Producer` answers instantly. **The unexplained residual is 0.11 ms**,
which is serialisation delay on packets of slightly different size. Link delays,
path length and queueing agree.

The *means* differ by more, and that is not a disagreement either. ndnSIM's
`ConsumerZipfMandelbrot` has all six consumers drawing from one popular-content
distribution, so a great many Interests arrive while an identical one is already
outstanding and are satisfied by PIT aggregation almost immediately. The Python
workload gives each consumer an overlapping but distinct slice of the catalog,
which aggregates less. The median — the uncontended round trip — is the quantity
the two setups define identically, and it is the one that agrees.

## Reproducing it

ndnSIM's ns-3 fork predates GCC 13 and needs two one-line fixes to build, both
the same missing-include problem: headers that used to arrive transitively no
longer do.

```bash
git clone --depth 1 https://github.com/named-data-ndnSIM/ns-3-dev.git ns-3
git clone --depth 1 --recursive https://github.com/named-data-ndnSIM/ndnSIM.git ns-3/src/ndnSIM
cd ns-3

# 1. src/network/utils/bit-deserializer.h needs <cstdint>.
#    Without it uint8_t is undeclared, std::vector<uint8_t> misparses, and the
#    member is silently deduced as int -- the error surfaces far from its cause.
sed -i 's|#include <vector>\n#include <deque>|#include <cstdint>\n&|' src/network/utils/bit-deserializer.h

# 2. src/ndnSIM/ndn-cxx/ndn-cxx/util/scheduler.hpp needs <optional>.
sed -i 's|#include "ndn-cxx/util/time.hpp"|&\n\n#include <optional>|' \
    src/ndnSIM/ndn-cxx/ndn-cxx/util/scheduler.hpp

# Only the modules the scenario needs. The wifi module has the same class of
# include problem and nothing here uses it.
python3 ./waf configure -d optimized --disable-python \
    --enable-modules=ndnSIM,point-to-point,internet,topology-read
python3 ./waf build -j$(nproc)

cp ../../Final_Project/ndnsim/gsndn-validation.cc scratch/
python3 ./waf --run "gsndn-validation --edges=6 --rate=150 --duration=30"
```

The scenario writes `results-ndnsim-delays.txt` and `results-ndnsim-rates.txt`
into the ns-3 directory and prints nothing. Then:

```bash
python ndnsim/compare.py --delays <ns-3>/results-ndnsim-delays.txt
```

which runs the Python model on the same configuration and exits non-zero if the
residual exceeds the tolerance.

## What this does not validate

Only exact-match forwarding. Semantic resolution, the Embedding Store,
verification and gossip have no ndnSIM counterpart — they are the contribution,
not the baseline, and there is nothing independent to check them against. What
is checked is that when both simulators are asked to move an Interest across the
same links to a producer and back, they agree about how long that takes.
