/**
 * Cross-validation of the Python discrete-event model against ndnSIM.
 *
 * The Python simulator in ../gsndn is where every reported result comes from.
 * It models NDN forwarding directly, which is what allows an encoder to be
 * plugged into the forwarding path and charged for its measured cost -- awkward
 * to arrange inside ns-3, where the semantic layer would have to be an external
 * process or a reimplementation of the model.
 *
 * That freedom is also the risk. A hand-written simulator can be internally
 * consistent and still get NDN itself wrong, and every comparison between
 * strategies would inherit the error silently. This scenario exists to check the
 * part that is not our contribution: given the same topology, link delays and
 * request pattern, does a real NDN stack produce the same round-trip times, the
 * same Interest satisfaction, and the same response to load?
 *
 * Only exact-match forwarding is exercised, deliberately. Semantic resolution is
 * the thing under study and has no ndnSIM counterpart to be validated against;
 * the transport underneath it does.
 *
 * Topology, matching gsndn.topology.multi_edge:
 *
 *     consumer-0 --10ms-- edge-0 --.
 *     consumer-1 --10ms-- edge-1 --+-- 5ms -- core-0 --20ms-- producer
 *     ...                          '
 *
 * Build:
 *   cp gsndn-validation.cc <ns-3>/scratch/
 *   cd <ns-3> && ./waf --run "gsndn-validation --edges=6 --rate=150"
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/ndnSIM-module.h"

#include <sstream>
#include <string>
#include <vector>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("GsndnValidation");

namespace {

/// Link delays, in milliseconds. Identical to gsndn.costs.LinkProfile, which in
/// turn takes SAF's reported edge-network measurements: 10 ms from client to
/// access router, 20 ms from router to provider.
constexpr double kConsumerToEdgeMs = 10.0;
constexpr double kRouterToRouterMs = 5.0;
constexpr double kRouterToProducerMs = 20.0;
constexpr uint32_t kBandwidthMbps = 100;

/// The catalog is 50 services; the Python model partitions them across four
/// producers. One producer prefix is enough to validate transport behaviour.
const std::string kPrefix = "/smart-hospital";

std::string
LinkDelay(double milliseconds)
{
  std::ostringstream out;
  out << milliseconds << "ms";
  return out.str();
}

std::string
LinkRate()
{
  std::ostringstream out;
  out << kBandwidthMbps << "Mbps";
  return out.str();
}

} // namespace

int
Run(int argc, char* argv[])
{
  uint32_t edges = 6;
  double rate = 150.0;      // aggregate Interests per second
  double duration = 30.0;   // seconds
  uint32_t csSize = 0;      // Content Store disabled, as in the Python model
                            // and as SAF does, to isolate forwarding from
                            // in-network caching dynamics.

  CommandLine cmd;
  cmd.AddValue("edges", "number of edge routers", edges);
  cmd.AddValue("rate", "aggregate Interests per second", rate);
  cmd.AddValue("duration", "simulated seconds", duration);
  cmd.AddValue("cs", "Content Store size in packets", csSize);
  cmd.Parse(argc, argv);

  NS_ABORT_MSG_IF(edges == 0, "need at least one edge router");

  // --- nodes ---------------------------------------------------------------

  NodeContainer consumers;
  NodeContainer edgeRouters;
  consumers.Create(edges);
  edgeRouters.Create(edges);

  Ptr<Node> core = CreateObject<Node>();
  Ptr<Node> producer = CreateObject<Node>();

  PointToPointHelper link;
  link.SetDeviceAttribute("DataRate", StringValue(LinkRate()));

  for (uint32_t i = 0; i < edges; ++i) {
    link.SetChannelAttribute("Delay", StringValue(LinkDelay(kConsumerToEdgeMs)));
    link.Install(consumers.Get(i), edgeRouters.Get(i));

    link.SetChannelAttribute("Delay", StringValue(LinkDelay(kRouterToRouterMs)));
    link.Install(edgeRouters.Get(i), core);
  }

  link.SetChannelAttribute("Delay", StringValue(LinkDelay(kRouterToProducerMs)));
  link.Install(core, producer);

  // --- NDN stack -----------------------------------------------------------

  ndn::StackHelper ndnHelper;
  if (csSize == 0) {
    ndnHelper.setPolicy("nfd::cs::lru");
    ndnHelper.setCsSize(0);
  }
  else {
    ndnHelper.setPolicy("nfd::cs::lru");
    ndnHelper.setCsSize(csSize);
  }
  ndnHelper.InstallAll();

  // Best-route, so an Interest follows the shortest path the routing protocol
  // installed. The Python model's FIBs are built by breadth-first search from
  // each producer, which is the same choice.
  ndn::StrategyChoiceHelper::InstallAll(kPrefix, "/localhost/nfd/strategy/best-route");

  ndn::GlobalRoutingHelper routing;
  routing.InstallAll();
  routing.AddOrigins(kPrefix, producer);
  routing.CalculateRoutes();

  // --- applications --------------------------------------------------------

  // Each consumer offers an equal share of the aggregate rate, matching the
  // Python workload's uniform assignment of requests to consumers.
  ndn::AppHelper consumerHelper("ns3::ndn::ConsumerZipfMandelbrot");
  consumerHelper.SetPrefix(kPrefix);
  consumerHelper.SetAttribute("Frequency", DoubleValue(rate / edges));
  consumerHelper.SetAttribute("NumberOfContents", UintegerValue(50));
  // Zipf alpha 1.2, the skew SAF uses and the Python workload's default.
  consumerHelper.SetAttribute("q", DoubleValue(0.0));
  consumerHelper.SetAttribute("s", DoubleValue(1.2));
  consumerHelper.Install(consumers);

  ndn::AppHelper producerHelper("ns3::ndn::Producer");
  producerHelper.SetPrefix(kPrefix);
  producerHelper.SetAttribute("PayloadSize", StringValue("256"));
  producerHelper.Install(producer);

  // --- tracing -------------------------------------------------------------

  // App-level delays give the round-trip time distribution to compare against
  // the Python model's Interest resolution time; rate traces give per-node
  // packet counts to compare against its forwarding totals.
  ndn::AppDelayTracer::InstallAll("results-ndnsim-delays.txt");
  ndn::L3RateTracer::InstallAll("results-ndnsim-rates.txt", Seconds(1.0));

  Simulator::Stop(Seconds(duration));
  Simulator::Run();
  Simulator::Destroy();

  return 0;
}

} // namespace ns3

int
main(int argc, char* argv[])
{
  return ns3::Run(argc, argv);
}
