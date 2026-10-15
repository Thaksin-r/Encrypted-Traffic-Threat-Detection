from ettd.flows.builder import build_flows
from ettd.ingestion.pcap_reader import PacketRecord, read_pcap


def test_build_flows_groups_bidirectional(sample_pcap):
    packets = read_pcap(sample_pcap)
    assert len(packets) >= 1
    flows = build_flows(packets, idle_timeout_seconds=120)
    assert len(flows) >= 1
    assert any(f.fwd_packets + f.bwd_packets >= 2 for f in flows)


def test_idle_timeout_splits_flows():
    packets = [
        PacketRecord(1.0, "1.1.1.1", "2.2.2.2", 1111, 80, "TCP", 100),
        PacketRecord(2.0, "2.2.2.2", "1.1.1.1", 80, 1111, "TCP", 100),
        PacketRecord(200.0, "1.1.1.1", "2.2.2.2", 1111, 80, "TCP", 100),
    ]
    flows = build_flows(packets, idle_timeout_seconds=60)
    assert len(flows) == 2
