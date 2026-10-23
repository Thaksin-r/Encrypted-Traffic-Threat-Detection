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


def test_flow_direction_both_orders():
    # Order A: 10.0.0.9 (alphabetically larger) sends first packet to 10.0.0.8:443
    packets_a = [
        PacketRecord(1.0, "10.0.0.9", "10.0.0.8", 50000, 443, "TCP", 60),
        PacketRecord(1.1, "10.0.0.8", "10.0.0.9", 443, 50000, "TCP", 120),
    ]
    flows_a = build_flows(packets_a)
    assert len(flows_a) == 1
    fa = flows_a[0]
    assert fa.src_ip == "10.0.0.9"
    assert fa.dst_ip == "10.0.0.8"
    assert fa.src_port == 50000
    assert fa.dst_port == 443
    assert fa.fwd_packets == 1
    assert fa.bwd_packets == 1
    assert fa.fwd_bytes == 60
    assert fa.bwd_bytes == 120

    # Order B: 10.0.0.8 (alphabetically smaller) sends first packet to 10.0.0.9:443
    packets_b = [
        PacketRecord(1.0, "10.0.0.8", "10.0.0.9", 50000, 443, "TCP", 70),
        PacketRecord(1.1, "10.0.0.9", "10.0.0.8", 443, 50000, "TCP", 140),
    ]
    flows_b = build_flows(packets_b)
    assert len(flows_b) == 1
    fb = flows_b[0]
    assert fb.src_ip == "10.0.0.8"
    assert fb.dst_ip == "10.0.0.9"
    assert fb.src_port == 50000
    assert fb.dst_port == 443
    assert fb.fwd_packets == 1
    assert fb.bwd_packets == 1
    assert fb.fwd_bytes == 70
    assert fb.bwd_bytes == 140


def test_flow_direction_syn_priority():
    # Stray server packet arrives first, but client sends SYN without ACK
    packets = [
        PacketRecord(1.0, "10.0.0.8", "10.0.0.9", 443, 50000, "TCP", 50, tcp_flags="A"),
        PacketRecord(1.1, "10.0.0.9", "10.0.0.8", 50000, 443, "TCP", 60, tcp_flags="S"),
    ]
    flows = build_flows(packets)
    assert len(flows) == 1
    f = flows[0]
    assert f.src_ip == "10.0.0.9"
    assert f.dst_ip == "10.0.0.8"
    assert f.src_port == 50000
    assert f.dst_port == 443
    assert f.fwd_packets == 1
    assert f.bwd_packets == 1

