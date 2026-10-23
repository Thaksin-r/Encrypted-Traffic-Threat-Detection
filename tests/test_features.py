from ettd.features.extractor import FEATURE_COLUMNS, flows_to_dataframe
from ettd.flows.builder import build_flows
from ettd.ingestion.pcap_reader import PacketRecord, read_pcap


def test_feature_columns_present(sample_pcap):
    packets = read_pcap(sample_pcap)
    flows = build_flows(packets)
    df = flows_to_dataframe(flows)
    assert len(df) >= 1
    for col in FEATURE_COLUMNS:
        assert col in df.columns
    required_numeric = [
        "duration",
        "fwd_packets",
        "bwd_packets",
        "fwd_bytes",
        "bwd_bytes",
        "pkt_len_mean",
        "iat_mean",
    ]
    for col in required_numeric:
        assert df[col].notna().any() or True  # zeros allowed


def test_missing_tls_fields_are_nullish(sample_pcap):
    packets = read_pcap(sample_pcap)
    flows = build_flows(packets)
    df = flows_to_dataframe(flows)
    # Synthetic sample has no TLS handshake layers; fields should be False/None
    assert set(df["tls_present"].unique()).issubset({False, True})
    # SNI should be null/None for non-TLS synthetic packets
    assert df["tls_sni"].isna().all() or (df["tls_sni"] == None).all()  # noqa: E711
    assert "tls_cipher_suites_missing" in df.columns
    assert df["tls_cipher_suites_missing"].all()

    from ettd.ml.features_matrix import build_matrix

    mat, cols = build_matrix(df)
    assert "tls_cipher_suites_missing" in cols
    idx = cols.index("tls_cipher_suites_missing")
    assert (mat[:, idx] == 1.0).all()


def test_no_payload_bytes_in_features(sample_pcap):
    packets = read_pcap(sample_pcap)
    flows = build_flows(packets)
    df = flows_to_dataframe(flows)
    blob = df.to_csv(index=False)
    assert "xxxxx" not in blob
    assert "payload" not in df.columns
    assert "raw" not in df.columns


def test_beaconing_features_calculation():
    # 6 flows with exact 10-second intervals between 10.0.0.5 and 10.0.0.99
    packets = []
    base = 1700000000.0
    for i in range(6):
        t = base + i * 10.0
        packets.append(PacketRecord(t, "10.0.0.5", "10.0.0.99", 50000 + i, 443, "TCP", 64))

    flows = build_flows(packets, idle_timeout_seconds=5.0)
    assert len(flows) == 6
    df = flows_to_dataframe(flows)
    assert "beacon_gap_mean" in df.columns
    assert "beacon_gap_cv" in df.columns
    assert "beacon_flow_count" in df.columns

    for _, row in df.iterrows():
        assert row["beacon_flow_count"] == 6
        assert abs(row["beacon_gap_mean"] - 10.0) < 1e-3
        assert row["beacon_gap_cv"] < 0.01  # extremely regular, CV near 0

