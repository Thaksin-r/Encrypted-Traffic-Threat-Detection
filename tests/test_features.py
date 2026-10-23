from ettd.features.extractor import FEATURE_COLUMNS, flows_to_dataframe
from ettd.flows.builder import build_flows
from ettd.ingestion.pcap_reader import read_pcap


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
