from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scapy.all import IP, TCP, load_layer, wrpcap  # type: ignore[attr-defined]

load_layer("tls")
from scapy.layers.tls.extensions import ServerName, TLS_Ext_ServerName  # type: ignore
from scapy.layers.tls.handshake import TLSClientHello  # type: ignore
from scapy.layers.tls.record import TLS  # type: ignore

from ettd.ingestion.pcap_reader import read_pcap


def test_tls_parsing_ports_and_fresh_process(tmp_path: Path):
    pcap_path = tmp_path / "tls_test.pcap"

    pkts = [
        IP(src="10.0.0.1", dst="10.0.0.2")
        / TCP(sport=50000, dport=443)
        / TLS(
            msg=[
                TLSClientHello(
                    version=0x0303,
                    ciphers=[0x1301, 0xC02F],
                    ext=[TLS_Ext_ServerName(servernames=[ServerName(servername=b"example443.com")])],
                )
            ]
        ),
        IP(src="10.0.0.1", dst="10.0.0.2")
        / TCP(sport=50001, dport=8443)
        / TLS(
            msg=[
                TLSClientHello(
                    version=0x0303,
                    ciphers=[0x1302, 0xC030],
                    ext=[TLS_Ext_ServerName(servernames=[ServerName(servername=b"example8443.com")])],
                )
            ]
        ),
    ]
    wrpcap(str(pcap_path), pkts)

    # 1. In-process test
    records = read_pcap(pcap_path)
    assert len(records) == 2

    r443 = records[0]
    assert r443.tls_meta.get("tls_present") is True
    assert r443.tls_meta.get("tls_sni") == "example443.com"
    assert r443.tls_meta.get("tls_version") == 0x0303
    assert 0x1301 in (r443.tls_meta.get("tls_cipher_suites") or [])

    r8443 = records[1]
    assert r8443.tls_meta.get("tls_present") is True
    assert r8443.tls_meta.get("tls_sni") == "example8443.com"
    assert r8443.tls_meta.get("tls_version") == 0x0303
    assert 0x1302 in (r8443.tls_meta.get("tls_cipher_suites") or [])

    # 2. Fresh process test (proves module import loads TLS layer before rdpcap)
    code = f"""
import sys
from pathlib import Path
ROOT = Path(r"{Path(__file__).resolve().parents[1]}")
sys.path.insert(0, str(ROOT / "src"))

from ettd.ingestion.pcap_reader import read_pcap

records = read_pcap(r"{pcap_path}")
assert len(records) == 2

r443 = records[0]
assert r443.tls_meta.get("tls_present") is True, "443 tls_present failed"
assert r443.tls_meta.get("tls_sni") == "example443.com", "443 tls_sni failed"
assert r443.tls_meta.get("tls_version") == 0x0303, "443 tls_version failed"
assert 0x1301 in (r443.tls_meta.get("tls_cipher_suites") or []), "443 cipher failed"

r8443 = records[1]
assert r8443.tls_meta.get("tls_present") is True, "8443 tls_present failed"
assert r8443.tls_meta.get("tls_sni") == "example8443.com", "8443 tls_sni failed"
assert r8443.tls_meta.get("tls_version") == 0x0303, "8443 tls_version failed"
assert 0x1302 in (r8443.tls_meta.get("tls_cipher_suites") or []), "8443 cipher failed"
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"Fresh process failed: {result.stderr}\n{result.stdout}"
