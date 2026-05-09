from __future__ import annotations

from subprocess import CompletedProcess

from benchmark import generate_net_variant


def test_generate_net_variant_success_builds_expected_command(monkeypatch):
    """generate_net_variant вызывает netconvert с -s, нужным типом TLS и выходным файлом."""
    calls: dict = {}

    def fake_run(cmd, **_kw):
        calls["cmd"] = cmd
        return CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr("benchmark.subprocess.run", fake_run)

    ok = generate_net_variant("base.net.xml", "actuated", "out.net.xml")
    assert ok is True
    assert calls["cmd"] == [
        "netconvert",
        "-s",
        "base.net.xml",
        "--tls.default-type",
        "actuated",
        "-o",
        "out.net.xml",
    ]


def test_generate_net_variant_handles_netconvert_failure(monkeypatch):
    """Любой сбой subprocess оборачивается в False без необработанного исключения наружу."""
    def boom(*args, **kwargs):  # noqa: ANN001, ANN002
        raise OSError("no netconvert")

    monkeypatch.setattr("benchmark.subprocess.run", boom)
    assert generate_net_variant("base.net.xml", "static", "out.net.xml") is False
