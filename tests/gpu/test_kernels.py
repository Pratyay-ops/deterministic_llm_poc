"""Kernel tests - need CUDA. In the container on the GB10:

    bash scripts/gb10/run.sh python3 -m pytest -m gpu tests/gpu -q
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

pytestmark = pytest.mark.gpu


@pytest.fixture
def extra():
    from detllm.kernels import extra_ops

    yield extra_ops
    extra_ops.disable()


def _gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def test_extra_bmm_is_correct_and_batch_invariant(extra):
    g = _gen(0)
    for dtype, tol in ((torch.bfloat16, 3e-2), (torch.float32, 1e-5)):
        a = torch.randn(64, 7, 128, generator=g, device="cuda", dtype=dtype)
        b = torch.randn(64, 128, 300, generator=g, device="cuda", dtype=dtype)
        ref = a.double() @ b.double()
        out = extra.bmm(a, b)
        assert ((out.double() - ref).abs().max() / ref.abs().max()).item() < tol
        first = extra.bmm(a[:1], b[:1])
        for n in (2, 5, 17, 64):
            assert torch.equal(extra.bmm(a[:n], b[:n])[:1], first), (dtype, n)


def test_extra_bmm_fp32_is_exact(extra):
    # RoPE pattern: integer positions through a K=1 fp32 matmul must survive exactly (no TF32).
    pos = torch.arange(8192, dtype=torch.float32, device="cuda").reshape(1, 1, -1)
    ones = torch.ones(1, 1, 1, device="cuda")
    assert torch.equal(extra.bmm(ones, pos), pos)


def test_extra_softmax_is_correct_and_batch_invariant(extra):
    x = torch.randn(300, 5000, generator=_gen(1), device="cuda") * 5
    out = extra.softmax(x, -1)
    assert torch.allclose(out, torch.softmax(x, -1), atol=1e-6, rtol=1e-4)
    for n in (1, 2, 33, 300):
        assert torch.equal(extra.softmax(x[:n], -1)[:1], out[:1]), n
    assert extra.softmax(x.to(torch.bfloat16), -1, half_to_float=True).dtype == torch.float32


def test_extra_ops_are_reached_through_the_dispatcher(extra):
    info = extra.enable()
    assert info["dispatch_check"] == {"matmul_routed": True, "softmax_routed": True}


def test_official_library_bf16_mm_is_batch_invariant():
    from detllm.kernels import official

    try:
        official.enable(probe_dtypes=())
    except ImportError as exc:
        pytest.skip(str(exc))
    try:
        g = _gen(2)
        a = torch.randn(512, 1024, generator=g, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(1024, 768, generator=g, device="cuda", dtype=torch.bfloat16)
        first = torch.mm(a[:1], w)
        for n in (2, 3, 64, 129, 512):
            assert torch.equal(torch.mm(a[:n], w)[:1], first), n
    finally:
        official.disable()


def test_custom_mode_is_batch_invariant_and_accurate():
    from detllm.kernels.custom import FixedChunkMode

    g = _gen(3)
    a = torch.randn(300, 512, generator=g, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(256, 512, generator=g, device="cuda", dtype=torch.bfloat16)
    stock = torch.nn.functional.linear(a, w)
    mode = FixedChunkMode()
    with mode:
        full = torch.nn.functional.linear(a, w)
        first = torch.nn.functional.linear(a[:1], w)
        rows = [torch.nn.functional.linear(a[:n], w)[:1] for n in (2, 63, 64, 65, 300)]
        sm_first = torch.softmax(a[:1].float(), -1)
        sm_full = torch.softmax(a.float(), -1)[:1]
    assert all(torch.equal(r, first) for r in rows)
    assert torch.equal(sm_first, sm_full)
    assert (full.float() - stock.float()).abs().max().item() < 0.5
    assert mode.stats()["chunked_calls"]
