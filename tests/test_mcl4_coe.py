"""Unit and mathematical invariant tests for MCL4-COE."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mcl4_coe import (
    ByteTokenizer,
    COEConfig,
    CPUOffloadAdamW,
    CognitiveOperatorEnsemble,
    CognitiveSchedule,
    GLiNER25CorpusGate,
    MCL4CEDBackbone,
    MCL4CEDConfig,
    MultiPassController,
    NVFP4Linear,
    NoveltyTaskBuffer,
    ScaffoldCache,
    TokenBlockDataset,
    collate_decisions,
    compute_harness_reward,
    compute_joint_coe_loss,
    compute_task_reward,
    convert_linear_to_nvfp4,
    evaluate_byte_sieve,
    get_gemm_counts,
    group_normalized_advantages,
    grpo_clipped_loss,
    grow_experts,
    normalize_prose_unwrapping,
    parse_request,
    pearson_similarity,
    prepare_model,
    prune_experts,
    reset_gemm_counts,
    staleness_weight,
)


def _tiny_model(device: str = "cpu") -> CognitiveOperatorEnsemble:
    torch.manual_seed(7)
    cfg = MCL4CEDConfig(
        d_model=64,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ple_dim=16,
        expert_hidden_dim=128,
        num_experts=4,
        top_k=2,
        max_seq_len=128,
    )
    backbone = MCL4CEDBackbone(cfg)
    coe = CognitiveOperatorEnsemble(backbone, COEConfig(projection_dim=16, recurrence_steps=2))
    return coe.to(device).eval()


def test_bitwise_permutation_invariance_across_candidates() -> None:
    """Reordering candidate options must permute logits and probabilities down to the last bit."""
    model = _tiny_model()
    request = {
        "state": "A rabbit carries two red balls.",
        "questions": {
            "animal": {
                "type": "choice",
                "instructions": "Which animal is here?",
                "options": [
                    {"id": "rabbit", "description": "A rabbit."},
                    {"id": "cat", "description": "A cat carries a bell."},
                    {"id": "dog", "description": "A large brown dog."},
                ],
            }
        },
    }
    examples = parse_request(request)
    batch = collate_decisions(examples, max_length=128)

    perm = torch.tensor([2, 0, 1], dtype=torch.long)
    batch_perm = {
        "context_ids": batch["context_ids"],
        "context_lengths": batch["context_lengths"],
        "option_ids": batch["option_ids"][:, perm, :],
        "option_lengths": batch["option_lengths"][:, perm],
    }

    with torch.no_grad():
        out_orig = model.forward_decision(**batch)
        out_perm = model.forward_decision(**batch_perm)

    assert torch.equal(out_orig.similarities[:, perm], out_perm.similarities)
    assert torch.equal(out_orig.logits[:, perm], out_perm.logits)
    assert torch.allclose(out_orig.probabilities[:, perm], out_perm.probabilities, atol=1e-7)


def test_absent_option_padding_invariance() -> None:
    """Adding a zero-length padded option slot must not alter valid option scores."""
    model = _tiny_model()
    request = {
        "state": "A cat carries one red ball.",
        "questions": {
            "has_ball": {
                "type": "noul",
                "instructions": "Is the animal carrying a ball?",
                "false_description": "The animal has no ball.",
                "true_description": "The animal carries a ball.",
            }
        },
    }
    examples = parse_request(request)
    batch = collate_decisions(examples, max_length=128)

    padded_opt_ids = torch.full(
        (1, 3, batch["option_ids"].shape[-1]),
        ByteTokenizer.pad_id,
        dtype=torch.long,
    )
    padded_opt_ids[:, :2, :] = batch["option_ids"]
    padded_opt_lengths = torch.tensor([[batch["option_lengths"][0, 0], batch["option_lengths"][0, 1], 0]])

    with torch.no_grad():
        out2 = model.forward_decision(**batch)
        out3 = model.forward_decision(
            context_ids=batch["context_ids"],
            context_lengths=batch["context_lengths"],
            option_ids=padded_opt_ids,
            option_lengths=padded_opt_lengths,
        )

    assert torch.equal(out2.similarities, out3.similarities[:, :2])
    assert torch.allclose(out2.probabilities, out3.probabilities[:, :2], atol=1e-7)
    assert out3.probabilities[0, 2].item() == 0.0


def test_0x8000_prompt_masking_and_mandatory_t_scratchpad(tmp_path: Path) -> None:
    """Verify 0x8000 high-bit prompt masking and mandatory #T$# scratchpad enforcement."""
    tok = ByteTokenizer()
    with pytest.raises(ValueError, match="mandatory"):
        tok.encode_masked_turn("What is 2+2?", "", "4", seq_len=64)

    block = tok.encode_masked_turn("What is 3*4?", "3 * 4 = 12.", "12", seq_len=64)
    assert len(block) == 64
    bin_path = tmp_path / "train.bin"
    # Write 2 blocks + 1 trailing token so len(dataset) == 2
    arr = np.asarray(block + block + [ByteTokenizer.pad_id], dtype="<u2")
    bin_path.write_bytes(arr.tobytes())

    ds = TokenBlockDataset(bin_path, seq_len=64)
    assert len(ds) == 2
    batch = ds.batch([0])
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    # High bit must be stripped in input_ids
    assert (input_ids < 260).all()
    decoded_in = tok.decode(input_ids)
    assert decoded_in.startswith("#U$# What is 3*4?\n#T$# 3 * 4 = 12.\n#A$# 12")

    # Prompt tokens and pad tokens must be masked as -100 in labels
    unmasked_target_tokens = [t for t in labels.tolist() if t != -100]
    decoded_target = tok.decode(unmasked_target_tokens)
    assert decoded_target == "#T$# 3 * 4 = 12.\n#A$# 12"


def test_coe_operators_and_novelty_buffer() -> None:
    """Verify Association + Novelty N(q), Deduction, Metacognition, Abstraction, and Inference."""
    model = _tiny_model()
    tok = ByteTokenizer()
    texts = ["#U$# Write a binary search.", "#U$# Write a merge sort.", "#U$# Prove sqrt(2) is irrational."]
    encoded = [tok.encode(t) for t in texts]
    max_len = max(map(len, encoded))
    item_ids = torch.full((len(texts), max_len), tok.pad_id, dtype=torch.long)
    item_lengths = torch.tensor([len(row) for row in encoded], dtype=torch.long)
    for i, row in enumerate(encoded):
        item_ids[i, : len(row)] = torch.tensor(row)

    history = torch.randn(1024, 16)
    with torch.no_grad():
        assoc = model.forward_association_matrix(item_ids, item_lengths, history_buffer=history)
        assert assoc.similarities.shape == (3, 3)
        assert assoc.novelty is not None and assoc.novelty.shape == (3,)

        ded = model.forward_deduction(
            item_ids[:2],
            item_lengths[:2],
            item_ids[1:],
            item_lengths[1:],
            targets=torch.tensor([0, 2]),
        )
        assert ded.probabilities.shape == (2, 3)
        assert ded.loss is not None and torch.isfinite(ded.loss)

        meta = model.forward_metacognition(
            item_ids[:2],
            item_lengths[:2],
            item_ids[1:],
            item_lengths[1:],
            targets=torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.5, 1.0]]),
        )
        assert meta.probabilities.shape == (2, 3)
        assert meta.loss is not None and torch.isfinite(meta.loss)

        summary = model.compress_abstraction(item_ids, item_lengths, level="brief")
        assert summary.shape == (3, 64)

        lm_out = model.backbone(item_ids, targets=item_ids)
        assert lm_out.logits.shape == (3, max_len, tok.vocab_size)
        assert lm_out.loss is not None and torch.isfinite(lm_out.loss)


def test_cpu_offload_adamw_step() -> None:
    """Verify CPUOffloadAdamW updates working weights and preserves FP32 masters on CPU and CUDA."""
    model = _tiny_model()
    prepare_model(model, device="cpu", bf16_working_weights=False)
    opt = CPUOffloadAdamW(model.parameters(), lr=1e-2, transfer_mode="per_parameter")
    x = torch.randint(0, 256, (2, 16), dtype=torch.long)
    out = model.backbone(x, targets=x)
    assert out.loss is not None
    initial_loss = out.loss.item()
    out.loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    out2 = model.backbone(x, targets=x)
    assert out2.loss is not None and out2.loss.item() < initial_loss
    assert pearson_similarity(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 4.0, 6.0])).item() == pytest.approx(1.0, abs=1e-5)

    if torch.cuda.is_available():
        cuda_model = _tiny_model().train()
        prepare_model(cuda_model, device="cuda", bf16_working_weights=True)
        cuda_opt = CPUOffloadAdamW(cuda_model.parameters(), lr=5e-3, transfer_mode="batched")
        x_cuda = x.to("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            c_out1 = cuda_model.backbone(x_cuda, targets=x_cuda)
        assert c_out1.loss is not None
        c_out1.loss.backward()
        cuda_opt.step()
        cuda_opt.zero_grad(set_to_none=True)
        assert all(torch.isfinite(p).all() for p in cuda_model.parameters())
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            c_out2 = cuda_model.backbone(x_cuda, targets=x_cuda)
        assert c_out2.loss is not None and torch.isfinite(c_out2.loss)
        assert c_out2.loss.item() < c_out1.loss.item()


def test_multipass_controller_and_joint_coe_loss() -> None:
    """Verify Milestone 3 MultiPassController, generate_uta, and compute_joint_coe_loss."""
    model = _tiny_model()
    sched = CognitiveSchedule.from_scaffold_text(
        "abduction -> association -> decision -> deduction -> evaluation -> metacognition",
        recurrence_steps=2,
    )
    controller = MultiPassController(model, schedule=sched)
    request = {
        "state": "A rabbit carries two red balls.",
        "questions": {
            "animal": {
                "type": "choice",
                "instructions": "Which animal is here?",
                "options": [
                    {"id": "rabbit", "description": "A rabbit."},
                    {"id": "cat", "description": "A cat."},
                ],
            }
        },
        "targets": {"animal": "rabbit"},
    }
    batch = collate_decisions(parse_request(request, require_targets=True), max_length=128)
    history = torch.randn(32, 16)
    out = controller(
        batch["context_ids"],
        batch["context_lengths"],
        batch["option_ids"],
        batch["option_lengths"],
        targets=batch["targets"],
        history_buffer=history,
    )
    assert out.selected_indices.shape == (1,)
    assert out.deduction is not None and out.deduction.probabilities.shape == (1, 3)
    assert out.evaluation is not None and out.evaluation.utilities.shape == (1,)
    assert out.metacognition is not None and out.metacognition.probabilities.shape == (1, 3)
    assert out.novelty is not None and out.novelty.shape == (1,)

    uta_text = controller.generate_uta("What is 2+2?", max_new_tokens=12)
    assert "#U$# What is 2+2?\n#T$# " in uta_text
    assert "\n#A$# " in uta_text

    lm_x = torch.randint(0, 256, (2, 16), dtype=torch.long)
    joint = compute_joint_coe_loss(
        model,
        lm_batch=(lm_x, lm_x),
        decision_batch=batch,
    )
    assert torch.isfinite(joint.total_loss)
    assert joint.lm_loss is not None and joint.decision_loss is not None


def test_ornith_10_and_15_rewards_and_grpo() -> None:
    """Verify Ornith-1.0 ScaffoldCache and Ornith-1.5 R_task, R_harness, and staleness GRPO."""
    buf = NoveltyTaskBuffer(projection_dim=16, max_capacity=64)
    v1 = torch.arange(16, dtype=torch.float32)
    buf.add(v1)
    assert len(buf) == 1
    assert buf.compute_novelty(v1).item() == pytest.approx(0.0, abs=1e-5)
    assert buf.compute_novelty(-v1).item() == pytest.approx(2.0, abs=1e-5)

    # Hard gate: V=0 => R_task=0 even if pass_rate=0.2 and novelty=1.0
    assert compute_task_reward(0.0, 0.2, 1.0).item() == 0.0
    # Frontier peak at p* = 0.2
    r_frontier = compute_task_reward(1.0, 0.2, 1.0).item()
    r_trivial = compute_task_reward(1.0, 1.0, 1.0).item()
    assert r_frontier == pytest.approx(1.0, abs=1e-6)
    assert r_frontier > r_trivial

    # Harness reward rejects negative-rollout leakage
    c = torch.tensor([1.0])
    h_scores = torch.tensor([[0.0, 0.5, 1.0]])
    ref_util = torch.tensor([[0.0, 0.5, 1.0]])
    assert compute_harness_reward(c, h_scores, ref_util).item() == pytest.approx(1.0, abs=1e-5)
    assert (
        compute_harness_reward(
            c, h_scores, ref_util, negative_harness_scores=torch.tensor([[0.2]])
        ).item()
        == 0.0
    )

    # Staleness weighting and topology invalidation
    w = staleness_weight(torch.tensor([0, 1, 2, 5]), k1=1, k2=3, decay_lambda=0.5)
    assert w[0].item() == 1.0 and w[1].item() == 1.0
    assert 0.0 < w[2].item() < 1.0
    assert w[3].item() == 0.0
    assert staleness_weight(0, topology_changed=True).item() == 0.0

    # ScaffoldCache monotonic update
    cache = ScaffoldCache()
    assert cache.update_if_better("t1", ["s_low", "s_high"], [0.1, 0.8], step=1)
    assert cache.get("t1") == "s_high"
    assert not cache.update_if_better("t1", ["s_worse"], [0.5], step=2)
    assert cache.get("t1") == "s_high"

    # GRPO loss
    adv = group_normalized_advantages(torch.tensor([0.0, 1.0, 2.0]))
    logp = torch.zeros((3, 4), requires_grad=True)
    beh_logp = torch.zeros((3, 4))
    loss = grpo_clipped_loss(logp, beh_logp, adv, policy_age=0)
    loss.backward()
    assert logp.grad is not None and torch.isfinite(logp.grad).all()


def test_expert_mitosis_and_pruning() -> None:
    """Verify grow_experts and prune_experts across encoder and decoder layers with CPUOffloadAdamW."""
    model = _tiny_model().train()
    prepare_model(model, device="cpu", bf16_working_weights=False)
    opt = CPUOffloadAdamW(model.parameters(), lr=1e-3, transfer_mode="per_parameter")

    res_grow = grow_experts(model, model.backbone, opt, parent_index=0)
    assert res_grow.old_num_experts == 4 and res_grow.new_num_experts == 5
    assert model.backbone.config.num_experts == 5

    res_prune = prune_experts(model, model.backbone, opt, expert_index=4)
    assert res_prune.old_num_experts == 5 and res_prune.new_num_experts == 4
    assert model.backbone.config.num_experts == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU for native NVFP4")
def test_nvfp4_linear_conversion_and_gemm_on_cuda() -> None:
    """Verify native Blackwell NVFP4 GEMM, 2:4 transposable sparsity, SR-STE, and causal scaling on RTX 5070."""
    from mcl4_coe import compute_2to4_mask

    # 1. Verify 2:4 transposable mask is exact 2:4 along BOTH rows (W) and columns (W.T)
    w_test = torch.randn(64, 128, device="cuda")
    mask = compute_2to4_mask(w_test, mode="2to4_transposable")
    assert mask.shape == (64, 128)
    assert torch.all(mask.view(64, 32, 4).sum(dim=-1) == 2)
    assert torch.all(mask.t().contiguous().view(128, 16, 4).sum(dim=-1) == 2)

    # 2. Verify causal_activation_scale=True prevents future tokens from leaking into earlier rows
    layer = NVFP4Linear(64, 64, bias=False, device="cuda", sparsity_mode="2to4_transposable").eval()
    seq_a = torch.randn(2, 16, 64, device="cuda")
    seq_b = seq_a.clone()
    seq_b[:, 8:, :] = seq_b[:, 8:, :] * 25.0 - 10.0
    with torch.no_grad():
        out_a = layer(seq_a)
        out_b = layer(seq_b)
    torch.testing.assert_close(out_a[:, :8, :], out_b[:, :8, :], rtol=0, atol=0)

    # 3. Verify expert-only conversion, 2:4 sparse NVFP4 GEMM execution, and SR-STE master weight update
    reset_gemm_counts()
    model = _tiny_model(device="cuda").train()
    num_converted = convert_linear_to_nvfp4(
        model.backbone,
        min_features=32,
        experts_only=True,
        sparsity_mode="2to4_transposable",
        sr_ste_decay=1e-3,
    )
    assert num_converted > 0
    assert any(isinstance(m, NVFP4Linear) for m in model.backbone.modules())
    # Routers and attention projections must remain standard high-precision nn.Linear
    for enc in model.backbone.encoder_layers:
        assert not isinstance(enc.moe.router, NVFP4Linear)
        assert not isinstance(enc.attention.qkv, NVFP4Linear)
        assert isinstance(enc.moe.experts[0].up_proj, NVFP4Linear)
        assert isinstance(enc.moe.experts[0].down_proj, NVFP4Linear)

    opt = torch.optim.AdamW(model.backbone.parameters(), lr=1e-2)
    w_before = model.backbone.encoder_layers[0].moe.experts[0].up_proj.weight.detach().clone()
    x = torch.randint(0, 256, (2, 16), device="cuda", dtype=torch.long)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model.backbone(x, targets=x)
    assert out.loss is not None
    out.loss.backward()
    opt.step()

    w_after = model.backbone.encoder_layers[0].moe.experts[0].up_proj.weight.detach()
    assert not torch.equal(w_before, w_after)
    counts = get_gemm_counts()
    assert counts["forward"] > 0 and counts["dgrad"] > 0 and counts["wgrad"] > 0
    assert counts["sparse_2to4"] > 0


def test_curation_byte_sieve_and_gliner25_gate() -> None:
    """Verify Stage-0 CPU byte sieve, marker sanitization, and Stage-1 GLiNER2.5-Decide gate."""
    # Too short -> rejected in Stage 0
    short_res = evaluate_byte_sieve("Too short.")
    assert not short_res.keep and short_res.stage_rejected == "stage0_byte_sieve"

    # High-quality algorithmic prose with accidental #T$# marker in raw text
    good_raw = (
        "In digital signal processing, a matched filter correlates a known template,\n"
        "or reference signal, with an unknown signal to detect the presence of the\n"
        "template in the unknown signal. Note that raw literal #T$# markers are escaped."
    )

    def _mock_gliner25(window: str, schema: dict) -> dict:
        assert "domain" in schema and "reasoning_density" in schema and "defects" in schema
        if "Buy now" in window:
            return {
                "domain": {"label": "seo_marketing_or_boilerplate", "confidence": 0.94},
                "reasoning_density": {"label": "low_value_filler_or_stub", "confidence": 0.91},
                "defects": [{"label": "paywall_or_navigation_chrome", "confidence": 0.88}],
            }
        return {
            "domain": {"label": "expository_prose_or_textbook", "confidence": 0.96},
            "reasoning_density": {"label": "high_logical_or_explanatory_density", "confidence": 0.92},
            "defects": [],
        }

    gate = GLiNER25CorpusGate(classifier_fn=_mock_gliner25)
    dec_good = gate.evaluate_document(good_raw)
    assert dec_good.keep
    assert dec_good.sanitized_text is not None and "#T$#" not in dec_good.sanitized_text
    unwrapped = normalize_prose_unwrapping(dec_good.sanitized_text)
    assert "\nor reference" not in unwrapped

    seo_spam = (
        "Buy now and subscribe to our premium newsletter for exclusive deals on cloud\n"
        "hosting, enterprise SaaS solutions, and limited-time promotional discounts today!"
    )
    dec_bad = gate.evaluate_document(seo_spam)
    assert not dec_bad.keep and dec_bad.stage_rejected == "stage1_gliner25_decide"


def test_sharded_pipeline_and_index_receipt(tmp_path: Path) -> None:
    """Verify BlockAlignedShardWriter, per-shard receipts, IndexReceipt, and ShardedTokenBlockDataset."""
    from mcl4_coe import (
        BlockAlignedShardWriter,
        EpochBlockSampler,
        HostBatchPrefetcher,
        ShardedTokenBlockDataset,
        extract_typed_decision_uta,
        extract_uta_episodes,
    )

    writer = BlockAlignedShardWriter(
        tmp_path,
        track="uta_trace",
        split="train",
        seq_len=256,
        max_blocks_per_shard=2,
    )

    episodes = extract_uta_episodes(
        {
            "messages": [
                {"role": "user", "content": "What is 6 * 7?"},
                {"role": "assistant", "content": "<think>6 * 7 = 42.</think>\n42"},
            ]
        }
    )
    assert len(episodes) == 1
    dec_ep = extract_typed_decision_uta(
        {
            "state": "A rabbit carries two red balls.",
            "kind": "choice",
            "question": "Which animal is present?",
            "options": ["A rabbit.", "A cat."],
            "target": [1.0, 0.0],
        }
    )
    assert dec_ep is not None

    for i in range(10):
        u, t, a = episodes[0]
        assert writer.add_uta_episode(f"{u} ({i})", t, a)
    assert writer.add_uta_episode(*dec_ep)
    # Oversize trace (>256 bytes) must be rejected rather than severed mid-thought
    assert not writer.add_uta_episode("Q" * 200, "T" * 100, "A")
    writer.record_provenance(
        corpus="synthetic_test",
        source_file="test.jsonl",
        row_start=1,
        row_end=7,
        episodes_accepted=7,
        episodes_rejected_stage0=0,
    )
    writer.flush_shard()

    index_file = tmp_path / "index_receipt.json"
    assert index_file.exists()
    ds = ShardedTokenBlockDataset(index_file, track="uta_trace", split="train")
    assert len(ds) >= 2
    assert len(ds.shards) >= 1

    sampler = EpochBlockSampler(len(ds), seed=42)
    prefetcher = HostBatchPrefetcher(ds, micro_batch_size=2, pin_memory=False)
    try:
        batches = prefetcher.get(sampler.take(len(ds)))
        tok = ByteTokenizer()
        first_in = tok.decode(batches[0]["input_ids"][0])
        assert "#U$# " in first_in and "#T$# " in first_in and "#A$# " in first_in
        unmasked = [t for t in batches[0]["labels"][0].tolist() if t != -100]
        decoded_tgt = tok.decode(unmasked)
        # Prompt (#U$# ...) must be masked out of targets; only #T$# and #A$# are supervised
        assert decoded_tgt.startswith("#T$# ")
    finally:
        prefetcher.close()


def test_multitrack_mixture_and_curriculum_sampler(tmp_path: Path) -> None:
    """Verify MultiTrackMixtureDataset, CurriculumMixtureSchedule, and CurriculumMixtureSampler."""
    from mcl4_coe import (
        BlockAlignedShardWriter,
        CurriculumMixtureSampler,
        CurriculumMixtureSchedule,
        HostBatchPrefetcher,
        MultiTrackMixtureDataset,
    )

    track_receipts: dict[str, Path] = {}
    for t_name in ("pretrain_lm", "uta_trace", "typed_decision"):
        t_dir = tmp_path / t_name
        w = BlockAlignedShardWriter(
            t_dir,
            track=t_name,
            split="train",
            seq_len=128,
            max_blocks_per_shard=4,
        )
        for i in range(6):
            if t_name == "pretrain_lm":
                w.add_pretrain_document(f"Expository theorem {i}: energy is conserved in closed systems. " * 3)
            else:
                w.add_uta_episode(f"Compute {i}+1", f"{i} + 1 = {i+1}", str(i + 1))
        w.flush_shard()
        track_receipts[t_name] = t_dir / "index_receipt.json"

    mix_ds = MultiTrackMixtureDataset(track_receipts)
    assert len(mix_ds) > 0
    sched = CurriculumMixtureSchedule(
        start_weights={"pretrain_lm": 0.7, "uta_trace": 0.2, "typed_decision": 0.1},
        end_weights={"pretrain_lm": 0.1, "uta_trace": 0.5, "typed_decision": 0.4},
        mode="cosine",
    )
    sampler = CurriculumMixtureSampler(mix_ds, sched, seed=99)

    q_start = sampler.allocate_quotas(10, progress=0.0)
    assert q_start == {"pretrain_lm": 7, "uta_trace": 2, "typed_decision": 1}
    q_end = sampler.allocate_quotas(10, progress=1.0)
    assert q_end == {"pretrain_lm": 1, "uta_trace": 5, "typed_decision": 4}

    idx_batch = sampler.take(10, progress=0.5)
    assert idx_batch.shape == (10,)

    # Verify state_dict checkpoint round-trip
    saved_state = sampler.state_dict()
    next_expected = sampler.take(6, progress=0.8)
    sampler.load_state_dict(saved_state)
    next_restored = sampler.take(6, progress=0.8)
    assert torch.equal(next_expected, next_restored)

    prefetcher = HostBatchPrefetcher(mix_ds, micro_batch_size=5, pin_memory=False)
    try:
        batches = prefetcher.get(idx_batch)
        assert len(batches) == 2
        assert batches[0]["input_ids"].shape == (5, 128)
        assert batches[0]["track_ids"].shape == (5,)
    finally:
        prefetcher.close()


def test_trainer_checkpoints_and_webui_endpoints(tmp_path: Path) -> None:
    """Verify MCL4COETrainer, checkpoint receipts, and Web UI REST endpoints."""
    import json
    import threading
    from urllib.request import Request, urlopen

    from mcl4_coe import (
        BlockAlignedShardWriter,
        MCL4COETrainer,
        TrainingControllerState,
        TrainingRunConfig,
        list_checkpoints,
        list_runs,
        load_run_metrics,
        start_web_ui_server,
    )

    shards_dir = tmp_path / "shards"
    runs_dir = tmp_path / "runs"
    for t_name in ("pretrain_lm", "uta_trace", "typed_decision"):
        w = BlockAlignedShardWriter(
            shards_dir / t_name,
            track=t_name,
            split="train",
            seq_len=64,
            max_blocks_per_shard=4,
        )
        for i in range(4):
            if t_name == "pretrain_lm":
                w.add_pretrain_document(f"Theorem {i}: conservation of momentum in isolated frames.")
            else:
                w.add_uta_episode(f"Add {i}+2", f"{i}+2={i+2}", str(i + 2))
        w.flush_shard()

    cfg = TrainingRunConfig(
        run_id="unit_test_run",
        runs_root=str(runs_dir),
        shards_root=str(shards_dir),
        corpora_root=str(tmp_path / "no_corpora"),
        d_model=32,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ple_dim=8,
        expert_hidden_dim=64,
        num_experts=2,
        top_k=1,
        max_seq_len=64,
        projection_dim=8,
        total_steps=2,
        batch_size=3,
        micro_batch_size=3,
        checkpoint_every=2,
        sample_every=2,
    )
    trainer = MCL4COETrainer(cfg)
    final_m = trainer.train()
    assert final_m["step"] == 2
    assert "loss_pretrain_lm" in final_m

    runs = list_runs(runs_dir)
    assert len(runs) == 1 and runs[0]["run_id"] == "unit_test_run"
    run_data = load_run_metrics("unit_test_run", runs_dir)
    assert len(run_data["metrics"]) == 2
    ckpts = list_checkpoints(runs_dir)
    assert len(ckpts) == 1

    controller = TrainingControllerState(
        runs_root=runs_dir,
        shards_root=shards_dir,
        corpora_root=tmp_path / "no_corpora",
    )
    server = start_web_ui_server(controller, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{port}"
        html = urlopen(f"{base_url}/").read().decode("utf-8")
        assert "MCL4-COE Training" in html

        st = json.loads(urlopen(f"{base_url}/api/status").read().decode("utf-8"))
        assert "is_running" in st and "pretrain_lm" in st["shards"]

        eval_req = Request(
            f"{base_url}/api/checkpoints/eval",
            data=json.dumps({"checkpoints": [ckpts[0]["meta_file"]], "prompt": "Add 1+2"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        eval_res = json.loads(urlopen(eval_req).read().decode("utf-8"))
        assert len(eval_res["results"]) == 1
        assert "#U$# Add 1+2" in eval_res["results"][0]["trace"]
    finally:
        server.shutdown()
        server.server_close()

    # Verify seamless 2E -> 4E Expert Mitosis upcycling when resuming from checkpoint
    upcycle_cfg = TrainingRunConfig(
        run_id="unit_test_upcycled_4e",
        runs_root=str(runs_dir),
        shards_root=str(shards_dir),
        corpora_root=str(tmp_path / "no_corpora"),
        resume_checkpoint=ckpts[0]["meta_file"],
        d_model=32,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ple_dim=8,
        expert_hidden_dim=64,
        num_experts=4,
        top_k=2,
        max_seq_len=64,
        projection_dim=8,
        total_steps=3,
        batch_size=3,
        micro_batch_size=3,
        checkpoint_every=3,
        sample_every=3,
    )
    upcycled_trainer = MCL4COETrainer(upcycle_cfg)
    assert upcycled_trainer.model.backbone.config.num_experts == 4
    up_m = upcycled_trainer.train()
    assert up_m["step"] == 3
    assert up_m["num_experts"] == 4
    assert len(up_m["expert_load"]) == 4


def test_forward_all_operators_and_ornith_self_improver() -> None:
    """Verify 9-operator joint pass (`forward_all_operators`), `OrnithSelfImprover`, and `<think>` compaction."""
    from mcl4_coe import (
        DecisionExample,
        OrnithSelfImprover,
        collate_decisions,
        extract_uta_episodes,
    )

    model = _tiny_model().train()
    examples = [
        DecisionExample(
            question_id="q1",
            kind="choice",
            context="#U$# A rabbit carries two red balls.\n#Q$# Which animal is present?",
            descriptions=("#T$# Grounded in premise.\n#A$# rabbit", "#T$# Contradicts premise.\n#A$# cat"),
            values=("rabbit", "cat"),
            target=0,
        ),
        DecisionExample(
            question_id="q2",
            kind="choice",
            context="#U$# Three servers process 12 requests each.\n#Q$# Total requests?",
            descriptions=("#T$# 3 + 12 = 15.\n#A$# 15", "#T$# 3 * 12 = 36.\n#A$# 36"),
            values=("15", "36"),
            target=1,
        ),
    ]
    batch = collate_decisions(examples, max_length=96)
    model.zero_grad(set_to_none=True)
    op_out = model.forward_all_operators(**batch)
    assert torch.isfinite(op_out.total_loss)
    assert torch.isfinite(op_out.deduction_loss)
    assert torch.isfinite(op_out.evaluation_loss)
    assert torch.isfinite(op_out.metacognition_loss)
    assert torch.isfinite(op_out.abstraction_loss)
    op_out.total_loss.backward()
    assert model.deduction_query_proj.weight.grad is not None
    assert model.deduction_prototypes.grad is not None
    assert model.meta_query_proj.weight.grad is not None
    assert model.assoc_alpha.grad is not None
    assert model.abstraction_proj.weight.grad is not None

    # Verify OrnithSelfImprover 3-stage self-improvement loop
    improver = OrnithSelfImprover(projection_dim=model.config.projection_dim, max_seq_len=96)
    model.zero_grad(set_to_none=True)
    ornith_out = improver.step_self_improvement(model)
    assert torch.isfinite(ornith_out.loss)
    assert 0.0 <= ornith_out.task_reward <= 2.0
    assert 0.0 <= ornith_out.harness_reward <= 1.0
    assert ornith_out.novelty_buffer_size >= 1
    assert ornith_out.scaffolds_cached >= 1
    ornith_out.loss.backward()

    # Verify verbose <think> monologue AND oversized post-</think> answer compaction fit inside seq_len=2048
    verbose_think = (
        "<think>"
        + ("Step derivation: let x_i = i^2 + 1.\n" * 80)
        + "</think>\n"
        + ("Detailed verification paragraph with UTF-8 math symbols (∀x ∈ ℝ, x² ≥ 0).\n" * 40)
    )
    eps = extract_uta_episodes(
        {
            "messages": [
                {"role": "user", "content": "Compute the sequence limit."},
                {"role": "assistant", "content": verbose_think},
            ]
        }
    )
    assert len(eps) == 1
    u, t, a = eps[0]
    tok_ids = ByteTokenizer().encode_masked_turn(u, t, a, seq_len=2048)
    assert len(tok_ids) <= 2048
    assert "\n...\n" not in t and t.endswith(".")

    # Verify Nemotron-Math-Proofs-v2 and hh-rlhf extraction schemas
    nemotron_eps = extract_uta_episodes(
        {
            "problem": "Prove that n + 0 = n in Lean 4.",
            "lean4_formalization": "theorem add_zero (n : Nat) : n + 0 = n := by rfl",
            "messages": [
                {"role": "user", "content": "Formalize and prove n + 0 = n."},
                {
                    "role": "assistant",
                    "reasoning_content": "By definition of Nat.add on the second argument, n + 0 reduces judgmentally to n.",
                    "content": "```lean4\ntheorem add_zero (n : Nat) : n + 0 = n := by rfl\n```",
                },
            ],
        }
    )
    assert len(nemotron_eps) == 1
    assert "Nat.add" in nemotron_eps[0][1]

    hh_eps = extract_uta_episodes(
        {
            "chosen": "\n\nHuman: How do I reverse a list in Python?\n\nAssistant: First consider slicing vs in-place reversal. You can use `my_list[::-1]` to return a reversed copy."
        }
    )
    assert len(hh_eps) == 1
    assert "reverse a list" in hh_eps[0][0]


def test_dream_rsi_simulator_and_mcl4_coe_integration(tmp_path: Path) -> None:
    """Verify Dream-RSI prefix-observable trees, zero-execution replay, Eq. 1 Pareto scoring,
    monotonic policy improvement, and end-to-end integration with MCL4-COE & Ornith-1.5.
    """
    from mcl4_coe import (
        DreamNode,
        DreamOutcome,
        DreamProgram,
        DreamRSIEngine,
        DreamWorld,
        OrnithSelfImprover,
        dream_online_rollout,
        dream_replay,
        improve_dream_policy,
        score_dream_policy,
    )

    # 1. Verify Program DSL parsing, canonical serialization, andCOE operator rank features
    prog_text = (
        "WIDTH 2\n"
        "ROOT 1\n"
        "RANK SCORE GAIN UTILITY CONFIDENCE\n"
        "STOP 0.9\n"
        "PATIENCE 3\n"
        "ROUNDS 4"
    )
    prog = DreamProgram.from_text(prog_text)
    assert prog.to_text() == prog_text
    assert prog.width == 2
    assert prog.rank == ("SCORE", "GAIN", "UTILITY", "CONFIDENCE")

    # 2. Verify online discovery tree construction and zero-execution historical replay
    call_counter = 0

    def _mock_discover(view, parent_id: int, node_id: int) -> DreamOutcome:
        nonlocal call_counter
        call_counter += 1
        # Branch 1 (first root child) starts at 0.30 and improves to 0.95 on refinement;
        # Branch 2 starts at 0.45 but plateaus at 0.46.
        if parent_id == 0:
            root_kids = sum(1 for n in view.nodes[1:] if n.parent == 0)
            sc = 0.30 if root_kids == 0 else 0.45
            util = 0.95 if root_kids == 0 else 0.40
        else:
            parent_node = next(n for n in view.nodes if n.id == parent_id)
            sc = 0.95 if parent_node.score == 0.30 else 0.46
            util = 0.98 if parent_node.score == 0.30 else 0.42
        return DreamOutcome(
            score=sc,
            observation={"utility": util, "confidence": util, "novelty": 0.5},
        )

    broad_policy = DreamProgram(width=2, root_every=1, rank=("SCORE", "GAIN"), stop=None, patience=0, rounds=4)
    online = dream_online_rollout(
        broad_policy,
        _mock_discover,
        world_id="w_test",
        root_observation={"task": "synth"},
        metadata={"test": True},
        max_rounds=4,
        workers=2,
        max_calls=6,
        rho=1.0,
    )
    assert online.agent_calls == call_counter
    assert len(online.world.nodes) == online.agent_calls + 1

    # Historical replay must perform ZERO discover calls while traversing the stored tree
    before_calls = call_counter
    rep = dream_replay(
        online.world,
        prog,
        rho=1.0,
        max_rounds=4,
        workers=2,
        max_calls=6,
    )
    assert call_counter == before_calls  # Zero real executions during replay!
    assert rep.best_score == 0.95

    # 3. Verify Eq. 1 Pareto hypervolume scoring & monotonic policy improvement (pi^{t+1} >= pi^t)
    incumbent = DreamProgram(width=1, root_every=1, rank=("SHALLOW",), stop=None, patience=0, rounds=4)
    better_candidate = DreamProgram(width=2, root_every=2, rank=("UTILITY", "GAIN", "SCORE"), stop=0.9, patience=2, rounds=4)

    proposals = [
        {"text": "INVALID POLICY SYNTAX", "finished": True},
        {"text": better_candidate.to_text(), "finished": True},
    ]

    def _propose(_inc, _fb, rev: int):
        return proposals[rev - 1]

    imp = improve_dream_policy(
        [online.world],
        incumbent,
        _propose,
        revisions=2,
        max_rounds=4,
        workers=2,
        max_calls=6,
    )
    assert imp["final_score"] >= imp["initial_score"]
    assert imp["accepted_revision"] == 2
    assert imp["program_text"] == better_candidate.to_text()

    # 4. Verify end-to-end DreamRSIEngine + OrnithSelfImprover on MCL4-COE
    model = _tiny_model().train()
    dream_dir = tmp_path / "dream_out"
    engine = DreamRSIEngine(
        max_rounds=2,
        workers=2,
        max_calls=3,
        revisions_per_cycle=2,
        out_dir=dream_dir,
    )
    improver = OrnithSelfImprover(
        projection_dim=model.config.projection_dim,
        max_seq_len=96,
        enable_dream_rsi=True,
        dream_every=1,
        dream_engine=engine,
    )
    model.zero_grad(set_to_none=True)
    out1 = improver.step_self_improvement(model)
    assert out1.dream_cycle == 1
    assert out1.dream_worlds_archived == 1
    assert out1.dream_replays_dreamed > 0
    assert "WIDTH " in out1.dream_active_policy
    assert (dream_dir / "dream_state.json").is_file()
    assert (dream_dir / "dream_policy.json").is_file()
    assert (dream_dir / "dream_revisions.jsonl").is_file()
    out1.loss.backward()


def test_context_memory_hierarchy_and_atxy_router(tmp_path) -> None:
    """Verify Tier 2 (TurboQuant), Tier 3 (ContextLMCompressor), and Tier 4 (A.TXY + COEAddressRouter)."""
    from mcl4_coe import (
        ATXYAddress,
        ATXYFactorizedEmbedding,
        ATXYPageStore,
        ATXYPointerExecutor,
        ATXYRecord,
        COEAddressRouter,
        ContextLMCompressor,
        compress_encoded_state,
        index_python_repo_to_atxy,
        parse_atxy,
        turboquant_compress,
        turboquant_inner_product,
        unpack_atxy,
    )

    model = _tiny_model().train()
    x = torch.randint(0, 256, (2, 48), dtype=torch.long)
    valid = torch.ones_like(x, dtype=torch.bool)

    # 1. Tier 2: TurboQuant KV / EncodedState compression
    enc = model.backbone.encode_state(x, valid)
    comp_state = compress_encoded_state(enc)
    assert comp_state.compressed_hidden.compression_ratio > 3.5
    decomp_enc = comp_state.decompress()
    cos = torch.nn.functional.cosine_similarity(
        enc.hidden.flatten(1), decomp_enc.hidden.flatten(1), dim=-1
    )
    assert float(cos.mean().item()) > 0.85

    q = torch.randn(2, 6, 32)
    k = torch.randn(2, 16, 32)
    ck = turboquant_compress(k)
    approx_scores = turboquant_inner_product(q, ck)
    exact_scores = torch.matmul(q, k.transpose(-1, -2))
    score_corr = torch.nn.functional.cosine_similarity(
        approx_scores.flatten(1), exact_scores.flatten(1), dim=-1
    )
    assert float(score_corr.mean().item()) > 0.85

    # 2. Tier 3: ContextLMCompressor multi-chunk recurrent state compression
    ctx_lm = ContextLMCompressor(
        d_model=model.backbone.config.d_model,
        chunk_size=16,
        gist_tokens_per_chunk=4,
    )
    gist_enc = ctx_lm(model.backbone, x, valid)
    assert gist_enc.hidden.shape == (2, 12, model.backbone.config.d_model)
    assert gist_enc.valid.shape == (2, 12)

    # 3. Tier 4: A.TXY Virtual Memory PageStore, N5 WAL, Factorized Embedding, Pointer Executor & COE Router
    addr = parse_atxy("1.G0B")
    assert str(addr) == "1.G0B"
    assert unpack_atxy(addr.packed) == addr

    fact_emb = ATXYFactorizedEmbedding(d_model=model.backbone.config.d_model)
    emb_out = fact_emb.encode_addresses(["1.G0B", "2.010"])
    assert emb_out.shape == (2, model.backbone.config.d_model)

    repo_root = Path(__file__).resolve().parents[1]
    store = index_python_repo_to_atxy(repo_root, subsystems=("src",))
    assert len(store) > 4

    # Verify N5 append-only WAL persistence + CRC32 verification
    wal_path = tmp_path / "atxy_wal.log"
    first_rec = store.records()[0]
    wal_rec = store.commit_write(
        ATXYRecord(
            address=first_rec.address,
            kind=first_rec.kind,
            payload="wal_verified_node",
            summary="Updated via N5 WAL",
            up=first_rec.up,
            down=first_rec.down,
            fields={"lineno": 42},
        ),
        wal_path=wal_path,
    )
    assert wal_rec.payload == "wal_verified_node"
    assert wal_rec.version == first_rec.version + 1
    store_reloaded = ATXYPageStore()
    replayed = store_reloaded.load_wal(wal_path)
    assert replayed == 1
    assert store_reloaded.lookup(first_rec.address).payload == "wal_verified_node"

    # Verify N6 single-register pointer executor
    executor = ATXYPointerExecutor(store)
    kind_out, val_out = executor.run(
        [
            ("READ", first_rec.address, None),
            ("PROJECT", first_rec.address, "lineno"),
            ("FIND", None, {"plane": 1}),
            ("COUNT", "REG", None),
        ]
    )
    assert kind_out == "scalar"
    assert isinstance(val_out, int) and val_out > 0
    assert len(executor.audit) == 4

    # Verify N3b-primary natural-language -> A.TXY routing via COE Association + Decision
    router = COEAddressRouter(store, max_seq_len=128)
    route_out = router.route_query(
        model,
        "Locate wal_verified_node and module definitions in mcl4_coe",
        top_k_candidates=4,
    )
    assert route_out["address"] in route_out["candidates"]
    assert len(route_out["candidates"]) <= 4


def test_verify_python_rollout_and_portable_shard_healing(tmp_path) -> None:
    """Verify sandboxed Python code execution reward (`verify_python_rollout`) and portable shard self-healing."""
    import shutil
    from mcl4_coe import (
        MultiTrackMixtureDataset,
        ensure_sharded_tracks_ready,
        verify_python_rollout,
    )

    harness_h = "assert solve(3, 4) == 14 and solve(0, 5) == 2"

    good_rollout = (
        "#T$#We compute a * b + 2 and verify both assertions.#A$#"
        "def solve(a, b): return a * b + 2"
    )
    assert verify_python_rollout(good_rollout, harness_h) == 1.0

    # Missing #T$# scratchpad must return 0.0
    no_thought_rollout = "#A$#def solve(a, b): return a * b + 2"
    assert verify_python_rollout(no_thought_rollout, harness_h) == 0.0

    # Constant boundary-hack stub must be rejected by anti-shortcut validator or second assertion
    stub_rollout = "#T$#Constant stub return 14.#A$#def solve(a, b): return 14"
    assert verify_python_rollout(stub_rollout, harness_h) == 0.0

    # Verify ensure_sharded_tracks_ready rebuilds missing .bin shards automatically
    shards_copy = tmp_path / "data_shards"
    repo_shards = Path(__file__).resolve().parents[1] / "data_shards"
    for track in ("pretrain_lm", "uta_trace", "typed_decision"):
        (shards_copy / track).mkdir(parents=True, exist_ok=True)
        src_receipt = repo_shards / track / "index_receipt.json"
        if src_receipt.is_file():
            shutil.copyfile(src_receipt, shards_copy / track / "index_receipt.json")

    healed_receipts = ensure_sharded_tracks_ready(shards_copy, seq_len=128)
    assert set(healed_receipts.keys()) == {"pretrain_lm", "uta_trace", "typed_decision"}
    ds = MultiTrackMixtureDataset(healed_receipts)
    assert len(ds) > 0
    assert (shards_copy / "typed_decision" / "decision_examples_cache.pt").is_file()


def test_nemotron_core_schema_extractors() -> None:
    """Verify Nemotron-Core schema extractors across pretrain_lm, uta_trace, and typed_decision."""
    from mcl4_coe import (
        extract_pretrain_text,
        extract_typed_decision_uta,
        extract_uta_episodes,
        parse_mcqa_decision_fields,
    )

    # 1. Nemotron-MIND (`Text` capital-T column)
    mind_row = {
        "Document ID": "OWM-1",
        "Text": (
            "Participant 1: Let us derive the eigenvalues of a real symmetric 2x2 matrix step by step.\n"
            "Participant 2: Because the matrix is symmetric, the discriminant of its characteristic "
            "polynomial is guaranteed to be non-negative, yielding real eigenvalues."
        ),
    }
    text_out, rej = extract_pretrain_text(mind_row)
    assert not rej and text_out is not None and "eigenvalues" in text_out

    # 2. Redacted user prompt in Nemotron-Post-Training-Dataset-v1 (`content == "-"`) must be rejected
    redacted_row = {
        "messages": [
            {"role": "user", "content": "-"},
            {"role": "assistant", "content": "<think>Some thought</think>Some answer"},
        ]
    }
    assert extract_uta_episodes(redacted_row) == []

    # 3. Nemotron-SFT-Code (`text` formatted as `input: ... output: ...`)
    sft_row = {
        "text": (
            "input: Write a Python function to reverse a list.\n"
            "output: First, slice the list with step -1.\n\ndef rev(xs):\n    return xs[::-1]"
        )
    }
    eps = extract_uta_episodes(sft_row)
    assert len(eps) == 1
    assert "reverse a list" in eps[0][0]
    assert "xs[::-1]" in eps[0][2]

    # 4. Nemotron-RL-knowledge-mcqa
    rl_mcqa_row = {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": (
                        "Answer the following multiple choice question.\n\n"
                        "Which mechanism guarantees O(T) complexity in MCL4-COE?\n"
                        "A: Full quadratic attention\n"
                        "B: Markov Local-4 Router\n"
                        "C: Unbounded recursion\n"
                        "D: Dense all-to-all MLP"
                    ),
                }
            ]
        },
        "expected_answer": "B",
        "options": [
            {"A": "Full quadratic attention", "B": None, "C": None, "D": None},
            {"A": None, "B": "Markov Local-4 Router", "C": None, "D": None},
            {"A": None, "B": None, "C": "Unbounded recursion", "D": None},
            {"A": None, "B": None, "C": None, "D": "Dense all-to-all MLP"},
        ],
    }
    parsed_rl = parse_mcqa_decision_fields(rl_mcqa_row)
    assert parsed_rl is not None
    assert parsed_rl[4] == 1
    assert parsed_rl[3][1] == "Markov Local-4 Router"

    # 5. Nemotron-CrossThink
    crossthink_row = {
        "meta_data": {
            "question": (
                "Which transform precedes Lloyd-Max scalar quantization in TurboQuant?\n"
                "A: Fast Fourier Transform\n"
                "B: Fast Walsh-Hadamard Transform\n"
                "C: Discrete Cosine Transform"
            )
        },
        "reward_model": {"ground_truth": "B: Fast Walsh-Hadamard Transform"},
    }
    parsed_ct = parse_mcqa_decision_fields(crossthink_row)
    assert parsed_ct is not None
    assert parsed_ct[4] == 1

    # 6. Nemotron-Pretraining-Specialized-v1.1 (`Nemotron-Pretraining-Multiple-Choice`)
    spec_mc_row = {
        "text": (
            "In the A.TXY addressing system, how many total addresses exist across 6 planes of 6x6x6 cubes?\n"
            "A. 216\n"
            "B. 1296\n"
            "C. 4096\n"
            "D. 65536\n"
            "Answer: B\n"
            "Because each of the 6 planes contains 6*6*6 = 216 addresses, 6 * 216 = 1296 total addresses."
        )
    }
    uta_mc = extract_typed_decision_uta(spec_mc_row)
    assert uta_mc is not None
    assert "1296 total addresses" in uta_mc[1]
    assert "[1] 1296" in uta_mc[2]


def test_antidoom_ftpo_and_housekeeping_fixes() -> None:
    """Verify LiquidAI Antidoom FTPO loop detection & loss, <think> tag sanitization, and boilerplate removal."""
    from mcl4_coe import (
        compute_antidoom_ftpo_loss,
        detect_doom_loop,
        extract_typed_decision_uta,
        sanitize_trace_markers,
    )

    # 1. Strip <think> and </think> tags
    cleaned = sanitize_trace_markers("<think> First derive the recurrence. </think> Therefore x = 4.")
    assert "<think>" not in cleaned and "</think>" not in cleaned

    # 2. Verify no synthetic "state evidence" boilerplate in extract_typed_decision_uta
    no_rat_row = {
        "state": "Markov Local-4 routes each token to 4 local causal windows.",
        "kind": "choice",
        "question": "Which complexity does Markov Local-4 achieve?",
        "options": ["O(T^2)", "O(T)", "O(T^3)", "O(2^T)"],
        "target": 1,
    }
    uta_no_rat = extract_typed_decision_uta(no_rat_row, require_rationale=False)
    assert uta_no_rat is not None
    assert "state evidence" not in uta_no_rat[1]
    assert extract_typed_decision_uta(no_rat_row, require_rationale=True) is None

    # 3. Verify Antidoom inner repetition detection and FTPO preference loss
    repeated_bytes = list(b"Compute step:" + b" state evidence," * 5)
    loop_hit = detect_doom_loop(repeated_bytes, min_period=4, min_repeats=3)
    assert loop_hit is not None
    start_idx, rejected_tok, period = loop_hit
    assert period == len(b" state evidence,")
    assert rejected_tok == repeated_bytes[start_idx]

    logits = torch.zeros(1, 260, requires_grad=True)
    ref_logits = logits.detach().clone()
    rej_t = torch.tensor([rejected_tok], dtype=torch.long)
    chosen_t = torch.tensor([[ord("a"), ord("b"), ord("c"), ord("d")]], dtype=torch.long)
    ftpo_loss, chosen_win = compute_antidoom_ftpo_loss(
        logits,
        ref_logits,
        rej_t,
        chosen_t,
        beta=0.5,
        target_chosen_win=1.0,
    )
    assert ftpo_loss.requires_grad
    ftpo_loss.backward()
    assert logits.grad is not None
    assert float(logits.grad[0, rejected_tok].item()) > 0.0  # gradient pushes rejected token logit down
    assert 0.0 <= chosen_win <= 1.0

    # 4. Verify Exact-Ratio Twin-Gated Mitosis preserves forward CE loss & serializes mitotic_pairs
    from mcl4_coe.trainer import compute_mitotic_schedule, upcycle_state_dict_experts

    torch.manual_seed(7)
    cfg4 = MCL4CEDConfig(
        d_model=64,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ple_dim=16,
        expert_hidden_dim=128,
        num_experts=4,
        top_k=2,
        max_seq_len=64,
    )
    m4 = CognitiveOperatorEnsemble(MCL4CEDBackbone(cfg4), COEConfig(projection_dim=16))
    m4.eval()
    inp = torch.randint(4, 120, (2, 32))
    lbl = torch.randint(4, 120, (2, 32))
    with torch.no_grad():
        ce4 = float(m4.forward_inference(inp, targets=lbl).ce_loss.item())

    loads4 = [0.10, 0.45, 0.35, 0.10]
    pairs_4to6 = compute_mitotic_schedule(4, 6, expert_loads=loads4)
    assert pairs_4to6 == [(1, 4), (2, 5)]
    sd6 = upcycle_state_dict_experts(m4.state_dict(), 4, 6, expert_loads=loads4, seed=42)
    cfg6 = MCL4CEDConfig(
        d_model=64,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ple_dim=16,
        expert_hidden_dim=128,
        num_experts=6,
        top_k=2,
        max_seq_len=64,
    )
    m6 = CognitiveOperatorEnsemble(MCL4CEDBackbone(cfg6), COEConfig(projection_dim=16))
    m6.load_state_dict(sd6, strict=True)
    assert m6.backbone.encoder_layers[0].moe.mitotic_pairs == [(1, 4), (2, 5)]
    m6.eval()
    with torch.no_grad():
        ce6 = float(m6.forward_inference(inp, targets=lbl).ce_loss.item())
    assert abs(ce6 - ce4) < 2e-3


def test_morphcode_16k_tokenizer_and_backbone(tmp_path) -> None:
    """Verify MorphCode-16K (`vocab_size=16384`) lossless round-trip, morpheme splitting, factorized PLE, and shard transcoding."""
    from mcl4_coe import (
        MORPHCODE_16K_VOCAB_SIZE,
        MorphCodeTokenizer,
        MultiPassController,
        ShardedTokenBlockDataset,
        get_tokenizer,
        transcode_byte_shards_to_morphcode,
    )
    from mcl4_coe.pipeline import BlockAlignedShardWriter

    mt = get_tokenizer(MORPHCODE_16K_VOCAB_SIZE)
    assert isinstance(mt, MorphCodeTokenizer)
    assert mt.vocab_size == 16384
    assert (mt.pad_id, mt.bos_id, mt.eos_id, mt.unk_id) == (256, 257, 258, 259)

    # 1. Atomic control markers + morpheme decomposition + lossless round-trip
    uta_sample = (
        "#U$# How do plants use capillary action to transport water?\n"
        "#T$# Plants use adhesion and cohesion in narrow xylem tubes.\n"
        "#A$# Capillary action pulls water upward against gravity."
    )
    ids = mt.encode(uta_sample)
    assert mt.decode(ids) == uta_sample
    assert mt.compression_ratio(uta_sample) > 4.0
    tokens = mt.tokenize_to_strings(uta_sample)
    assert tokens[0] == "#U$#"
    assert "\n#T$#" in tokens and "\n#A$#" in tokens
    # Verify inflectional morpheme splitting (`plant + s`, `adhes + ion`, `cohes + ion`)
    assert " plant" in tokens and "s" in tokens
    assert " adhes" in tokens and "ion" in tokens

    # 2. Single-digit splitting for arithmetic and Python code operators
    math_sample = "Solve 3 * x + 5 = 20 for x == 5 using \\frac{15}{3}."
    math_ids = mt.encode(math_sample)
    assert mt.decode(math_ids) == math_sample
    math_toks = mt.tokenize_to_strings(math_sample)
    assert "2" in math_toks and "0" in math_toks
    assert " ==" in math_toks and "\\frac{" in math_toks

    # 3. Arbitrary UTF-8 byte fallback (emoji + accented characters)
    utf8_sample = "Café Δx ≈ 0.001 🚀"
    assert mt.decode(mt.encode(utf8_sample)) == utf8_sample

    # 4. Factorized PerLayerEmbedding + tied lm_head on 16,384 vocab
    cfg16k = MCL4CEDConfig(
        vocab_size=16384,
        d_model=64,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ple_dim=16,
        expert_hidden_dim=128,
        num_experts=4,
        top_k=2,
        max_seq_len=64,
    )
    m16k = CognitiveOperatorEnsemble(MCL4CEDBackbone(cfg16k), COEConfig(projection_dim=16))
    assert m16k.backbone.lm_head.weight is m16k.backbone.token_embedding.weight
    assert m16k.backbone.encoder_layers[0].ple.factorized_vocab

    inp16k = torch.tensor([ids[:32]], dtype=torch.long)
    lbl16k = torch.tensor([ids[1:33]], dtype=torch.long)
    out16k = m16k.forward_inference(inp16k, targets=lbl16k)
    assert out16k.logits.shape == (1, 32, 16384)
    assert out16k.ce_loss is not None and float(out16k.ce_loss.item()) > 0.0
    assert mt.count_utf8_bytes(lbl16k) == len(mt.decode(ids[1:33]).encode("utf-8"))

    ctrl16k = MultiPassController(m16k)
    coe_gen = ctrl16k.generate_uta_with_coe(
        "What is 2 + 2?",
        max_new_tokens=16,
        candidate_specs=((1, 0.0), (2, 0.35)),
    )
    assert (
        "#U$#" in coe_gen.text
        and "#T$#" in coe_gen.text
        and "#A$#" in coe_gen.text
    )

    # 5. Transcode uint16 byte shards with 0x8000 prompt masking to MorphCode-16K shards
    src_shards = tmp_path / "byte_shards"
    dst_shards = tmp_path / "morph16k_shards"
    writer = BlockAlignedShardWriter(
        src_shards / "uta_trace",
        track="uta_trace",
        seq_len=192,
        max_blocks_per_shard=64,
        vocab_size=260,
    )
    for i in range(6):
        assert writer.add_uta_episode(
            f"Question {i}: Why do objects fall?",
            "Gravity accelerates masses toward each other.",
            "Gravitational attraction pulls objects downward.",
        )
    writer.flush_shard()
    receipts_16k = transcode_byte_shards_to_morphcode(src_shards, dst_shards, seq_len=64)
    assert "uta_trace" in receipts_16k
    ds16k = ShardedTokenBlockDataset(
        dst_shards / "uta_trace" / "index_receipt.json",
        track="uta_trace",
        split="train",
    )
    assert len(ds16k) > 0
    item0 = ds16k[0]
    assert int(item0["input_ids"].max().item()) < 16384
    assert int((item0["labels"] == -100).sum().item()) > 0


def test_context_memory_turboquant_atxy_and_codebase_warmup(tmp_path: Path) -> None:
    """Verify TurboQuant 4-bit page compression, A.TXY virtual memory, and CodebaseWarmupEngine RL warmup + exact reset."""
    from mcl4_coe import (
        CodebaseWarmupEngine,
        compress_encoded_state,
    )

    # 1. Verify TurboQuant 4-bit PolarQuant + QJL compression and EncodedState round-trip
    model = _tiny_model()
    sample_ids = torch.randint(4, 120, (1, 48), dtype=torch.long)
    with torch.no_grad():
        enc = model.backbone.encode_state(sample_ids)
    comp = compress_encoded_state(enc)
    assert comp.compression_ratio > 5.0
    recon = comp.decompress()
    cos = float(
        torch.nn.functional.cosine_similarity(
            enc.hidden.reshape(-1, 64).float(),
            recon.hidden.reshape(-1, 64).float(),
            dim=-1,
        )
        .mean()
        .item()
    )
    assert cos > 0.95

    # 2. Create a mini Python codebase in tmp_path and run CodebaseWarmupEngine
    pkg = tmp_path / "mini_pkg"
    pkg.mkdir()
    (pkg / "math_ops.py").write_text(
        '"""Arithmetic and normalization utilities."""\n\n'
        "def normalize_scores(scores: list[float], eps: float = 1e-5) -> list[float]:\n"
        '    """Normalize a list of float scores by their L2 norm."""\n'
        "    norm_sq = sum(x * x for x in scores)\n"
        "    if norm_sq <= 0.0:\n"
        "        return [0.0 for _ in scores]\n"
        "    scale = (norm_sq + eps) ** 0.5\n"
        "    return [x / scale for x in scores]\n\n"
        "def discounted_return(rewards: list[float], gamma: float = 0.99) -> float:\n"
        '    """Compute discounted cumulative return across trajectory rewards."""\n'
        "    total = 0.0\n"
        "    for r in reversed(rewards):\n"
        "        if r != 0.0:\n"
        "            total = r + gamma * total\n"
        "    return total\n",
        encoding="utf-8",
    )
    (pkg / "router.py").write_text(
        '"""Query router and page cache."""\n\n'
        "from .math_ops import normalize_scores, discounted_return\n\n"
        "def select_best_page(raw_scores: list[float], min_prob: float = 0.1) -> int:\n"
        '    """Return index of highest normalized page score above threshold."""\n'
        "    probs = normalize_scores(raw_scores)\n"
        "    best_idx = max(range(len(probs)), key=lambda i: probs[i])\n"
        "    if probs[best_idx] < min_prob:\n"
        "        return 0\n"
        "    return best_idx\n\n"
        "def score_trajectory_pages(rewards: list[float], bonus: float = 1.0) -> float:\n"
        '    """Score trajectory pages with discounted cumulative return plus bonus."""\n'
        "    base_ret = discounted_return(rewards)\n"
        "    if base_ret > 0.0:\n"
        "        return base_ret + bonus\n"
        "    return base_ret\n",
        encoding="utf-8",
    )

    engine = CodebaseWarmupEngine(
        subsystems=("mini_pkg",),
        max_seq_len=64,
        num_route_options=4,
    )
    session = engine.index_repository(model, tmp_path, max_tasks_per_family=8)
    assert session.total_repo_files == 2
    assert session.total_repo_symbols >= 4
    assert session.compression_ratio > 5.0
    assert len(session.train_route_tasks) >= 4
    assert len(session.train_verify_tasks) >= 2

    report = engine.warmup_session(model, session, steps=4, lr=2e-3, batch_size=2)
    assert report.warmup_steps == 4
    assert report.step_losses[-1] <= report.step_losses[0]
    assert len(report.step_losses) == 4

    q_res = engine.query_codebase(
        model,
        session,
        "Where is normalize_scores defined to normalize float scores by L2 norm?",
        top_k_candidates=4,
    )
    assert "symbol" in q_res and "address" in q_res

    # 3. Verify Tier-3 ContextLMCompressor multi-file synthesis & end-to-end codebase-grounded generation
    mfile = engine.synthesize_multifile_context(
        model,
        session,
        "How does select_best_page call normalize_scores across modules?",
        max_files=2,
        top_k_candidates=4,
    )
    assert mfile.num_files_folded == 2
    assert mfile.compressed_gist_tokens == 12
    assert mfile.compressed_state.hidden.shape == (1, 12, 64)
    assert mfile.multifile_latent_prior.shape == (1, 64)

    ctrl = MultiPassController(model)
    rag_out = ctrl.generate_uta_with_codebase_session(
        "How does select_best_page call normalize_scores?",
        engine,
        session,
        max_files=2,
        top_k_candidates=4,
        max_new_tokens=20,
    )
    assert "#T$#" in rag_out.text and "#A$#" in rag_out.text
    assert rag_out.num_files_folded == 2
    assert rag_out.extra_prompt_tokens == 0
    assert len(rag_out.candidates) == 9

    # 4. Verify Incremental Dirty-Page Refresh (modifying an existing file + adding a new symbol + WAL framing)
    assert engine.detect_dirty_files(session) == []
    (pkg / "router.py").write_text(
        'from .math_ops import normalize_scores\n\n'
        'def select_best_page(scores: list[float]) -> int:\n'
        '    """Route query to the highest normalized page index with tie-breaking."""\n'
        '    normed = normalize_scores(scores)\n'
        '    return int(normed.index(max(normed)))\n\n'
        'def evict_stale_page(ages: list[int], max_age: int = 64) -> list[int]:\n'
        '    """Evict cached TurboQuant pages older than max_age ticks."""\n'
        '    return [idx for idx, a in enumerate(ages) if a > max_age]\n',
        encoding="utf-8",
    )
    detected_dirty = engine.detect_dirty_files(session)
    assert detected_dirty == ["mini_pkg/router.py"]

    wal_file = tmp_path / "session_dirty.wal"
    refresh_rep = engine.refresh_dirty_files(
        model,
        session,
        micro_warmup_steps=2,
        wal_path=wal_file,
    )
    assert refresh_rep.dirty_files == ("mini_pkg/router.py",)
    assert "mini_pkg/router.py:evict_stale_page" in refresh_rep.added_symbols
    assert "mini_pkg/router.py:select_best_page" in refresh_rep.modified_symbols
    assert refresh_rep.recompressed_pages == 1
    assert refresh_rep.wal_frames_appended >= 3
    assert wal_file.is_file() and wal_file.stat().st_size > 0
    assert engine.detect_dirty_files(session) == []
    assert session.store.resolve("mini_pkg/router.py:evict_stale_page") is not None

    # 5. Verify exact <1ms restoration of base weights (0.0 parameter drift)
    session.restore_base_weights(model)
    for name, param in model.named_parameters():
        if name in session.base_operator_state:
            drift = float((param.detach().cpu() - session.base_operator_state[name]).abs().max().item())
            assert drift == 0.0


def test_recurrent_depth_trajectory_pondering_and_adaptive_halting() -> None:
    """Verify Recurrent Depth trajectory pondering (`step_states`), `recurrent_kv`, `margin_halt_threshold`, 0-ULP candidate permutation invariance, and adaptive entropy halting."""
    model = _tiny_model().eval()
    x = torch.randint(4, 120, (2, 24), dtype=torch.long)

    # 1. Verify step_states trajectory length and entropy-gated residual recurrent depth + recurrent_kv
    out_r1 = model.forward_inference(x, recurrence_steps=1)
    assert len(out_r1.step_states) == 1
    out_r3 = model.forward_inference(
        x,
        recurrence_steps=3,
        recurrent_kv=True,
        depth_ponder=True,
    )
    assert len(out_r3.step_states) == 3
    assert out_r3.logits.shape == out_r1.logits.shape
    assert not torch.allclose(out_r3.logits, out_r1.logits)

    # 2. Verify 0-ULP bitwise candidate permutation invariance under depth_ponder and margin_halt_threshold
    ctx_ids = torch.randint(4, 120, (2, 20), dtype=torch.long)
    ctx_lens = torch.tensor([20, 16], dtype=torch.long)
    opt_ids = torch.randint(4, 120, (2, 4, 18), dtype=torch.long)
    opt_lens = torch.tensor([[18, 15, 12, 14], [14, 18, 11, 16]], dtype=torch.long)
    targets = torch.tensor([1, 2], dtype=torch.long)

    perm_idx = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    inv_idx = torch.argsort(perm_idx)
    opt_ids_perm = opt_ids[:, perm_idx, :]
    opt_lens_perm = opt_lens[:, perm_idx]

    dec_ponder = model.forward_decision(
        ctx_ids,
        ctx_lens,
        opt_ids,
        opt_lens,
        recurrence_steps=2,
        depth_ponder=True,
        margin_halt_threshold=0.35,
    )
    dec_ponder_perm = model.forward_decision(
        ctx_ids,
        ctx_lens,
        opt_ids_perm,
        opt_lens_perm,
        recurrence_steps=2,
        depth_ponder=True,
        margin_halt_threshold=0.35,
    )
    assert torch.equal(dec_ponder.similarities, dec_ponder_perm.similarities[:, inv_idx])
    assert torch.equal(dec_ponder.logits, dec_ponder_perm.logits[:, inv_idx])
    assert torch.allclose(
        dec_ponder.probabilities, dec_ponder_perm.probabilities[:, inv_idx], atol=1e-7
    )

    all_op_ponder = model.forward_all_operators(
        ctx_ids,
        ctx_lens,
        opt_ids,
        opt_lens,
        targets,
        recurrence_steps=2,
        depth_ponder=True,
        margin_halt_threshold=0.35,
    )
    assert torch.isfinite(all_op_ponder.total_loss)

    # 3. Verify adaptive_recurrent_depth in MultiPassController.generate_uta
    ctrl = MultiPassController(model)
    trace_adapt = ctrl.generate_uta(
        "Explain recurrent depth pondering.",
        max_new_tokens=20,
        adaptive_recurrent_depth=True,
        max_recurrence_steps=3,
        recurrent_entropy_threshold=1.5,
        recurrent_kv=True,
    )
    assert "#U$#" in trace_adapt and "#T$#" in trace_adapt and "#A$#" in trace_adapt


def test_pre_scaling_housekeeping_gist_conditioning_and_repo_uta_traces(tmp_path: Path) -> None:
    from mcl4_coe import TrainingRunConfig, synthesize_repository_uta_traces

    torch.manual_seed(1234)
    # 1. Verify projection_dim auto-scaling in TrainingRunConfig
    assert TrainingRunConfig(d_model=128).projection_dim == 32
    assert TrainingRunConfig(d_model=384).projection_dim == 96
    assert TrainingRunConfig(d_model=768).projection_dim == 192
    assert TrainingRunConfig(d_model=768, projection_dim=128).projection_dim == 128

    # 2. Verify end-to-end gradient flow into gist_queries, gist_norm, and abstraction_proj
    #    plus deep recurrent trajectory supervision
    b_cfg = MCL4CEDConfig(
        vocab_size=260,
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        ple_dim=16,
        expert_hidden_dim=128,
        num_experts=4,
        top_k=2,
        max_seq_len=128,
        dropout=0.0,
    )
    c_cfg = COEConfig(projection_dim=16, gist_tokens_per_chunk=4, gist_chunk_size=16)
    coe = CognitiveOperatorEnsemble(MCL4CEDBackbone(b_cfg), c_cfg)

    # Verify legacy state_dict without gist_queries / gist_norm loads cleanly under strict=True
    legacy_sd = {
        k: v for k, v in coe.state_dict().items() if not (k == "gist_queries" or k.startswith("gist_norm."))
    }
    coe.load_state_dict(legacy_sd, strict=True)

    coe.train()
    coe.zero_grad(set_to_none=True)
    x = torch.randint(4, 240, (2, 48), dtype=torch.long)
    loss_dict = compute_joint_coe_loss(
        coe,
        lm_batch=(x, x.clone()),
        recurrence_steps=2,
        recurrent_kv=True,
        depth_ponder=True,
        deep_supervision_weight=0.25,
        condition_on_prefix_gist=True,
        latent_prior_scale=0.35,
    )
    total_loss = loss_dict.total_loss
    assert torch.isfinite(total_loss)
    total_loss.backward()

    assert coe.gist_queries.grad is not None
    assert float(coe.gist_queries.grad.abs().sum().item()) > 0.0
    assert coe.gist_norm.weight.grad is not None
    assert float(coe.gist_norm.weight.grad.abs().sum().item()) > 0.0
    assert coe.abstraction_proj.weight.grad is not None
    assert float(coe.abstraction_proj.weight.grad.abs().sum().item()) > 0.0

    # 3. Verify strict prefix causality: mutating token (prefix_len - 1) must NOT leak backward
    #    into prefix positions 0 .. prefix_len - 2 even when condition_on_prefix_gist=True
    coe.eval()
    prefix_len = max(8, min(48 // 3, 48 - 8))  # 16
    x_a = x[:1].clone()
    x_b = x[:1].clone()
    x_b[0, prefix_len - 1] = (x_b[0, prefix_len - 1] + 17) % 200 + 10
    with torch.no_grad():
        out_a = coe.forward_inference(
            x_a,
            recurrence_steps=2,
            recurrent_kv=True,
            depth_ponder=True,
            condition_on_prefix_gist=True,
        )
        out_b = coe.forward_inference(
            x_b,
            recurrence_steps=2,
            recurrent_kv=True,
            depth_ponder=True,
            condition_on_prefix_gist=True,
        )
    # Prefix tokens 0 .. prefix_len - 2 must have 0-diff logits (zero lookahead leakage)
    max_prefix_leak = float((out_a.logits[:, : prefix_len - 1] - out_b.logits[:, : prefix_len - 1]).abs().max().item())
    assert max_prefix_leak < 1e-5, f"Lookahead leakage detected inside prefix: {max_prefix_leak}"
    # Continuation tokens >= prefix_len must differ because gist_state and latent_prior encode prefix_len - 1
    cont_diff = float((out_a.logits[:, prefix_len:] - out_b.logits[:, prefix_len:]).abs().max().item())
    assert cont_diff > 1e-4

    # 4. Verify AST-grounded repository UTA trace synthesis
    sample_repo = tmp_path / "repo"
    sample_repo.mkdir(parents=True, exist_ok=True)
    (sample_repo / "module_a.py").write_text(
        'class SampleRouter:\n'
        '    """Routes tokens across local context pages."""\n'
        '    def route_pages(self, query_vec: list[float], top_k: int = 2) -> list[int]:\n'
        '        """Return top-k page indices."""\n'
        '        if top_k <= 0:\n'
        '            return []\n'
        '        return list(range(min(len(query_vec), top_k)))\n',
        encoding="utf-8",
    )
    traces = synthesize_repository_uta_traces(sample_repo, max_episodes=16)
    assert len(traces) >= 2
    assert any("SampleRouter" in u or "route_pages" in u for u, _, _ in traces)

    # 5. Verify `codeparrot/github-code` streaming normalization, path sieve, and pretrain/UTA extraction
    from mcl4_coe import (
        extract_pretrain_text,
        extract_uta_episodes,
        infer_github_code_language,
        normalize_github_code_record,
    )

    assert infer_github_code_language("app/mods/mod190.js") == "JavaScript"
    assert infer_github_code_language("src/mcl4_coe/backbone.py") == "Python"
    raw_parquet_row = {
        "content": (
            "import mod189 from './mod189';\n"
            "export function computeNextValue(seed) {\n"
            "  const value = mod189 + seed + 1;\n"
            "  if (value <= 0) {\n"
            "    return 0;\n"
            "  }\n"
            "  return value;\n"
            "}\n"
            "export default computeNextValue;\n"
        ),
        "repo_name": "MirekSz/webpack-es6-ts",
        "path": "app/mods/mod190.js",
        "license": "isc",
        "size": "214",
    }
    norm_row = normalize_github_code_record(raw_parquet_row)
    assert norm_row["language"] == "JavaScript"
    assert norm_row["size"] == 214
    pretrain_txt, rej = extract_pretrain_text(norm_row)
    assert not rej and pretrain_txt is not None
    assert "# Repo: MirekSz/webpack-es6-ts | Path: app/mods/mod190.js | Language: JavaScript" in pretrain_txt
    # Ensure code newlines are never collapsed by prose unwrapping
    assert "import mod189 from './mod189';\nexport function computeNextValue" in pretrain_txt

    uta_eps = extract_uta_episodes(norm_row, max_seq_len=2048)
    assert len(uta_eps) == 1
    assert "MirekSz/webpack-es6-ts" in uta_eps[0][0]
    assert "computeNextValue" in uta_eps[0][2]

    # Vendored node_modules path should be rejected at Stage 0
    noisy_row = dict(norm_row, path="wp-content/node_modules/es6-set/polyfill.js")
    _, noisy_rej = extract_pretrain_text(noisy_row)
    assert noisy_rej is True
    assert extract_uta_episodes(noisy_row, max_seq_len=2048) == []

    # 6. Verify RoPE n_prefix alignment in project_global_kv and <eos> boundary isolation in packed blocks
    from mcl4_coe.pipeline import _is_clean_uta_episode

    dec_layer = coe.backbone.decoder_layers[0]
    dummy_seq = torch.randn(2, 16, b_cfg.d_model)
    dummy_prefix = torch.randn(2, 6, b_cfg.d_model)
    k_seq, _ = dec_layer.project_global_kv(dummy_seq, n_prefix=0)
    k_prefixed, _ = dec_layer.project_global_kv(
        torch.cat([dummy_prefix, dummy_seq], dim=1), n_prefix=6
    )
    assert torch.allclose(k_seq, k_prefixed[:, :, 6:, :], atol=1e-6)
    assert _is_clean_uta_episode("Explain X.", "Step 1: check X.", "X is valid.") is True
    assert _is_clean_uta_episode("Score Candidate 1 now.", "Evaluate step-by-step: [0]", "[0]") is False
    assert _is_clean_uta_episode("{'messages': []}", "TRUE", "{'messages': []}") is False






