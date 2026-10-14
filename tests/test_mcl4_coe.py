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
    """Verify CPUOffloadAdamW updates working weights and preserves FP32 masters on CPU."""
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

    # Verify verbose <think> monologue compaction fits inside seq_len=2048
    verbose_think = "<think>" + ("Step derivation: let x_i = i^2 + 1.\n" * 80) + "</think>\nFinal answer: 42"
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
    assert len(u) + len(t) + len(a) <= 1950
    assert "..." in t


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


