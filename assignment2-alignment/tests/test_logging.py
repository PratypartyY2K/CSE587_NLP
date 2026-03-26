from __future__ import annotations

import json

import pytest

from alignment import log_generations


def test_log_generations_writes_jsonl(tmp_path):
    output_path = tmp_path / "generations.jsonl"

    rows = log_generations(
        prompts=["p1", "p2"],
        generations=["g1", "g2"],
        ground_truths=["r1", "r2"],
        reward_info=[
            {"format_reward": 1.0, "answer_reward": 1.0, "reward": 2.0},
            {"format_reward": 1.0, "answer_reward": 0.0, "reward": 1.0},
        ],
        avg_token_entropies=[0.2, 0.4],
        response_lengths=[3, 5],
        metadata=[{"split": "train"}, {"split": "eval"}],
        output_path=output_path,
    )

    assert rows == [
        {
            "index": 0,
            "prompt": "p1",
            "generation": "g1",
            "ground_truth": "r1",
            "reward_info": {"format_reward": 1.0, "answer_reward": 1.0, "reward": 2.0},
            "avg_token_entropy": 0.2,
            "response_length": 3,
            "is_correct": True,
            "metadata": {"split": "train"},
        },
        {
            "index": 1,
            "prompt": "p2",
            "generation": "g2",
            "ground_truth": "r2",
            "reward_info": {"format_reward": 1.0, "answer_reward": 0.0, "reward": 1.0},
            "avg_token_entropy": 0.4,
            "response_length": 5,
            "is_correct": False,
            "metadata": {"split": "eval"},
        },
    ]
    assert rows.summary == {
        "num_examples": 2,
        "avg_response_length": 4.0,
        "avg_response_length_correct": 3.0,
        "avg_response_length_incorrect": 5.0,
    }

    written_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert written_rows == rows
    assert json.loads(output_path.with_name("generations_summary.json").read_text(encoding="utf-8")) == rows.summary


def test_log_generations_rejects_length_mismatches():
    with pytest.raises(ValueError, match="same length"):
        log_generations(prompts=["p1"], generations=["g1", "g2"])

    with pytest.raises(ValueError, match="ground_truths must have length 1, got 2"):
        log_generations(prompts=["p1"], generations=["g1"], ground_truths=["r1", "r2"])

    with pytest.raises(ValueError, match="pass only one of ground_truths or references"):
        log_generations(
            prompts=["p1"],
            generations=["g1"],
            ground_truths=["r1"],
            references=["r1"],
        )


def test_log_generations_respects_max_rows():
    rows = log_generations(
        prompts=["p1", "p2", "p3"],
        generations=["g1", "g2", "g3"],
        reward_info=[
            {"answer_reward": 1.0, "reward": 1.0},
            {"answer_reward": 0.0, "reward": 0.0},
            {"answer_reward": 1.0, "reward": 1.0},
        ],
        response_lengths=[1, 2, 3],
        max_rows=2,
    )

    assert rows == [
        {
            "index": 0,
            "prompt": "p1",
            "generation": "g1",
            "reward_info": {"answer_reward": 1.0, "reward": 1.0},
            "response_length": 1,
            "is_correct": True,
        },
        {
            "index": 1,
            "prompt": "p2",
            "generation": "g2",
            "reward_info": {"answer_reward": 0.0, "reward": 0.0},
            "response_length": 2,
            "is_correct": False,
        },
    ]
    assert rows.summary == {
        "num_examples": 2,
        "avg_response_length": 1.5,
        "avg_response_length_correct": 1.0,
        "avg_response_length_incorrect": 2.0,
    }
