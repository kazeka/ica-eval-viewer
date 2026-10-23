---
title: ICA evaluation viewer
emoji: 📊
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
---

# ICA evaluation viewer

Interactive view of the ICA Phase 1 evaluation framework on development data only:

- **Model 1 (objection trigger)** on the public
  [Sakda101/explorer-customer-objection-data](https://huggingface.co/datasets/Sakda101/explorer-customer-objection-data)
  dataset, turned into synthetic multi-turn calls. The fine-tuned encoders are near ceiling here, so the views exercise
  the method rather than make the real choices.
- **Model 2 (reply recommender)** and the **T1–T8 scoring checks** on the team's synthetic sandbox (simulated calls,
  replies and SMEs).

Pages: overview; Model 1 (context length × objection type, metrics and checks with Good / Bad / Inconclusive verdicts,
business view, keyword cues, early warning); Model 2 (LLM-free metrics, rule checks and failures by objection type, with
replies to read); scoring checks T1–T8.
