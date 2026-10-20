---
title: ICA Model 1 evaluation viewer
emoji: 📊
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
---

# Model 1 (objection trigger): evaluation viewer

Interactive view of the Model 1 evaluation on the public
[Sakda101/explorer-customer-objection-data](https://huggingface.co/datasets/Sakda101/explorer-customer-objection-data)
dataset, turned into synthetic multi-turn calls. Development data only: the fine-tuned encoders are near ceiling here, so
the views exercise the method rather than make the real choices.

Tabs: context length × objection type (with the chosen context length N*), metrics and checks with
Good / Bad / Inconclusive verdicts, business view, keyword cues, early warning, definitions.
