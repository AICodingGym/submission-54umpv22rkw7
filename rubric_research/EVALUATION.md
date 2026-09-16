# Teacher rubric experiments

User updated stopping rule: evaluate when successive review batches stop introducing new scoring dimensions. Batches 14–19 primarily supplied counterexamples/refinements to existing dimensions. First evaluation starts with **342/10,905 essays actually reviewed**, not a claim of complete semantic review.

## Fixed design

- Main assistant writes/refines prompts. Qwen3-4B only scores; no Qwen reflection and no weight training.
- Same local model revision and decoding as prior Qwen GEPA: BF16, nonthinking, batch 2, 2,048 essay tokens, 4,096 total context, constrained digit logits.
- Report argmax and conditional expected score with fixed half-integer thresholds separately. No old calibration thresholds.
- Full selection set (1,557 rows) provides development metrics. Final holdout remains sealed.
- Reviewed training essays provide diagnostic errors; this set is deliberately score-balanced and is not representative of the whole training population.
- Exclude prompt demonstrations from diagnostic metrics. Full selection contains no demonstrations.
- Save every prompt, input IDs, probabilities, truncation flags, configuration and result. Repeated selection comparisons can overfit; no final-generalization claims from these scores.

## v0001

Teacher's rubric distilled from review, no demonstrations. Started with `teacher_eval.py --prompt rubric_research/rubrics/v0001_prompt.txt --output outputs_teacher_v0001`.

Training diagnostic: QWK 0.4310, MAE 1.1901 over 342 essays. No predictions of 1 or 6; 1 prediction of 2, 71 of 3, 149 of 4 and 121 of 5. e03613b and 8cad5af (both human 1) predicted 5. This suggests severe scale compression and overrating source-like material. It does not establish selection performance.

## v0002 — prepared after training diagnostic, before v0001 selection result

Shorter rubric plus four already reviewed train examples at scores 1, 2, 4, 6. The goal is to demonstrate the actual school-writing scale and reduce compression. Examples: aa52b3b, 610d03b, 7cc82fb, f260d4e. These four must be excluded from train diagnostic metrics; compare versions on the same remaining 338 IDs. Example choice is manual, not optimized using selection errors. Prompt is 1,572 tokenizer tokens before guard/chat overhead and retains the same essay budget.

Run with `teacher_eval.py --prompt rubric_research/rubrics/v0002_prompt.txt --examples rubric_research/rubrics/v0002_examples.json --output outputs_teacher_v0002` after the first GPU run ends.
