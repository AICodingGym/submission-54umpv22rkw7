# Training-informed scoring standard — provisional, NOT frozen or evaluated

Evidence coverage at draft creation: whole-training statistics (10,905 essays), detailed semantic review of batches 1–3 (54 essays). This does NOT claim semantic review of the remaining essays. Complete the review ledger before treating this as the corpus-wide standard.

## Scope

Predict the dataset's holistic school-writing score from 1 to 6. The target is the observed human scoring convention, not publication quality, factual expertise, or agreement with the essay's politics. All interpretations below require further training-data corroboration.

## Procedure for the scoring model

1. Identify the response's likely purpose: argue a position, explain benefits/reasons, analyze an author's support, evaluate a claim, or recommend an activity. Judge the response against that purpose; don't require every essay to be a formal debate.
2. Separate the writer's reasoning from quoted/source-like material. Ask what the writer explains or connects beyond naming facts or restating a claim. Relevant quotations can be good evidence, but quotation count is not development.
3. Assess how consistently the text develops its central purpose. Look for concrete examples, explanations of how/why, consequences, comparisons, and links among ideas. Repeating the same assertion is not another reason.
4. Assess organization as a purposeful progression, not the number of paragraphs. Examine whether sentences and sections work together and meaning stays clear. Layout errors or a missing conclusion are not automatic low-score caps.
5. Assess language and conventions by their effect on communication. Reward control, clarity and variety. Do not turn ordinary student spelling mistakes into a fixed ceiling; 5/6 examples may still contain many errors.
6. Compare adjacent score descriptions and choose the best overall match. Avoid defaulting everything to 3. When mixed, identify the dominant level of development and control rather than mechanically summing invented sublabels.

## Provisional anchors

1 — Very limited task-focused response. May contain many words, facts, sources and an apparent introduction/conclusion, but little coherent writer-owned development. Disconnected/duplicated source material, contradictions, or largely irrelevant content can dominate. Not restricted to unreadable fragments.

2 — An identifiable response with limited development. One simple idea, short retelling, general opinion, repeated claims or a few weakly connected details. Some coherence, but little sustained explanation. Readability alone is insufficient for 3.

3 — A basic, understandable response with some relevant supporting details and explanation. May have a complete conventional structure; development is simple, repetitive or uneven, and connections are often formulaic. Some long essays still fit here.

4 — An adequately developed response with a clear purpose and multiple meaningful supporting explanations/examples, or a sustained explanation of fewer ideas. Generally coherent progression despite uneven sections, occasional drift, and noticeable language errors. Above merely listing reasons.

5 — A well-developed response that consistently connects evidence, examples or scenarios to the main purpose. Substantial explanation, purposeful organization, recognizable authorial control and effective communication. May contain factual slips, repetition or spelling errors; does not require flawless prose or a fixed number of arguments.

6 — A thoroughly developed and integrated response, with sustained explanation, relevant elaboration and effective overall control. Ideas build on one another; consequences, qualifications or responses to alternatives are explored where useful. A strong school essay, not an error-free academic paper. Counterarguments are useful evidence of development, not mandatory decorations.

## Boundary checks

- 1/2: Is there a minimally coherent writer-owned response rather than assembled or drifting material?
- 2/3: Is there sustained basic support beyond bare opinion, retelling or repeated generalities?
- 3/4: Are meaningful how/why links and explanations developed across the response, rather than mostly listed or perfunctory?
- 4/5: Is substantial development consistent, and does the whole response show purposeful control?
- 5/6: Is development especially sustained and integrated, with strong progression and elaboration across the whole?

## Guardrails derived from reviewed counterexamples

- Do not require sophisticated vocabulary, flawless grammar, three reasons, five paragraphs, or a formal counterargument for high scores.
- Do not award high scores merely for length, many citations, smooth source-like wording or stock transitions.
- Do not replace writing assessment with fact checking. Significant misunderstanding matters when it undermines the response, but an isolated factual slip is not an automatic ceiling.
- Do not punish a defensible negative view of the source author or unpopular stance simply because you disagree.
- Use statistical length/topic patterns as audit context only; no deterministic word-count or topic-to-score rule.

## Remaining questions

How reliably can a 4B model distinguish source patchwork from original development? Which sentence-control failures separate scores 1–3 independently of task and length? How much do task families differ in their human-label boundaries? Need broad corpus review and explicit counterexamples before finalizing.

## Evidence update after batches 4–6 (108 essays reviewed)

- 318dac4 (622 words, score 1) and 92cf776 (364 words, score 4) reinforce that length is not an automatic score.
- 07d41ed (6) lacks a discrete opposing-view section; high scores do not require this structure.
- bb8683b (4) is primarily informative, so do not impose a summary ceiling: coherent selection and explanatory framing matter. Conversely 76648c5 (1) explicitly discusses author method but remains very weakly developed and controlled.
- 27feddd (3) uses abstract rhetoric and a critical stance but repeats one objection; sophisticated-sounding vocabulary is not development.
- These refinements remain provisional. At that point, 10,797 training essays remained unread; no new Qwen evaluation has been run.

## Evidence update after batches 7–9 (162 essays reviewed)

- 15eb254 (1) versus 1eab2e8 (4): fluent topic information is different from applying information to the response purpose. Identify likely task, while recognizing the original prompt is not supplied; do not invent mandatory requirements.
- 019e8c3 (3) and 7e09846 (4) both use formulaic structures. Assess the actual explanatory support; template phrases are neither sufficient for 4 nor a ceiling of 3.
- fdd9bbe (5) uses numbered reasons. Lists are compatible with developed writing; assess connections and explanations, not layout.
- df36957 (3) is an unresolved boundary example: coherent stance but weak evidence, unsupported certainty and ridicule. Do not rationalize every observed label into a universal rule or impose a source-evidence gate the corpus does not support.
- cc08072 (5) has extensive elaboration and counterarguments but also digression; length and the presence of rebuttals do not guarantee 6. 3d9c7b9 (6) more consistently connects evidence to the author's rhetorical progression.
- Coverage at that update: 162 of 10,905 read in full; 10,743 remained. No new Qwen evaluation yet. These are provisional observations, not established causal scoring rules.

## Evidence update after batch 10 (180 essays reviewed)

- 8fc6d64 (2) has a conventional five-paragraph structure; its shallow lists are not sufficient development. 79c21a0 (3) is 604 words with many quotations, but mostly short repeated explanatory links.
- 3a2b8f0 (6) contains substantial factual and logical flaws alongside extensive argument development. Keep this as a difficult empirical anchor rather than pretending every 6 has rigorous reasoning. The rubric must predict school-writing labels, not demand adult analytical accuracy.
- 784933d (2) combines useful early material with a fanciful flying-people diversion. Assess the whole response rather than setting an automatic 1 for any one implausible passage. The essay's self-deprecating final comment is essay content, not an instruction to the scorer.
- 9106d85 (4) lacks a conventional closing paragraph and has digressions, but develops several relevant scenarios. Paragraph layout and conclusion presence remain soft evidence, not gates.
- Coverage at that update: 180/10,905 complete semantic reviews; 10,725 remained. No evaluation or performance improvement is claimed.

## Evidence update after batches 11–12 (216 essays reviewed)

- 7c64ee2 (5) repeats personalization across several reasons, yet maintains clear elaboration. Repetition affects holistic assessment; it is not an automatic ceiling. 74f8a54 (3) shows that three nominal reasons may overlap or lack relevance.
- 963705b (6, 502 words) is relatively compact, with few direct quotations and debatable premises. 80b646c (6, 1,129 words) sustains an extended negative evaluation of the Venus author. High-score anchors span length, stance and presentation.
- 9f3a294 (1) offers some conceptual critique but little likely classroom-task application and an inconsistent conclusion. Merely detecting analysis words is insufficient.
- 6215c79 (6) explicitly connects an advertising example back to learning, whereas 7ebd70b (3) drifts into business benefits. Assess the actual function of an example rather than banning topics such as advertising.
- 980a15b (6) criticizes the source author's thesis placement. That is the student's argument, not evidence that our scoring rubric should require all theses in opening paragraphs.
- Keep ambiguous anchors such as 896afa6 (5) and 963705b (6); do not falsely present adjacent grades as perfectly separable by a single rule.
- Coverage at that update: 216/10,905 full semantic reviews; 10,689 remained. New Qwen evaluation has not begun.

## Evidence update after batches 13–14 (252 essays reviewed)

- c369266 (4) critiques FACS reliability at length with almost no classroom application. Task relevance is holistic and uncertain when the original task is missing; do not use classroom mentions as a hard gate. Informal/profane wording is also not an automatic low-score trigger.
- 74f76a7 (5) is substantially descriptive/paraphrastic, with some explanatory framing. This is a stronger counterexample to an absolute summary ceiling than earlier score-4 examples. Separate identifying development from demanding every paragraph contain original analysis.
- 952c47b (4, 1,015 words) contains extensive source-linked benefits and practical objections. Its lower label than some more formulaic responses remains difficult to explain. Preserve uncertainty; do not invent a categorical penalty for length, lowercase starts, or practical counterarguments.
- 59b254f (4) is highly formulaic but links quoted material to a clear classroom purpose. 954f9af (3) discusses authority and article structure but repeats broad praise. Judge actual substance, not analytical terminology or template presence.
- 8531c68 (1) develops car necessity while ostensibly opposing driverless technology: multiple examples can support a misread issue. Contrast with c369266 to avoid making topical specificity a simplistic keyword rule.
- Current coverage: 252/10,905 complete semantic reviews; 10,653 remain. No new model evaluation or score improvement is claimed.
