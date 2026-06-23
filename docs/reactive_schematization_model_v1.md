# The Reactive Schematization Model (RSM)

**A formal framework for language acquisition as response-confirmed pattern accumulation**

**Authors:** [AUTHORS REDACTED FOR ANONYMOUS REVIEW]
**Status:** Foundational draft v1.0
**Date:** 2026-06-13
**Project:** IMT Attention Bias paper (companion theoretical document)
**Origin:** Observation by the first author from an L2 traveler experience, generalized into a developmental model.

---

## Preface

This document formalizes a model of first-language acquisition derived from a single observation made by the first author during international travel without prior L2 instruction: practical communicative competence is achievable through isolated content words, prosodic contour, gesture, and frozen formulae, with no productive grammar. Reflection on this observation against the empirical results of the present typological corpus study ([CITATION REDACTED FOR ANONYMOUS REVIEW], IMT Attention Bias paper) suggested that the standard model of L1 acquisition has misidentified what the child is doing.

The conventional model treats the infant as a grammar acquirer who uses attention to filter input toward grammatical structure. The model proposed here treats the infant as a communicator who uses attention to move the caregiver, and treats grammar as the residual sediment of communicative success.

This is not a usage-based-model variant. It is not a UG-with-attention model. It is a different ontological commitment about what the learner is doing and why structure emerges.

---

## §1. Three Claims

The model rests on three claims, ordered from most empirical to most philosophical.

### Claim 1 (Empirical): The infant is not acquiring grammar.

The infant in the prelinguistic and early-linguistic period is engaged in **communicative success-seeking**: deploying available signaling resources (cry, gaze, gesture, isolated words, prosodic contour) to move the caregiver toward states the infant desires (food, comfort, play, removal of discomfort, joint attention).

What the infant is monitoring is not the structure of the input. It is **the responsiveness of the caregiver to its own outputs**.

The grammar of the ambient language is incidentally present in the input the child receives, but it is not the target of acquisition. The target is **response**.

### Claim 2 (Mechanical): Response confirms pattern, and confirmed patterns accumulate.

When an infant produces a signaling event — a sound, a gesture, a word, a word combination, an intonation contour — and the caregiver responds in a way that satisfies the infant's communicative goal, that signaling event becomes a confirmed pattern.

Confirmed patterns are stored. We call the store the **Book**. The Book is a memory structure of all the signaling events that have been response-confirmed in the infant's history.

The Book is not initially organized by grammatical category. It is organized by **what worked**: this sound got food, this gesture got pickup, this two-word combination got toy, this sentence-final intonation got affirmation.

### Claim 3 (Philosophical): Grammar is the schema that emerges from the Book.

As the Book accumulates response-confirmed patterns, regularities across patterns become extractable. These regularities are what we call **schemas**.

Schemas are not learned as rules. They are statistical residues of repeated successes. When the child has been understood thousands of times after producing utterance-final rising intonation in the context of a request, "rising intonation = request" becomes a schema. When the same child has been understood thousands of times after placing an agent word before a verb word, "agent-before-verb" becomes a schema.

What linguists call grammar is, on this view, the sedimented residue of the Book. It is not present in the child's mind as rules. It is the post-hoc description of the schemas that the Book has accumulated.

---

## §2. Formal Architecture

### §2.1 Variables

We define five variables that together specify the model.

**I (Input)**
The total stream of caregiver speech, gesture, and contextual stimulation directed to the infant. I is the raw signal before any filtering.

**A (Attention)**
A function over I that selects portions of I as candidates for processing. A operates over the five salience dimensions defined in our typological study: acoustic edge (S_acoustic), positional edge (S_positional), frequency (S_frequency), repetition (S_repetition), and perceptual distinctness (S_perceptual). The output of A is the set of input fragments that the infant attends to.

**O (Output)**
The signaling events the infant produces — cries, gazes, gestures, sounds, words, combinations, intonations. Early O is heavily prelinguistic; late O is increasingly linguistic.

**R (Response)**
The caregiver's reaction to each O. R is the variable the infant is actually monitoring. R can satisfy (S+) or fail to satisfy (S−) the infant's communicative intent.

**B (Book)**
The store of (O, R) pairs where R was S+. The Book is the infant's accumulated record of which signaling events have produced satisfying responses.

### §2.2 The acquisition pipeline

The standard model is:

> I → A → grammar acquisition → O

The Reactive Schematization Model is:

> I → A → O → R → B → schema → next-generation O

The reversal of the causal arrow at the central step is the substantive difference. In the standard model, attention feeds grammar, and grammar produces output. In RSM, attention feeds output (which is communicatively motivated), and output is updated by response feedback, with grammar emerging as a derivative summary of the Book.

### §2.3 What gets into the Book

A signaling event O is admitted to the Book when:

(a) O is attentionally available — the infant noticed both its own production and the caregiver's response. This is the role of A: attention is what makes the (O, R) pairing perceivable as a unit.

(b) R is satisfying — the caregiver's response moved the world toward the state the infant intended, or close enough.

(c) The pairing is salient enough to enter memory — frequency of co-occurrence, contextual distinctiveness, and emotional valence all contribute.

Note what is absent from this list: **grammatical correctness**. The Book does not check whether O conformed to ambient grammar. It checks only whether R satisfied. A non-grammatical O that produced an S+ response is admitted. A grammatical O that failed to produce S+ is not.

### §2.4 Schema extraction from the Book

Once the Book contains enough (O, R) pairs, regularities can be extracted. We define schema extraction as a clustering operation over the Book: signaling events that share structural properties and that have similar response profiles cluster together.

The clusters are schemas. They have approximately the form:

> Schema = {feature set F, response prediction R̂, confidence c}

For example, after many response-confirmed events involving rising final intonation followed by caregiver compliance, a schema emerges:

> {F = rising final intonation; R̂ = compliance / acknowledgment; c = high}

This is the prelinguistic ancestor of the interrogative.

After many events involving word A before word B in the context of action where A acts on B, a schema emerges:

> {F = order(A, B) in action context; R̂ = caregiver interprets A as agent; c = high}

This is the prelinguistic ancestor of subject-before-object word order in SVO languages.

Schemas accumulate, generalize, and recombine. The mature grammar is the high-confidence subset of the schema set.

---

## §3. Cross-linguistic predictions

The model makes specific predictions that the typological corpus data in the parent paper either supports or can adjudicate.

### Prediction 1: Universal attention substrate

A operates on cross-linguistically invariant dimensions. Therefore the attentional substrate of acquisition is the same across all languages. This is what we observed in the parent paper: utterance-edge cues exhibit the highest Attention Index in all seven typologically distinct languages examined.

### Prediction 2: Language-specific Book contents

What accumulates in the Book is language-specific because what gets response-confirmed depends on what the ambient language uses to mark communicative function. The Book of a Japanese-learning infant fills with sentence-final particles, case markers, and verb-final morphology because those are the elements that, when produced, occasion satisfying caregiver responses in Japanese-speaking environments. The Book of an English-learning infant fills with word-order patterns, articles, and auxiliaries.

This is what we observed in the parent paper: the highest-Reliability_form cues are radically different across languages (Mandarin R_form = 0.91 dominated by isolating markers; English R_form = 0.56 with positional dominance; Russian morphological dominance), yet the universal attention dimensions (acoustic edge, position) are remarkably consistent.

### Prediction 3: The grammar of a language is a description of its Book schemas

If grammar is the sedimented residue of response-confirmed schemas, then the grammar that linguists describe and the grammar that the infant develops should converge — not because the infant is acquiring the linguist's grammar, but because both are descriptions of the same accumulated schema set. The convergence is incidental, not constitutive.

This predicts that languages will exhibit grammatical regularities even where there is no genetic or contact relationship between them, simply because all human infants run the same I → A → O → R → B → schema pipeline. Universal grammar is, on this view, not innate but inevitable: it is what you get when you run RSM in any natural-language environment.

### Prediction 4: Acquisition difficulty tracks Book admission, not grammatical complexity

Elements that are grammatically present but poorly response-confirmed will be acquired late or imperfectly. Elements that are grammatically optional but strongly response-confirmed will be acquired early.

This predicts the well-known but theoretically uncomfortable observation that Japanese object marker -wo is frequently omitted by both caregivers and children: the marker is grammatically present in the ambient language, but its omission rarely produces R−, so it is weakly response-confirmed and weakly entered into the Book. The parent paper reported -wo at only 554 tokens against -ga at 2,959 tokens in Japanese caregiver speech. RSM predicts this asymmetry directly: -wo is grammatical but communicatively dispensable, and so under-represented in the Book.

### Prediction 5: The Tourist Phenomenon

If the model is correct, then attention-driven communication does not require the schema layer at all. The substrate I → A → O → R is intact in adult monolinguals encountering an L2 environment; what is missing is only the language-specific Book. The traveler is running the same pipeline as the infant, with the same attentional substrate, but with no accumulated Book in the new language.

This explains why practical communication succeeds without grammar: the substrate is sufficient. Grammar is a developmental and historical extension of the substrate, not a prerequisite for communication.

---

## §4. Relation to the Invisible Mediator Theorem

The model is an application of the Invisible Mediator Theorem ([CITATION REDACTED FOR ANONYMOUS REVIEW], multiple papers).

The conventional equation in the language acquisition literature has been:

> **Attention bias** = **filter for grammar acquisition**

This equation has been treated as definitional. RSM identifies the hidden mediator that has been silently held constant:

> M = "The learner is engaged in grammar acquisition as a goal."

Once M is recognized as a mediator rather than a constant, the equation reveals its conditional form:

> **Attention bias** = **filter for grammar acquisition** | (Learner is engaged in grammar acquisition as a goal)

When M is false — when the learner is engaged in communicative success-seeking rather than in grammar acquisition — the right-hand side of the equation does not apply. Attention bias is still operative, but its function is different: it is the substrate of communication itself, not a filter for grammar.

The discovery of M is the central theoretical contribution. Once M is named, the entire structure of L1 acquisition theory becomes reorganizable around the more accurate equation:

> **Attention bias** = **substrate of communicative success** | (any human language environment)

The grammar-relevant filtering function is a special case of the substrate when M (grammar acquisition) is in fact occurring, which is the developmental period of childhood.

---

## §5. What the model is not

We close by clarifying what the model does not claim.

**RSM is not behaviorism.** The Book is not a stimulus-response association table. It is a structured memory of signaling events with response confirmation, organized internally by the learner's communicative goals. The internal organization is constructivist and goal-directed, not reactive in the Skinnerian sense.

**RSM is not anti-UG.** The model is silent on whether the schema extraction mechanism is biologically constrained or biologically open. It is compatible with the view that the human brain has specific architectural properties that bias schema extraction in certain directions. It denies only that those properties are pre-formed grammatical knowledge.

**RSM does not deny the existence of grammar.** It denies that grammar is the target of acquisition. Grammar exists as a real cognitive and computational system in the mature speaker. RSM asserts that this system is constructed from the Book, not learned as a system.

**RSM does not equate child language with tourist communication.** The communicative repertoire of a fluent adult speaker vastly exceeds that of both children and tourists. The model asserts continuity of substrate, not equivalence of competence.

**RSM does not claim that all schemas are equally easy to acquire.** Schemas extracted from densely response-confirmed regions of the Book are robust; schemas extracted from sparsely confirmed regions are fragile. This predicts variation across both languages and individuals in schema acquisition trajectories.

---

## §6. The minimal commitments

To adopt the Reactive Schematization Model, a researcher must commit to four claims and four claims only:

(1) The infant's primary cognitive engagement with caregivers is communicative success-seeking, not grammar acquisition.

(2) Attention bias is a substrate property of human cognition that operates whether or not grammar acquisition is occurring.

(3) Response confirmation is what determines which signaling events enter long-term memory as patterns.

(4) Grammar is the descriptive summary of the accumulated response-confirmed schemas. It is real but derivative.

All five predictions in §3 follow from these four commitments, and the empirical results of the parent paper are consistent with all of them.

---

## §7. Future formalization needed

This document is a foundational draft. Several components of the model await further formalization:

- **The functional form of schema extraction from the Book.** What clustering algorithm best captures the empirical trajectory of child schema emergence? Is it parametric (with developmental schedule) or non-parametric (response-frequency-driven only)?

- **The relationship between Book size and schema confidence.** How many response-confirmed instances are required before a schema crosses the confidence threshold for productive use? Does this threshold vary by schema type (positional vs. morphological vs. lexical)?

- **The role of negative evidence (R−).** When a signaling event fails to produce S+, does it accumulate as anti-evidence in the Book, or is it simply not admitted? Both options have empirical consequences for overgeneralization patterns.

- **Cross-linguistic Book typology.** Can we measure the actual composition of children's Books across languages, and does it predict the Reliability profiles we observed in caregiver input?

- **The bridge to second-language acquisition.** If the substrate is universal but the Book is language-specific, then L2 acquisition involves Book-building on top of an already-mature substrate. What is the relationship between L1 Book and L2 Book acquisition?

These questions define the immediate research program that RSM opens. The parent paper provides the empirical foundation (typological mapping of the attentional substrate). RSM provides the theoretical scaffolding. The next set of papers will need to characterize the Book itself, both developmentally and cross-linguistically.

---

## §8. Coda: Why this matters

The standard model of language acquisition has had a strange property: it has produced increasingly precise descriptions of what the child does without ever quite explaining why. Why does the child generalize from "the dog ran" to "the cat ran" rather than to "the dog flew"? The standard answer invokes innate constraints or distributional statistics. RSM gives a simpler answer: because "the dog ran" was response-confirmed and "the dog flew" was not, and the schema extracted from the Book respects this.

The Tourist Phenomenon is not an anomaly. It is the model's behavior in the limit case where M (grammar acquisition) is absent. The traveler in Hawaiʻi is doing what every human language user is, at root, doing: deploying signaling resources to move others, and refining those resources based on whether the others move.

The infant differs from the tourist in only one respect: the infant has the entire developmental window in which to fill its Book, and so eventually develops the dense schema set that we call native grammar. The tourist, with a richer attentional substrate but an empty L2 Book, gets food and a bus ticket. The infant, given enough time, gets a language.

In both cases, what is doing the work is the same: I → A → O → R → B → schema.

This is the entire model.

---

## Appendix: Notation reference

| Symbol | Meaning |
|---|---|
| **I** | Input stream (caregiver speech, gesture, context) |
| **A** | Attention function over I; uses 5 salience dimensions |
| **O** | Infant output (cry, gaze, gesture, sound, word, combination) |
| **R** | Caregiver response to O |
| **R+ / R−** | Satisfying / non-satisfying response |
| **B (Book)** | Store of (O, R+) pairs |
| **Schema** | Cluster extracted from B |
| **M** | Hidden mediator: "Learner is engaged in grammar acquisition" |

The pipeline:

> I → A → O → R → B → schema → next-generation O

The IMT-revealed equation:

> Attention bias = substrate of communicative success | (any human language environment)
