# Direct Preference Optimization Implementation

This repository contains an implementation of **Direct Preference Optimization (DPO)**, a policy optimization method for reinforcement learning that directly optimizes user preferences.

In this repository, we make available the implementation, experiments comparing DPO with Supervised Fine-Tuning (SFT), and a full theoretical and experimental report on the DPO method.

## Report

The report, titled *Direct Preference Optimization: A Theoretical and Experimental Analysis*, includes a full mathematical revision of the method, covering the complete theoretical framework from the *Plackett General Model for Preference Probabilities* to the most recent advances in *Direct Preference Optimization*.

It includes:

- A clear derivation of the DPO loss function using the Bradley-Terry and Plackett-Luce models.
- A reformulation of the KL-constrained reward objective into a closed-form expression, leading to a simpler training procedure.
- Detailed theoretical insights, including proofs showing that language models can be seen as *implicitly trained reward models*.
- A comparison of DPO with PPO and RLHF methods, highlighting DPO's *computational simplicity and stability*.
- Experimental results based on *GPT-2-medium*, along with preference datasets, showing the *effectiveness of DPO* even with small models.
- Evaluation using *Sentence-BERT* to approximate win-rates based on semantic similarity.


The full report can be found [here](./report.pdf) and provides an accessible yet rigorous mathematical background for the method, its motivation, and implications.

## Installation

### Prerequisites

Ensure you have **Python 3.8+** installed. Then, clone the repository and install the dependencies:

```bash
git clone https://github.com/giovanni-br/DPO-Implementation.git
cd DPO-Implementation

pip install -r requirements.txt
