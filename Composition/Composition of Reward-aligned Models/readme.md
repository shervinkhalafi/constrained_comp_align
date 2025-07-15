This folder implements the experiments on Composing models fine-tuned on different rewards, using the Product composition of diffusion models.

To reproduce the experiments described on presented in Section 5 (I) and Appendix F.2, first you need to finetune adapters for each reward using the bash script ``single_reward_for_composition.sh``.

Then run one of the three bash scripts:

* `all_animals_constrained.sh`: product composition using constraints
* `all_animals_equal_weights.sh`: baseline that gives all adapters the same weight
* `all_animals_single_reward.sh`: sampling solely from each adapter, used to normalize the rewards for comparison

 You can also change ``classifier_class_names`` to change the conditioning prompt.
