# Autonomous Option Invention for Continual Hierarchical Reinforcement Learning and Planning

The repository serves as an implementation of the paper "Autonomous Option Invention for Continual Hierarchical Reinforcement Learning and Planning" above providing a framework for continual hierarchical planning and reinforcement learning. It takes as input a set of state variables and a stochastic simulator, and invents options with abstract representations. With every new task in a continual stream of problems, CHiRP transfers these options and invents new options, building a model of options that is more broadly useful. 

Full paper is available at: https://aair-lab.github.io/Publications/rkn_aaai25.pdf

## Instructions to reproduce the experiments

1. Set the domain name, the approach name, and trial number in the shell script files provided. Use run_experiment_single.sh to run for a single method and domain, and use run_experiments_together.sh to run multiple domains and methods.

2. All the hyperparameters are provided in hyper_param.py and can be changed if needed. 

3. Run the shell script file.

```
sh run_experiment_single.sh
sh run_experiments_together.sh
```

4. The results will be stored in the results/ directory.


# Citation
```
@article{nayyar2024autonomous,
  title={Autonomous Option Invention for Continual Hierarchical Reinforcement Learning and Planning},
  author={Nayyar, Rashmeet Kaur and Srivastava, Siddharth},
  journal={arXiv preprint arXiv:2412.16395},
  year={2024}
}
```