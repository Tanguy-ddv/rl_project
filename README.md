# Reinforcement Learning project of Machine Learning and Deep Learning 2024

## Students

Student group:
s282575_s331543_s321277_SUPPA_PERRUCCI_DUGASDUVILLARD

Lorenzo Suppa, Vito Perrucci, Tanguy Dugas du Villard

Politecnico di Torino

Machine Learning and Deep Learning course, Spring semester 2024.

## Code organisation overview

The code presented here doesn't stick to the provided template. In order to make it easier to deal with multi-step training and data saving, a wrapper class named 'Session' has been created.
In one single session, it is easy to train one, or multiple agents, load, reload and save models, as well as keep track on the metrics. The main part of the codes are the train.py and test_agent.py functions, that are called at nearly every steps. The student is able to interact with the session using its methods

A detailed explanation of the project's structure, approach, and results is provided in the project report, which can be found in the PDF file named s282575_s331543_s321277_SUPPA_PERRUCCI_DUGASDUVILLARD.pdf.

## Tasks

### Reinforce and AC

The two version of the agent (reinforce with (and without) baseline and the actor-critic) has been coded in two different files: agent/actor_critic_agent.py and agent/reinforce_agent.py. Both extend the agent.agent.Agent class, that implements all the agents methods provided in the template, but the update_policy one that is defined differently for each type of agent. Both agent classes represent the implementation of the tasks 2 and 3.

### PPO

We use stable-baseline's PPO, on which we add callbacks to save the training reward. The PPOSession is used to perform experiment on it.

To train, we use the TrainAndTestCallback to store the training reward, optionally regularly train the PPO on a different environment during its training, and optionnally set a maximum number of episodes.

### UDR and GDR.

We slightly modifed the CustomHopper to make a GDRHopper and an UDRHopper, whose masses can be modified easily. We use them to perfrom UDR or GDR experiments. With PPO, we use GDRCallback and UDRCallback.

### ADR

The ADR is perform on PPO via th ADRCallback, using discriminator and particles defined in the domain_randomization folder, alongside with the UDRHopper and GDRHopper.
