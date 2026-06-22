I want to recreate repo for manifold steering myself in my own style but replicate the results
I am attaching the paper and its following repo:
https://github.com/goodfire-ai/causalab/tree/manifold_steering
The repo is cloned to C:\projects\causalab-reference-manifold-steering

please lets code together section by section i will be the guide, you create the code and i will execute the code and guide further

first rule: dont code without we reaching there together. no extra coding. we collaborate and do this together. 
example: you couldve asked first before coding it all if we wanna do all the 4 different prompts. let's do just weekdays. also simple script please

Python is at C:\Python313\python.exe
Use this exact interpreter for project scripts:
  C:\Python313\python.exe <>.py

PyTorch is installed in the user-site packages:
  C:\Users\ragha\AppData\Roaming\Python\Python313\site-packages

Codex sandboxed commands may not be able to read that user-site path, which makes
`import torch` look missing even though it works in the user's terminal. When
running torch/model code from Codex, request escalated execution for
`C:\Python313\python.exe` so the interpreter can see the user-site packages.

following plan lines:
1. load data

2. load model

3. get activation and probabilities

4. Baseline Experiment
Run all 49 weekday prompts through the model and save:
raw_input, expected answer, model top prediction, correctness, weekday probabilities, and maybe "other" probability.

5. Choose Target Layer(s)
Decide which layers we care about. For SmolLM maybe sweep a few layers; for Llama 8B later we can mirror the paper’s likely residual-stream layer choice.

6. Collect Dataset Activations
Use our combined function over all weekday prompts:
activations per layer, probabilities per prompt, metadata from WeekdayExample.

7. Make Activation Subspace
Start simple with PCA on activations for one layer.
This is the repo’s subspace idea, but we can do a small version: fit PCA, project activations to 2D/3D, color by answer weekday or source weekday.

8. Output/Probability Geometry
Analyze the model’s probability vectors over weekdays.
The repo calls this output manifold-ish: probabilities live in belief/simplex space. We can visualize weekday probability distributions, maybe with PCA on sqrt(probabilities) later.

9. Compare Activation Geometry vs Output Geometry
Ask: do activations form a cyclic weekday structure? Do probability vectors form the same cycle?
This is where the “manifold” idea starts becoming visible.

10. Steering
Compute class centroids in activation space, e.g. mean activation for prompts whose answer is Monday, Tuesday, etc.
Then patch/add/interpolate activations toward another weekday centroid and see if probabilities move accordingly.

11. Path Steering
Instead of jumping from one centroid to another, interpolate along a path between weekday centroids and collect output probabilities at each step.
