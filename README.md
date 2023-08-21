# mario-ai-2023
A new fun-AI-course on Game and AI

## V1 ready
- Slides at [here](./slides/pub_v1_szhu_ai_game_intro_082023.pdf)
- Three notebooks
  - [Cart Pole AI](./notebooks/cartpole_e2e_08202023.ipynb)
  - [Mario Game Visualization](./notebooks/mario_visualize.ipynb)
  - [Mario AI skeleton](./notebooks/mario_ai_e2e_08202023.ipynb)
- A small sup lib
  - Mario control/env/visualization related

## Plan as of 08/07
- Write helper config and functions as committed files, including but not limited to
  - package dep as requirements.txt
  - support methods such as downsampling and frame stacking
- Add a new trainer colab to use DQN (instead of more complex algorithms) to get started
  - Fix the reward calculation
  - Add support eval and recording functions
  - Add support of eval of multiple stages (multiple runs per stage, avg of multiple stages)
- A new version of slides
  - A more enhanced version with AI-generated AI Mario pics
  - A simpler version to only touch DQN, but leaving referencs to other algorithms
  - Explain the challenge, and give a few hints

## Progress log
- 08/07: Initial plan
- 08/08: Add basic env/visual/control lib, and a visualization notebook of all 32 stages
- 08/20: V1 slides and 3 notebooks ready
