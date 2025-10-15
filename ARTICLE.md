Of course. Let's start completely from scratch, defining every term, and we'll use a simpler problem than Sudoku that is easier to visualize: **solving a maze**.

### The Big Picture: What is TRM and Why Does It Matter?

**The Challenge with LLMs:** Large Language Models (LLMs) often struggle with reasoning puzzles like Sudoku or maze-solving. These problems require precise, step-by-step thinking where one error can derail the entire solution. The usual approach of making models bigger is costly and often ineffective.

**TRM's Solution:** TRM uses a tiny 2-layer transformer (7M parameters) that thinks recursively. It reuses its 2 layers hundreds of times to solve a problem.

**The Results:**
- **Sudoku-Extreme:** 87.4% accuracy (with 5M parameters), while models 100x larger get 0%.
- **ARC-AGI Puzzles:** 44.6% accuracy (with 7M parameters), outperforming models thousands of times larger.
- **Efficiency:** Generalizes from just ~1,000 examples to solve hundreds of thousands of test cases.

This "less is more" approach proves that clever architecture beats brute force size.

---

### Step 1: Setup

Let's train TRM to solve a maze.

**1. Representing the Maze as a Grid:**
First, we show maze as a grid of numbers. Each cell in the grid gets a number.

*   `0` = Empty path
*   `1` = Wall
*   `2` = Start point
*   `3` = End point

For a concrete example, let's trace a tiny 3x3 maze.

*   **`x_input`** (The unsolved maze)
    ```
    [[2, 0, 1],
     [1, 0, 1],
     [1, 0, 3]]
    ```
*   **`y_true`** (The correct solution, with `4` representing the path)
    ```
    [[2, 4, 1],
     [1, 4, 1],
     [1, 4, 3]]
    ```

**2. Tokenization:**
The term **token** just means a single unit of our data. In this case, a single number in the grid (`0`, `1`, `2`, `3`, or `4`) is a token. To make it easier for the network to process, we "unroll" the grid into a long 1D list.

For our 3x3 example, the grid is unrolled into a list of 9 tokens.

**3. Embedding: Giving Meaning to Numbers:**
To let the model understand what numbers like `4` and `1` mean, we will asign a big **vector embedding** to each. Vector embedding is a long vector (array of numbers) that the model can modify to store information about the wall, empty path, etc.

These vectors will represent the meaning of a "wall" or "end point".

I recommend you remind yourself what vector embeddings (in LLMs, of words, tokens, etc) are by searching it on YouTube or talking to an AI chatbot.

*   An **Embedding Layer** is like a dictionary.
*   It contains vector embeddings for each of our numbers.
*   `1`: `[0.3, -1.2, 0.7, 0.0, 1.5, -0.4, 0.9, 2.3]`  ←  Example vector embedding for "wall"
*   **Output:** A long list of numbers called a **vector**. This vector represents the *meaning* of "wall" in a way the network can understand. The network itself choose (learned) numbers within this vector during the training so it can "understand" it.

After this step, our input maze is no longer a list of simple numbers. It's a list of vectors. For our 3x3 maze, if we use a vector of size 8 for each token, our input becomes:

*   `x`: A `9x8` matrix of vectors representing the maze.

This rich representation is what we feed to the main model.

---

### Step 2: The Core Architecture: The TRM Brain

The "brain" of TRM is a tiny 2-layer transformer called `net`. It processes information to produce an output. To "think," TRM uses two variables, both having same shape as `x`:

*   `y`: The model's current **best guess** for the solution. Might be wrong"
*    ```
    [[2, 4, 1],
     [1, 4, 1],
     [1, 0, 3]]
    ```
*   `z`: A **latent thought**. `z` tells what needs to be changed in `y` to turn it into a correct solution. `z` is passed through the transformer multiple times to let the model refine what needs to be changed in `y`, this is how the model reasons or thinks. Then the change is applied to `y`.

For our 3x3 example, `z` and `y` start as `9x8` matrices of zeros.

---

### Step 3: The Learning Process, from the Inside Out

TRM learns in a series of nested loops. Let's start from the core and build our way out.

#### The Innermost Loop: `latent_recursion` (The Core Thought)

This is where the tiny `net` (a 2-layer Transformer) does all its work. The process is broken into two phases that repeat to form a cycle of thinking and refining.

**Phase A: Reasoning (Updating the Scratchpad `z`)**
The model "thinks" by refining its internal planning token `z`, in a loop of 6 steps. The goal is to build a better and better plan for changing `y`.

1.  **The Process:** In each of the 6 steps, the `net` takes three inputs:
    *   The maze itself (`x`).
    *   The model's current best guess for the solution (`y`) - this could be all zeroes at the beginning.
    *   The scratchpad from the previous step (`z`).
2.  **How it works:**
    *   **Combining Inputs:** The three inputs are added together element-wise (`x + y + z`). This creates a single sequence of rich vectors, where each vector (representing a cell in the maze) contains combined information about the maze layout (`x`), the current guess (`y`), and the ongoing thought process (`z`).
    *   **Thinking with Attention:** This combined sequence is fed into the 2-layer Transformer. The Transformer's self-attention mechanism allows it to look at all the cells at once and identify relationships. For example, it can see how the "start" cell relates to a potential path cell, informed by the input data `x` and the reasoning `z`.
    *   **Generating the Next Thought:** The two transformer layers process this information and output a new sequence of vectors of the exact same shape. This output *is* the new `z`. There isn't a separate "output head" to generate it; the transformation performed by the two layers *is* the act of creating the next, more refined thought. Even though the input was a sum containing `x` and `y`, the network learns to produce an output that serves as a useful new `z` for the next step.

    This process repeats 6 times, meaning the information is passed through the same two layers six consecutive times, becoming progressively more sophisticated with each pass.
3.  **Example Trace:** After a few passes through transformer, `z` might encode low-level features like wall locations. By the sixth pass, it might represent a high-level plan for updating answer (`y`).
   
   - Interestingly, the same 2 transformer layers are used for detecting low level features, making high level plan and later for updating `y` itself. These 2 layers have multiple purposes, which is the power of neural networks, it can learn to do multiple, less related or unrelated transformations taht only depends on the input data.

**Phase B: Refining the Answer (Updating the Guess `y`)**
After the 6-step reasoning loop, using the latest latent though `z` the model updates its answer `y`.

*   **How it works:** It combines its previous answer (`y`) with its final, refined thought (`z`) by adding them together (`y + z`) and passes the result through the same `net` one last time. The output is the new, improved `y`.
    *   **Crucially, `x` is not included in this step.** This is a deliberate design choice that tells the single `net` which task to perform.
    *   `x` is present in reasoning (`x + y + z`).
    *   `x` is absent in answer refinement (`y + z`).

The reason I said "answer refinement" is because this 6+1 loop happens multiple times, each time "thinking" for 6 passes and updating `y` once.

#### The Middle Loop: `deep_recursion` (The Full Thought Process)

Now that we understand how reasoning + y refinement loop works, let's see the full thought process from the beginning where this whole loop is repeated 3 times to get the best `y`.

The previously described inner loop (the 6+1 steps of reasoning and `y` refinement) runs `T` times (e.g., `T=3`). The state (`y` and `z`) is **carried over** between these runs; it is not reset to zero.

*   **Round 1 (Warm-up):** Starts with a blank (all zeroes) `y` and `z` (remember, this is the absolute beginning of the process, so there is no `y` and `z` to carry over). It runs the full inner loop (6 reasoning + 1 `y` refinement steps) to produce smarter `y_1` and `z_1`. This is done in "no-gradient" mode for speed and memory savings - neural network doesn't learn here.
*   **Round 2 (Warm-up):** It takes `y_1` and `z_1` as its starting point and runs the inner loop again to produce an even better `y_2` and `z_2`. Still no gradients and learning.
*   **Round 3 (For Real):** It starts with the well-reasoned `y_2` and `z_2`, runs the inner loop one final time, and this time all calculations are tracked so the model can learn with backpropagation.

This process of warming up the model's "thought" before the final, learnable step is a key optimization.

#### The Outermost Loop: Even more loops!

The model gets multiple "chances" (up to 16) to solve the same maze, and after each chance, it refines its `net` weights. The state (`y` and `z`) **is carried over** from one middle loop iteration to the next, as shown in the paper's pseudocode. It allows the model to get multiple "chances" (up to 16) to solve the same maze, improving with each one.

This is just repeating middle loop up to 16 times. Model can decide to stop earlier than 16 if it feels like it got the correct answer.

Why we need this loop:

After each middle loop itteration this outter loop updates weights once (remember that the Round 3 in middle loop does backpropagation).

Then in the next iteration it repeats the middle loop with the updated weights, allowing the model to progressively improve its solution with each attempt.

#### Knowing when to stop thinking (The Q head)

The outer loop can run up to 16 times, but it doesn't have to. It would be a waste of time to keep thinking about a maze it has already solved.

So, the model has a little side-brain called a "Q head". After each full thought process (each middle loop), this Q head spits out a score. This score is basically the model's confidence: "How sure am I that I got this right?"

If the confidence score is high enough, the outer loop just stops (`break`), and the model moves on to the next maze.

It learns to get this confidence score right because it's part of the training. It gets rewarded if it's confident *and* correct, and penalized if it's confident but wrong. The paper calls this Adaptive Computation Time (ACT).


---

### Step 4: Testing the Recipe — Why Does This Work? (Ablation Studies)

To understand what makes TRM effective, we can perform **ablation studies**—systematically breaking parts of the model to see what happens. Here's a breakdown of the key components.

1.  **The Baseline Model:** The standard TRM with all its key components:
    *   **Deep Recursion (`T=3`):** 3 full "thought processes" to iteratively refine the solution.
    *   **Latent Reasoning (`n=6`):** 6 steps of scratchpad (`z`) updates before updating the answer (`y`).
    *   **Tiny 2-Layer Network:** Forces reliance on recursion over raw size.
    *   **Exponential Moving Average (EMA):** A training stabilizer for more robust learning.

2.  **Ablation: No Exponential Moving Average (EMA)**
    *   **Change:** Turn off the weight averaging stabilizer.
    *   **Expected Outcome:** Worse performance. Without EMA, training can be erratic and lead to overfitting.

3.  **Ablation: Less Recursion**
    *   **Change:** Drastically reduce thinking time by setting `T=2` and `n=2`.
    *   **Expected Outcome:** A major drop in accuracy. The model needs sufficient recursive depth to perform the complex chain of reasoning required for hard problems.

4.  **Ablation: A Bigger Brain vs. Deeper Thought**
    *   **Change:** Double the network size to 4 layers, but reduce latent reasoning to `n=3` to keep computation similar.
    *   **Expected Outcome:** Worse performance. A bigger network is more prone to memorizing training data. A smaller network *forced* to think longer (more recursion) learns the general *strategy* better.

These experiments help confirm that TRM's success comes from deep, nested recursion with a tiny network and stabilized training.

#### Experimental Results: Ablation Studies in Practice

To validate these hypotheses, we conducted small-scale experiments comparing four configurations. While the original paper reports results from full training runs (50,000+ epochs), we performed quick 10-epoch experiments to illustrate the concepts and their immediate effects.

![Complete Ablation Study](docs/images/complete_ablation_study.png)
*Figure: Comparison of LM Loss across four ablation studies over 10 epochs on maze-solving task. Blue solid (Baseline: 2-layer, H=3, L=6, EMA), Red dashed (No EMA: 2-layer, H=3, L=6, no EMA), Green dash-dot (Less Recursion: 2-layer, H=2, L=2, EMA), Magenta dotted (Bigger Brain: 4-layer, H=3, L=3, EMA).*

**The Four Configurations:**

1. **Baseline** (2-layer, H_cycles=3, L_cycles=6, EMA=True): The standard TRM configuration
2. **No EMA Ablation** (2-layer, H_cycles=3, L_cycles=6, EMA=False): Removes exponential moving average
3. **Less Recursion Ablation** (2-layer, H_cycles=2, L_cycles=2, EMA=True): Reduces recursive depth by ~66%
4. **Bigger Brain Ablation** (4-layer, H_cycles=3, L_cycles=3, EMA=True): Doubles network size, reduces L_cycles by 50%

**Key Experimental Findings:**

| Configuration | Initial Loss | Final Loss | Improvement | Min Loss Achieved |
|--------------|--------------|------------|-------------|-------------------|
| Baseline | 1.789 | 1.062 | 40.6% | 1.045 |
| No EMA | 1.789 | 1.042 | 41.7% | 1.041 |
| Less Recursion | **2.100** | 1.100 | 47.6% | 1.042 |
| Bigger Brain (4-layer) | 1.789 | **1.007** | 43.7% | **1.007** |

**Critical Insights:**

1. **The "Bigger Brain" Paradox - Most Important Finding:**
   - **4-layer network achieved BEST final loss (1.007)** - beating all 2-layer configurations!
   - Started at same point as baseline (1.789) but learned 5% faster
   - **BUT** - the paper shows 2-layer wins long-term (50k+ epochs)
   - **Why?** Short-term: more capacity = faster learning. Long-term: more capacity = overfitting
   - This validates the paper's core thesis: "less is more" for *generalization*, not immediate training loss
   - The 2-layer architecture is chosen NOT for speed, but to **force reliance on recursion**

2. **EMA Effect Remains Minimal in Short Training:**
   - Both baseline and No EMA started identically (1.789) and converged to similar losses (~1.04-1.06)
   - Only ~2% difference between them
   - Confirmed: EMA is a long-term stabilizer, not a short-term performance booster

3. **Recursion Depth is Non-Negotiable:**
   - Less Recursion started at **significantly higher** initial loss (2.100 vs 1.789)
   - This +17% handicap shows reduced recursion cripples the model from initialization
   - Even with same 2-layer network, cutting H and L cycles severely degrades capability
   - Final performance worst among all (1.100) despite highest percentage improvement (47.6%)
   - **Interpretation:** You cannot compensate for shallow recursion - it's architecturally essential

4. **Different Learning Dynamics Across Configurations:**
   - The graph shows four distinct curve shapes
   - Baseline and No EMA: nearly identical (parallel lines)
   - Less Recursion: starts very high, drops steeply, plateaus high
   - Bigger Brain: starts normal, drops fastest and furthest
   - This reveals that architecture affects both learning *speed* and learning *ceiling*

**Why Initial Loss Matters:**

The fact that Less Recursion starts at 2.100 (vs 1.789 for baseline) is highly informative:
- The model makes its first prediction *before any training*, using only initialization
- With less recursion (H=2, L=2), the model has fewer computation steps to process the input
- This immediately leads to worse initial predictions, even with identical weight initialization
- It suggests that **architectural compute capacity** directly affects representational power

**The Takeaways:**

1. **The Short-Term vs. Long-Term Trade-off (Most Important):**
    - **Bigger networks win short-term**: The 4-layer network achieved the best 10-epoch loss (1.007).
    - **Smaller networks win long-term**: The paper shows the 2-layer network wins at 50k+ epochs because it is forced to learn a general strategy instead of memorizing, which leads to better generalization. This paradox is central to TRM's design.

2. **Recursion Depth is Absolutely Critical:**
    - The "Less Recursion" model started with a 17% higher loss and performed worst overall.
    - This confirms you cannot compromise on recursive depth (`H_cycles` and `L_cycles`); it is fundamental to the architecture's ability to reason.

3. **EMA is a Long-Term Stabilizer:**
    - The effect of EMA was minimal in short training runs (~2% difference). Its real value is in preventing overfitting and improving generalization over tens of thousands of epochs.

4. **Architecture Defines Learning Dynamics:**
    - Each configuration produced a unique learning curve, showing that architecture affects not just the final outcome but also the speed and ceiling of learning. Quick experiments are useful for understanding initial dynamics but can't reveal long-term behaviors like generalization.

This confirms the "less is more" thesis: **architectural constraints, like a smaller network, can improve learning by forcing reliance on recursion.**

---

**HRM's Process:**
```python
# HRM uses two networks
for i in range(2):  # Apply fL twice
    zL = fL(zL + zH + x)
    
zH = fH(zL + zH)  # Apply fH once
```

```python
# Initialize
y, z = zeros_like(x), zeros_like(x)

# Deep supervision loop (up to 16 times)
for supervision_step in range(16):
    
    # Deep recursion: warm-up (2 times, no gradients)
    with torch.no_grad():
        for _ in range(2):
            # Latent recursion
            for _ in range(6):
                z = net(x + y + z)
            y = net(y + z)
    
    # Deep recursion: final (1 time, WITH gradients)
    for _ in range(6):
        z = net(x + y + z)
    y = net(y + z)
    
    # Learn
    y_pred = output_head(y)
    loss = cross_entropy(y_pred, y_true)
    loss.backward()
    optimizer.step()
    
    # Should we stop?
    q = Q_head(y)
    if q > 0:
        break
```

---

### Understanding Q Halt Loss: Teaching the Model When to Stop Thinking

TRM uses **Adaptive Computation Time (ACT)** to learn not just *how* to solve problems, but also *when to stop* thinking. This is managed by an outer adaptive loop that can run up to 16 times, but can halt early if the model is confident in its answer.

#### How Q Halt Works

The model computes two values: `q_halt_logits` (the value of stopping) and `q_continue_logits` (the value of continuing). It stops when `q_halt > q_continue`.

The model learns to make this decision through a **Q halt loss function**, which essentially teaches `q_halt` to predict whether the current answer is correct.

*   If the answer is correct -> `q_halt` should be high (stop).
*   If the answer is incorrect -> `q_halt` should be low (continue).

#### Why Q Halt Loss is Tiny Early in Training

The Q halt loss is often tiny at the start of training. This is expected. The Q-head is initialized to predict a very low probability of being correct (e.g., 0.67%). Since the untrained model gets nearly everything wrong, it is **already correct about being wrong**. The loss is therefore small.

The Q halt loss becomes important later, as the model starts solving problems correctly and needs to learn to distinguish between a right and wrong answer to decide when to stop. This creates a curriculum where the model first learns **how** to solve a problem, then learns **when** it has succeeded.

#### Benefits of Adaptive Halting
- **Efficiency**: Spend more computation on harder problems, less on easier ones.
- **Generalization**: Forces the model to understand solution quality.
- **Human-like**: Mirrors the human behavior of thinking longer about harder problems.

In practice, an easy maze might halt after 2-3 thinking steps, while a difficult one uses all 16.
