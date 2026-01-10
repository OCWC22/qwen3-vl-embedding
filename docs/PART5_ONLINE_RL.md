# PART 5: ONLINE REINFORCEMENT LEARNING
## PPO and GRPO - Learning from Environment Interaction

---

## 5.1 WHAT IS ONLINE RL?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ONLINE vs OFFLINE RL                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OFFLINE RL (DPO):                                                          │
│  ────────────────                                                           │
│  ┌─────────────┐                                                           │
│  │ Pre-collected│──► Train ──► Model                                       │
│  │ Preferences  │                                                          │
│  └─────────────┘                                                           │
│  No environment interaction during training                                │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────     │
│                                                                             │
│  ONLINE RL (PPO/GRPO):                                                      │
│  ─────────────────────                                                      │
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                  │
│  │   Model     │────►│ Environment │────►│   Reward    │                  │
│  │ (generates  │     │ (Higgsfield)│     │ (task done?)│                  │
│  │  actions)   │◄────│             │◄────│             │                  │
│  └─────────────┘     └─────────────┘     └─────────────┘                  │
│         │                   │                   │                          │
│         └───────── LEARNING LOOP ───────────────┘                          │
│                                                                             │
│  Model INTERACTS with environment, gets FEEDBACK, improves                 │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  ANALOGY:                                                                   │
│                                                                             │
│  Offline RL = Learning to cook from recipe books                           │
│  Online RL  = Learning to cook by actually cooking (and tasting)           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5.2 PPO vs GRPO

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PPO (Proximal Policy Optimization)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ARCHITECTURE:                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                                                                       │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │ │
│  │  │  Policy  │  │  Value   │  │  Reward  │  │Reference │             │ │
│  │  │  Model   │  │  Model   │  │  Model   │  │  Model   │             │ │
│  │  │  (LLM)   │  │ (Critic) │  │ (Critic) │  │  (LLM)   │             │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘             │ │
│  │      32B          32B           32B           32B                    │ │
│  │                                                                       │ │
│  │  TOTAL: 4 × 32B = 128B parameters!                                   │ │
│  │                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  LOSS FUNCTION:                                                             │
│  L_PPO = min(r_t · A_t, clip(r_t, 1-ε, 1+ε) · A_t)                        │
│                                                                             │
│  Where:                                                                     │
│  - r_t = π_θ(a|s) / π_old(a|s)  (probability ratio)                       │
│  - A_t = advantage (how much better than expected)                         │
│  - ε = clipping parameter (typically 0.2)                                  │
│                                                                             │
│  PROS:                                                                      │
│  ✓ Can explore beyond training data                                        │
│  ✓ Learns from actual environment feedback                                 │
│  ✓ Theoretically higher ceiling                                            │
│                                                                             │
│  CONS:                                                                      │
│  ✗ 4 models = massive memory                                               │
│  ✗ Reward model can be hacked                                              │
│  ✗ Training is unstable                                                    │
│  ✗ Slow (generate → evaluate → update loop)                               │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                    GRPO (Group Relative Policy Optimization)                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ARCHITECTURE:                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                                                                       │ │
│  │  ┌──────────┐                              ┌──────────┐              │ │
│  │  │  Policy  │ ─── Generate N responses ──► │Reference │              │ │
│  │  │  Model   │                              │  Model   │              │ │
│  │  │  (LLM)   │ ◄── Group-based advantage ── │  (LLM)   │              │ │
│  │  └──────────┘                              └──────────┘              │ │
│  │      32B                                       32B                    │ │
│  │                                                                       │ │
│  │  NO VALUE MODEL! NO REWARD MODEL!                                    │ │
│  │  TOTAL: 2 × 32B = 64B parameters (50% less than PPO!)               │ │
│  │                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  KEY INNOVATION:                                                            │
│  Instead of learning a value function, GRPO:                               │
│  1. Generates N responses per prompt (e.g., N=8)                           │
│  2. Gets reward for each response                                          │
│  3. Uses GROUP MEAN as baseline                                            │
│  4. Advantage = reward - group_mean                                        │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │ Prompt: "Click the login button"                                   │   │
│  │                                                                     │   │
│  │ Response 1: "ACTION: CLICK(512, 420)"        Reward: +1.0         │   │
│  │ Response 2: "I'll help you login..."         Reward: +0.5         │   │
│  │ Response 3: "What's your password?"          Reward: -0.5         │   │
│  │ Response 4: "ERROR: cannot compute"          Reward: -1.0         │   │
│  │                                                                     │   │
│  │ Group Mean = (1.0 + 0.5 - 0.5 - 1.0) / 4 = 0.0                    │   │
│  │                                                                     │   │
│  │ Advantages:                                                         │   │
│  │ Response 1: 1.0 - 0.0 = +1.0 (reinforce!)                         │   │
│  │ Response 2: 0.5 - 0.0 = +0.5 (slightly reinforce)                 │   │
│  │ Response 3: -0.5 - 0.0 = -0.5 (discourage)                        │   │
│  │ Response 4: -1.0 - 0.0 = -1.0 (strongly discourage)               │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  LOSS FUNCTION (GRPO):                                                      │
│  L_GRPO = -E[ A_i · log π_θ(y_i|x) ] + β · KL[π_θ || π_ref]              │
│                                                                             │
│  PROS:                                                                      │
│  ✓ 50% less memory than PPO                                                │
│  ✓ No value model to train                                                 │
│  ✓ More stable (group normalization reduces variance)                      │
│  ✓ Used by DeepSeek-R1 (state-of-the-art reasoning)                       │
│                                                                             │
│  CONS:                                                                      │
│  ✗ Needs to generate N responses (slower than PPO per step)               │
│  ✗ Still needs environment interaction                                     │
│  ✗ Reward function must be designed carefully                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5.3 REWARD FUNCTION DESIGN

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REWARD FUNCTION FOR COMPUTER USE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SPARSE REWARD (simple but slow to learn):                                 │
│  ──────────────────────────────────────────                                │
│  reward = +1 if task_completed else 0                                      │
│                                                                             │
│  Problem: Model gets no signal until task is done                          │
│           May take 10+ steps → no learning signal for first 9              │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────     │
│                                                                             │
│  DENSE REWARD (better learning signal):                                     │
│  ─────────────────────────────────────                                      │
│                                                                             │
│  def compute_reward(action, prev_state, curr_state, goal):                 │
│      reward = 0                                                             │
│                                                                             │
│      # Format reward: Did it output valid ACTION?                          │
│      if "ACTION:" in action:                                               │
│          reward += 0.1                                                      │
│      else:                                                                  │
│          reward -= 0.2  # Penalize no action                               │
│                                                                             │
│      # Progress reward: Did UI change in right direction?                  │
│      if closer_to_goal(curr_state, goal):                                  │
│          reward += 0.3                                                      │
│      elif further_from_goal(curr_state, goal):                             │
│          reward -= 0.2                                                      │
│                                                                             │
│      # Efficiency: Fewer steps is better                                   │
│      reward -= 0.05  # Small penalty per step                              │
│                                                                             │
│      # Success bonus                                                        │
│      if task_completed(curr_state, goal):                                  │
│          reward += 1.0                                                      │
│                                                                             │
│      # Error penalty                                                        │
│      if caused_error(curr_state):                                          │
│          reward -= 0.5                                                      │
│                                                                             │
│      return reward                                                          │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────     │
│                                                                             │
│  REWARD COMPONENTS FOR HIGGSFIELD:                                          │
│                                                                             │
│  │ Component           │ Reward  │ Trigger                     │          │
│  ├─────────────────────┼─────────┼─────────────────────────────┤          │
│  │ Valid ACTION format │ +0.1    │ "ACTION:" in response       │          │
│  │ Clicked correct UI  │ +0.3    │ Click on expected element   │          │
│  │ Typed correct text  │ +0.2    │ Input matches expected      │          │
│  │ Screen progressed   │ +0.2    │ New screen is goal-related  │          │
│  │ Task completed      │ +1.0    │ Video generation started    │          │
│  │ No ACTION           │ -0.2    │ Missing ACTION command      │          │
│  │ Wrong element       │ -0.3    │ Clicked wrong thing         │          │
│  │ Caused error        │ -0.5    │ Error dialog appeared       │          │
│  │ Step penalty        │ -0.05   │ Per step (efficiency)       │          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5.4 GRPO TRAINING PIPELINE

```python
# GRPO TRAINING FOR VISION-LANGUAGE MODELS

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import LoraConfig, get_peft_model
from typing import List, Dict

class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer.

    Based on DeepSeek-R1's training approach:
    - No critic/value model
    - Group-based advantage estimation
    - KL penalty to reference model
    """

    def __init__(
        self,
        model_name: str,
        dpo_checkpoint: str,  # Start from DPO model
        group_size: int = 8,  # Number of responses per prompt
        beta: float = 0.1,    # KL penalty
        learning_rate: float = 1e-6,
    ):
        # Policy model (trainable)
        self.policy = AutoModelForCausalLM.from_pretrained(
            dpo_checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # Reference model (frozen)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            dpo_checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # Add LoRA for memory efficiency
        lora_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.policy = get_peft_model(self.policy, lora_config)

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.group_size = group_size
        self.beta = beta

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )

    def generate_group(
        self,
        prompt: str,
        image_path: str,
        temperature: float = 0.7,
    ) -> List[str]:
        """Generate N responses for a single prompt."""

        inputs = self.processor(
            text=prompt,
            images=[image_path],
            return_tensors="pt",
        ).to(self.policy.device)

        responses = []
        for _ in range(self.group_size):
            with torch.no_grad():
                outputs = self.policy.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                )

            response = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            responses.append(response)

        return responses

    def compute_rewards(
        self,
        prompt: str,
        responses: List[str],
        environment,  # Your Higgsfield environment
    ) -> List[float]:
        """Execute responses in environment and get rewards."""

        rewards = []
        for response in responses:
            # Parse action from response
            action = self.parse_action(response)

            if action is None:
                rewards.append(-0.2)  # No valid action
                continue

            # Execute in environment
            prev_state = environment.get_state()
            result = environment.execute(action)
            curr_state = environment.get_state()

            # Compute reward
            reward = self.reward_function(
                action, prev_state, curr_state, result
            )
            rewards.append(reward)

            # Reset environment for next response
            environment.reset_to_state(prev_state)

        return rewards

    def compute_advantages(self, rewards: List[float]) -> List[float]:
        """Compute group-relative advantages."""
        mean_reward = sum(rewards) / len(rewards)
        std_reward = (sum((r - mean_reward)**2 for r in rewards) / len(rewards)) ** 0.5

        # Normalize advantages
        if std_reward > 0:
            advantages = [(r - mean_reward) / std_reward for r in rewards]
        else:
            advantages = [0.0] * len(rewards)

        return advantages

    def compute_log_probs(
        self,
        model,
        prompt: str,
        response: str,
        image_path: str,
    ) -> torch.Tensor:
        """Compute log probability of response given prompt."""

        full_text = prompt + response
        inputs = self.processor(
            text=full_text,
            images=[image_path],
            return_tensors="pt",
        ).to(model.device)

        prompt_inputs = self.processor(
            text=prompt,
            images=[image_path],
            return_tensors="pt",
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        with torch.set_grad_enabled(model.training):
            outputs = model(**inputs, labels=inputs["input_ids"])

        # Sum log probs over response tokens only
        logits = outputs.logits[:, prompt_len-1:-1, :]
        labels = inputs["input_ids"][:, prompt_len:]

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        return token_log_probs.sum()

    def train_step(
        self,
        prompt: str,
        image_path: str,
        environment,
    ) -> Dict[str, float]:
        """One GRPO training step."""

        # 1. Generate group of responses
        responses = self.generate_group(prompt, image_path)

        # 2. Get rewards from environment
        rewards = self.compute_rewards(prompt, responses, environment)

        # 3. Compute advantages
        advantages = self.compute_advantages(rewards)

        # 4. Compute loss
        total_loss = 0
        total_kl = 0

        for response, advantage in zip(responses, advantages):
            # Policy log prob
            policy_log_prob = self.compute_log_probs(
                self.policy, prompt, response, image_path
            )

            # Reference log prob (for KL)
            with torch.no_grad():
                ref_log_prob = self.compute_log_probs(
                    self.ref_model, prompt, response, image_path
                )

            # KL divergence
            kl = policy_log_prob - ref_log_prob

            # GRPO loss: -advantage * log_prob + beta * KL
            loss = -advantage * policy_log_prob + self.beta * kl

            total_loss += loss
            total_kl += kl.item()

        # Average over group
        total_loss = total_loss / self.group_size

        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "mean_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "kl": total_kl / self.group_size,
        }

    def train(
        self,
        prompts: List[str],
        images: List[str],
        environment,
        num_epochs: int = 1,
    ):
        """Full training loop."""

        for epoch in range(num_epochs):
            for prompt, image in zip(prompts, images):
                metrics = self.train_step(prompt, image, environment)
                print(f"Loss: {metrics['loss']:.4f}, "
                      f"Reward: {metrics['mean_reward']:.4f}, "
                      f"KL: {metrics['kl']:.4f}")


# ============================================================================
# ENVIRONMENT WRAPPER FOR HIGGSFIELD
# ============================================================================

class HiggsfieldEnvironment:
    """Environment wrapper for Higgsfield automation."""

    def __init__(self, playwright_browser):
        self.browser = playwright_browser
        self.page = self.browser.new_page()
        self.page.goto("https://higgsfield.ai")

    def get_state(self):
        """Capture current state (screenshot + HTML)."""
        screenshot = self.page.screenshot()
        html = self.page.content()
        url = self.page.url
        return {"screenshot": screenshot, "html": html, "url": url}

    def execute(self, action):
        """Execute an action."""
        if action["type"] == "click":
            self.page.mouse.click(action["x"], action["y"])
        elif action["type"] == "type":
            self.page.keyboard.type(action["text"])
        elif action["type"] == "scroll":
            self.page.mouse.wheel(0, action["amount"])

        self.page.wait_for_timeout(500)  # Wait for UI to settle

        return {"success": True}

    def reset_to_state(self, state):
        """Reset to a previous state (for group evaluation)."""
        # Refresh and navigate back
        self.page.goto(state["url"])
```

---

## 5.5 ONLINE RL SHORTCOMINGS

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ONLINE RL LIMITATIONS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. ENVIRONMENT COST                                                        │
│     ────────────────                                                        │
│     Problem: Need actual environment for training                          │
│                                                                             │
│     For Higgsfield:                                                         │
│     - Must run real browser                                                 │
│     - Each action takes 500ms+ of real time                                │
│     - Need API access / credits                                            │
│     - Can't parallelize easily                                             │
│                                                                             │
│     1000 training steps × 10 actions × 0.5s = 5000s = 1.4 hours            │
│     Just for interaction, not counting compute!                            │
│                                                                             │
│     Mitigation: Environment simulation, parallel browsers                  │
│                                                                             │
│  2. REWARD HACKING                                                          │
│     ──────────────                                                          │
│     Problem: Model finds loopholes in reward function                      │
│                                                                             │
│     Example: If reward = +0.1 for "ACTION:" in response                   │
│     Model learns: "ACTION: ACTION: ACTION: ACTION:" (maximizes reward!)   │
│                                                                             │
│     Mitigation: Careful reward design, multiple reward components          │
│                                                                             │
│  3. SAMPLE INEFFICIENCY                                                     │
│     ────────────────────                                                    │
│     Problem: Need many episodes to learn                                   │
│                                                                             │
│     DPO: 50 examples = good model                                          │
│     GRPO: 5000+ environment steps = comparable model                       │
│                                                                             │
│     Mitigation: Start from DPO model, use experience replay                │
│                                                                             │
│  4. EXPLORATION-EXPLOITATION TRADEOFF                                       │
│     ────────────────────────────────                                        │
│     Problem: How to balance trying new things vs using what works          │
│                                                                             │
│     Too much exploration: Wastes time on bad actions                       │
│     Too little exploration: Gets stuck in local optima                     │
│                                                                             │
│     Mitigation: Temperature scheduling, curiosity bonuses                  │
│                                                                             │
│  5. NON-STATIONARITY                                                        │
│     ────────────────                                                        │
│     Problem: Environment can change                                        │
│                                                                             │
│     If Higgsfield updates UI, model's learned actions become wrong        │
│                                                                             │
│     Mitigation: Continuous training, robust representations                │
│                                                                             │
│  6. CREDIT ASSIGNMENT                                                       │
│     ─────────────────                                                       │
│     Problem: Which action caused the reward?                               │
│                                                                             │
│     Task takes 10 steps, success reward at end                             │
│     Was step 3 important or step 7?                                        │
│                                                                             │
│     Mitigation: Dense rewards, temporal difference learning                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5.6 WHEN TO USE ONLINE RL

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DECISION: SHOULD YOU USE ONLINE RL?                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  USE ONLINE RL (GRPO) WHEN:                                                 │
│                                                                             │
│  ✓ DPO has plateaued (can't improve further)                               │
│  ✓ You have environment access (can run Higgsfield)                        │
│  ✓ You have compute budget (8+ A100s for reasonable time)                  │
│  ✓ You can design good reward function                                     │
│  ✓ Task requires exploration (many possible valid paths)                   │
│  ✓ You want maximum performance, cost is secondary                         │
│                                                                             │
│  SKIP ONLINE RL WHEN:                                                       │
│                                                                             │
│  ✗ DPO gives acceptable performance                                        │
│  ✗ Environment is expensive/slow to run                                    │
│  ✗ Time-constrained (need model in days, not weeks)                        │
│  ✗ Reward function is hard to define                                       │
│  ✗ Task is simple/repetitive (retrieval + SFT + DPO is enough)            │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FOR HIGGSFIELD AUTOMATION:                                                 │
│                                                                             │
│  RECOMMENDATION: SKIP ONLINE RL (for now)                                  │
│                                                                             │
│  Why:                                                                       │
│  1. Retrieval (Embedding + Reranker) already provides strong grounding    │
│  2. SFT + DPO achieves 80%+ task success                                   │
│  3. Higgsfield UI is relatively fixed → exploration not critical          │
│  4. Environment cost is high (real browser, API credits)                   │
│                                                                             │
│  Revisit online RL if:                                                      │
│  - DPO model makes systematic errors that can't be fixed with data        │
│  - Need to handle highly dynamic scenarios                                 │
│  - Have resources for weeks of training                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5.7 TRAINING REQUIREMENTS

| Model | GRPO Group Size | VRAM | Environment Cost | Time |
|-------|-----------------|------|------------------|------|
| **3B** | 8 | 32GB | Low | 1-2 days |
| **A3B** | 4 | 120GB | Medium | 3-5 days |
| **32B** | 2 | 130GB | High | 1-2 weeks |

---

## NEXT: Part 6 - Evaluation & Benchmarks

Continue to [PART6_EVALUATION.md](./PART6_EVALUATION.md)
