# How to Handle UI Changes: Making Your Agent Bulletproof

## THE PROBLEM

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WHY UI CHANGES BREAK AGENTS                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Your agent today:                                                          │
│  "Click Generate button at (640, 600)"                                      │
│                                                                             │
│  Higgsfield updates their UI next week:                                     │
│  - Button moved to (720, 550)                                               │
│  - Button text changed to "Create Video"                                    │
│  - Button color changed from purple to blue                                 │
│  - New popup added before button                                            │
│                                                                             │
│  Your agent:                                                                │
│  *clicks wrong location*                                                   │
│  *misses button*                                                            │
│  *clicks on new popup instead*                                              │
│                                                                             │
│  RESULT: Complete failure. All training data now useless.                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## THE SOLUTION: MULTI-LAYER ROBUSTNESS

You need MULTIPLE strategies working together. No single approach is enough.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ROBUSTNESS STACK                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Layer 5: SELF-HEALING (runtime recovery)                                  │
│      ↑                                                                      │
│  Layer 4: SEMANTIC UNDERSTANDING (intent, not pixels)                      │
│      ↑                                                                      │
│  Layer 3: SET-OF-MARK PROMPTING (numbered regions)                         │
│      ↑                                                                      │
│  Layer 2: MULTI-ATTRIBUTE MATCHING (not just coordinates)                  │
│      ↑                                                                      │
│  Layer 1: DATA DIVERSITY (variations in training)                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## LAYER 1: DATA DIVERSITY (During Curation)

**This is what YOU do as data curator.**

### Strategy 1.1: Multiple UI States

Don't just capture one screenshot of each page. Capture MANY variations:

```
For each page, capture:
├── Default state (just loaded)
├── Hover states (each interactive element)
├── Click states (each interactive element)
├── Focused states (form fields)
├── Error states (validation errors)
├── Loading states (spinners)
├── Different window sizes (if responsive)
├── Light mode
├── Dark mode (if available)
├── With popups/modals
├── With tooltips visible
└── After scrolling (different scroll positions)
```

### Strategy 1.2: Paraphrase Everything

Each action needs 10+ ways to describe it:

```
Original: "Click the Generate button"

Paraphrases:
- "Click Generate"
- "Press the Generate button"
- "Hit Generate"
- "Start generation"
- "Create the video"
- "Make the video"
- "Generate video"
- "Click the purple Generate button"
- "Click the button that says Generate"
- "Press the Create Video button" (anticipate rename!)
- "Click the main action button"
- "Start the video creation process"
```

### Strategy 1.3: Include Position-Independent Descriptions

Don't just use coordinates. Describe WHAT the element is:

**BAD (fragile):**
```json
{
  "text": "Purple button at coordinates (640, 600)"
}
```

**GOOD (robust):**
```json
{
  "text": "Generate button - main call-to-action button at bottom of video workspace, purple/blue color, contains text 'Generate' or 'Create Video', located below the prompt input field"
}
```

### Strategy 1.4: Describe Spatial Relationships

```json
{
  "text": "Generate button, located below the prompt text area, to the right of the duration slider, at the bottom of the workspace panel"
}
```

### Strategy 1.5: Anticipate Changes

Include descriptions that would still work if things move:

```json
{
  "text": "The primary action button for starting video generation, typically the most prominent button in the workspace, may be labeled 'Generate', 'Create', or 'Create Video'"
}
```

---

## LAYER 2: MULTI-ATTRIBUTE MATCHING (Architecture Change)

Instead of training on coordinates alone, train on MULTIPLE attributes:

### Current (Fragile) Approach:
```
Screenshot → Model → Coordinates (640, 600)
```

### Robust Approach:
```
Screenshot → Model → {
  "element_type": "button",
  "text_content": "Generate",
  "visual_description": "purple/blue prominent CTA",
  "spatial_context": "bottom of workspace, below prompt",
  "aria_role": "button",
  "estimated_coordinates": (640, 600),
  "confidence": 0.95
}
```

### Update Your SFT Data Format:

**OLD FORMAT (fragile):**
```json
{
  "action": "click",
  "x": 640,
  "y": 600,
  "reasoning": "Clicking Generate button"
}
```

**NEW FORMAT (robust):**
```json
{
  "action": "click",
  "target": {
    "type": "button",
    "text": ["Generate", "Create Video", "Create"],
    "visual": "prominent CTA button, purple or blue",
    "location": "bottom of workspace panel",
    "near": ["prompt input field", "duration slider"],
    "role": "primary action"
  },
  "fallback_coordinates": {"x": 640, "y": 600},
  "reasoning": "Clicking the main action button to start video generation"
}
```

### Runtime: Verify Before Click

```python
def robust_click(target_description, fallback_coords):
    """
    1. First, try to find element by semantic description
    2. If found, click it
    3. If not found, try visual search
    4. If still not found, use fallback coordinates with low confidence
    """

    # Try semantic search first
    element = find_by_text(target_description["text"])
    if element:
        click(element.center)
        return

    # Try visual search
    element = find_by_visual(target_description["visual"])
    if element:
        click(element.center)
        return

    # Try spatial relationship
    element = find_near(target_description["near"], target_description["location"])
    if element:
        click(element.center)
        return

    # Last resort: fallback coordinates (log warning!)
    log.warning(f"Using fallback coordinates - UI may have changed!")
    click(fallback_coords["x"], fallback_coords["y"])
```

---

## LAYER 3: SET-OF-MARK (SoM) PROMPTING

This is a **game-changer** from Microsoft Research ([Paper](https://arxiv.org/abs/2310.11441), [GitHub](https://github.com/microsoft/SoM)).

### How It Works:

Instead of asking the model "where should I click?", you:

1. Detect all UI elements in the screenshot
2. Draw numbered boxes around each element
3. Ask the model "which number should I click?"

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SET-OF-MARK EXAMPLE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ORIGINAL SCREENSHOT:                                                       │
│  ┌────────────────────────────────────┐                                    │
│  │  [Create ▼]  My Videos  (40 credits)                                    │
│  │                                     │                                    │
│  │  Model: [WAN 2.6 ▼]                │                                    │
│  │                                     │                                    │
│  │  ┌────────────────────┐            │                                    │
│  │  │    Upload Image    │            │                                    │
│  │  └────────────────────┘            │                                    │
│  │                                     │                                    │
│  │  Prompt: [                    ]    │                                    │
│  │                                     │                                    │
│  │  ┌────────────────────┐            │                                    │
│  │  │     Generate       │            │                                    │
│  │  └────────────────────┘            │                                    │
│  └────────────────────────────────────┘                                    │
│                                                                             │
│  WITH SET-OF-MARK:                                                          │
│  ┌────────────────────────────────────┐                                    │
│  │  [1]  [2]  [3]                     │  ← Numbered regions                │
│  │                                     │                                    │
│  │  Model: [4]                        │                                    │
│  │                                     │                                    │
│  │  ┌────────────────────┐            │                                    │
│  │  │    [5]             │            │                                    │
│  │  └────────────────────┘            │                                    │
│  │                                     │                                    │
│  │  Prompt: [6]                       │                                    │
│  │                                     │                                    │
│  │  ┌────────────────────┐            │                                    │
│  │  │     [7]            │            │                                    │
│  │  └────────────────────┘            │                                    │
│  └────────────────────────────────────┘                                    │
│                                                                             │
│  PROMPT TO MODEL:                                                           │
│  "The image shows a video creation interface with numbered regions.        │
│   User wants to: Generate the video                                        │
│   Which numbered region should be clicked?"                                │
│                                                                             │
│  MODEL RESPONSE:                                                            │
│  "Click region [7] - the Generate button"                                  │
│                                                                             │
│  WHY THIS IS ROBUST:                                                        │
│  - Even if button moves, it gets a NEW number                              │
│  - Model identifies by MEANING not position                                │
│  - Works even if UI completely redesigned                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation:

```python
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def apply_set_of_mark(screenshot_path):
    """
    Apply Set-of-Mark prompting to a screenshot.
    Returns: marked_image, region_info
    """

    # Load image
    image = cv2.imread(screenshot_path)

    # Use SAM to segment UI elements
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)

    # Or use a UI-specific detector (better for GUIs)
    # detector = load_ui_detector("omniparser")  # or similar
    # elements = detector.detect(image)

    # Number each region
    marked_image = image.copy()
    region_info = {}

    for i, mask in enumerate(masks):
        # Get bounding box
        bbox = mask["bbox"]  # x, y, w, h
        center_x = bbox[0] + bbox[2] // 2
        center_y = bbox[1] + bbox[3] // 2

        # Draw number on image
        cv2.rectangle(marked_image,
                      (bbox[0], bbox[1]),
                      (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                      (0, 255, 0), 2)
        cv2.putText(marked_image, str(i+1),
                    (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Store region info
        region_info[i+1] = {
            "bbox": bbox,
            "center": (center_x, center_y),
            "area": mask["area"]
        }

    return marked_image, region_info


def get_click_from_som(marked_image, user_instruction, region_info):
    """
    Use VLM to identify which region to click.
    """

    prompt = f"""This image shows a user interface with numbered regions marked in green boxes.

User instruction: {user_instruction}

Which numbered region should be clicked to accomplish this task?
Respond with just the region number."""

    # Call your VLM (Qwen3-VL, GPT-4V, etc.)
    response = vlm.generate(marked_image, prompt)

    # Parse response
    region_num = int(response.strip())

    # Get coordinates from region_info
    coords = region_info[region_num]["center"]

    return coords
```

### Training Data for SoM:

Update your SFT data to include SoM-style examples:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a UI assistant. Given a screenshot with numbered regions and a user instruction, identify which region to interact with."
    },
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "screenshots/workspace_som_marked.png"},
        {"type": "text", "text": "User wants to: Generate the video\nWhich numbered region should be clicked?"}
      ]
    },
    {
      "role": "assistant",
      "content": "Region 7 - the Generate button at the bottom of the workspace"
    }
  ]
}
```

---

## LAYER 4: SEMANTIC UNDERSTANDING

Train the model to understand INTENT, not just match patterns.

### Include "Why" in Your Data:

**BAD:**
```json
{
  "instruction": "Click Generate",
  "action": "click",
  "x": 640,
  "y": 600
}
```

**GOOD:**
```json
{
  "instruction": "Click Generate",
  "intent": "Start the video creation process after all inputs are configured",
  "preconditions": ["image uploaded", "prompt entered", "model selected"],
  "expected_result": "Video generation begins, progress indicator appears",
  "action": "click",
  "target": "primary action button for video generation",
  "x": 640,
  "y": 600
}
```

### Train on Edge Cases:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": [
      {"type": "image", "image": "screenshots/workspace_generate_disabled.png"},
      {"type": "text", "text": "Generate the video"}
    ]},
    {"role": "assistant", "content": "{\"action\": \"none\", \"reasoning\": \"Generate button is disabled because no image has been uploaded yet. First upload an image.\", \"suggestion\": \"Click the Upload Image button first\"}"}
  ]
}
```

### Train on UI Variations:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": [
      {"type": "image", "image": "screenshots/workspace_new_ui_v2.png"},
      {"type": "text", "text": "Generate the video"}
    ]},
    {"role": "assistant", "content": "{\"action\": \"click\", \"reasoning\": \"The Generate button has moved to a new location in this UI version. It is now labeled 'Create Video' and located in the top right instead of bottom center. Clicking the primary action button.\", \"x\": 850, \"y\": 120}"}
  ]
}
```

---

## LAYER 5: SELF-HEALING (Runtime)

Based on [UiPath Healing Agent](https://www.uipath.com/blog/product-and-updates/technical-tuesday-how-healing-agent-solves-ui-automation-challenges) and research.

### Strategy 5.1: Verify After Click

```python
def verified_click(target, expected_result):
    """
    Click and verify the result.
    If verification fails, try to recover.
    """

    # Take screenshot before
    before = take_screenshot()

    # Click
    click(target)

    # Wait and take screenshot after
    time.sleep(0.5)
    after = take_screenshot()

    # Verify expected result
    if not verify_result(after, expected_result):
        # Something went wrong - try to recover
        return self_heal(before, after, target, expected_result)

    return True


def self_heal(before, after, target, expected_result):
    """
    Try to recover from a failed action.
    """

    # Check what went wrong
    analysis = analyze_failure(before, after, target)

    if analysis["type"] == "element_moved":
        # Element moved - find new location
        new_target = find_similar_element(after, target["description"])
        if new_target:
            log.info(f"Element moved from {target} to {new_target}")
            return verified_click(new_target, expected_result)

    elif analysis["type"] == "popup_appeared":
        # Popup blocking - dismiss it first
        dismiss_popup(after)
        return verified_click(target, expected_result)

    elif analysis["type"] == "element_renamed":
        # Element has new text - find by other attributes
        new_target = find_by_role_and_position(after, target)
        if new_target:
            return verified_click(new_target, expected_result)

    # Could not recover
    log.error("Self-healing failed")
    return False
```

### Strategy 5.2: Multiple Locator Fallbacks

```python
def find_element(target):
    """
    Try multiple strategies to find an element.
    """

    strategies = [
        # 1. Exact text match
        lambda: find_by_text(target["text"]),

        # 2. Partial text match
        lambda: find_by_partial_text(target["text"]),

        # 3. Similar text (semantic)
        lambda: find_by_semantic_similarity(target["text"]),

        # 4. Visual similarity
        lambda: find_by_visual_template(target["visual_template"]),

        # 5. Spatial relationship
        lambda: find_by_spatial_context(target["near"], target["position"]),

        # 6. Role/type
        lambda: find_by_role(target["role"], target["type"]),

        # 7. Fallback coordinates (last resort)
        lambda: {"coords": target["fallback_coords"], "confidence": 0.3}
    ]

    for strategy in strategies:
        result = strategy()
        if result and result.get("confidence", 0) > 0.7:
            return result

    # Return best low-confidence match
    for strategy in strategies:
        result = strategy()
        if result:
            return result

    return None
```

### Strategy 5.3: Semantic Change Detection

From [UiPath](https://docs.uipath.com/agents/automation-cloud/latest/user-guide-ha/what-is-healing-agent):

```python
def detect_semantic_change(old_element, new_candidates):
    """
    Detect when UI elements have changed in meaning.
    E.g., "First Name" → "Given Name" (same meaning)
    E.g., "Submit" → "Cancel" (different meaning - don't match!)
    """

    for candidate in new_candidates:
        # Check semantic similarity
        similarity = semantic_similarity(
            old_element["text"],
            candidate["text"]
        )

        if similarity > 0.8:
            # High semantic similarity - probably same element
            return candidate

        # Also check functional similarity
        if old_element["role"] == candidate["role"]:
            if old_element["position_relative"] == candidate["position_relative"]:
                # Same role, same relative position - likely same element
                return candidate

    return None
```

---

## LAYER 6: CONTINUOUS IMPROVEMENT

### Strategy 6.1: Log Everything

```python
@dataclass
class ActionLog:
    timestamp: datetime
    screenshot_before: str
    screenshot_after: str
    instruction: str
    target: dict
    action_taken: dict
    success: bool
    recovery_attempted: bool
    recovery_success: bool
    notes: str
```

### Strategy 6.2: Automatic Retraining Triggers

```python
def should_retrain(logs: List[ActionLog]) -> bool:
    """
    Detect when retraining is needed.
    """

    recent_logs = logs[-100:]  # Last 100 actions

    # Calculate failure rate
    failures = sum(1 for log in recent_logs if not log.success)
    failure_rate = failures / len(recent_logs)

    # Calculate recovery rate
    recoveries = sum(1 for log in recent_logs if log.recovery_attempted)
    recovery_rate = recoveries / len(recent_logs)

    # Trigger retraining if:
    # - Failure rate > 10%
    # - Recovery rate > 20% (means UI is changing frequently)
    if failure_rate > 0.10 or recovery_rate > 0.20:
        notify_team("Retraining recommended - UI may have changed significantly")
        return True

    return False
```

### Strategy 6.3: Automatic Data Collection

```python
def collect_training_data_on_recovery(log: ActionLog):
    """
    When self-healing succeeds, automatically create training data.
    """

    if log.recovery_attempted and log.recovery_success:
        # This is a NEW example of how to handle this UI state

        new_example = {
            "query": {
                "image": log.screenshot_before,
                "text": log.instruction
            },
            "positive": {
                "image": log.screenshot_after,
                "text": f"Element found at new location after UI change: {log.action_taken}"
            },
            "negatives": [
                {
                    "image": log.screenshot_before,
                    "text": f"Old location no longer valid: {log.target}",
                    "negative_type": "outdated_ui"
                }
            ]
        }

        # Save for review and inclusion in next training round
        save_candidate_example(new_example)
```

---

## PRACTICAL CHECKLIST FOR DATA CURATOR

### When Taking Screenshots:

```
□ Capture multiple UI states (default, hover, clicked, disabled)
□ Capture light mode AND dark mode if available
□ Capture at multiple window sizes (if responsive)
□ Capture with different amounts of content (empty, partial, full)
□ Capture error states and popups
□ Check for A/B tests (is the UI different for different users?)
□ Re-capture screenshots periodically (weekly/monthly)
```

### When Writing Text Descriptions:

```
□ Include element type (button, input, dropdown, link)
□ Include visible text on element
□ Include visual description (color, size, prominence)
□ Include spatial relationships (above X, below Y, next to Z)
□ Include functional role (primary action, cancel, navigation)
□ Use multiple phrasings (10+ per action)
□ Anticipate text changes (Generate → Create → Make)
```

### When Creating Training Data:

```
□ Include position-independent descriptions
□ Include "why" reasoning for each action
□ Include preconditions and expected results
□ Include edge cases (disabled states, errors, popups)
□ Include variations of same action with different wording
□ Balance between specific coordinates and semantic descriptions
□ Create Set-of-Mark examples for critical workflows
```

---

## SUMMARY

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     UI ROBUSTNESS SUMMARY                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DURING DATA CURATION (Your job):                                          │
│  ═══════════════════════════════                                           │
│  1. Capture MANY UI variations (states, modes, sizes)                      │
│  2. Use SEMANTIC descriptions (not just coordinates)                       │
│  3. Include SPATIAL relationships ("below prompt", "next to upload")       │
│  4. Paraphrase EVERYTHING (10+ ways per action)                            │
│  5. Anticipate CHANGES (alternative button texts)                          │
│  6. Include EDGE CASES (disabled, errors, popups)                          │
│                                                                             │
│  DURING ARCHITECTURE (Engineering):                                        │
│  ══════════════════════════════════                                        │
│  1. Output MULTI-ATTRIBUTE targets (not just x,y)                          │
│  2. Implement SET-OF-MARK for robust grounding                             │
│  3. Use MULTIPLE LOCATOR STRATEGIES with fallbacks                         │
│  4. Add VERIFICATION after each action                                     │
│  5. Implement SELF-HEALING for runtime recovery                            │
│                                                                             │
│  DURING OPERATIONS:                                                        │
│  ═══════════════════                                                       │
│  1. LOG everything (before/after screenshots)                              │
│  2. MONITOR failure and recovery rates                                     │
│  3. TRIGGER retraining when UI changes detected                            │
│  4. AUTOMATICALLY COLLECT new training data from recoveries                │
│  5. RE-CAPTURE screenshots periodically                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## REFERENCES

- [Set-of-Mark Prompting (Microsoft)](https://arxiv.org/abs/2310.11441)
- [SoM GitHub](https://github.com/microsoft/SoM)
- [UiPath Healing Agent](https://www.uipath.com/blog/product-and-updates/technical-tuesday-how-healing-agent-solves-ui-automation-challenges)
- [ShowUI: Vision-Language-Action Model (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_ShowUI_One_Vision-Language-Action_Model_for_GUI_Visual_Agent_CVPR_2025_paper.pdf)
- [R-VLM: Region-Aware VLM for GUI Grounding](https://arxiv.org/abs/2507.05673)
- [ScreenAI (Google)](https://research.google/blog/screenai-a-visual-language-model-for-ui-and-visually-situated-language-understanding/)
- [Playwright Semantic Locators](https://www.browserstack.com/guide/playwright-locator)
- [Self-Healing Test Automation (Functionize)](https://www.functionize.com/automated-testing/self-healing-test-automation)
