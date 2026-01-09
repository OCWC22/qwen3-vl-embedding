# COMPLETE INTERN GUIDE: Creating Dataset for 3B Model
## FOR SOMEONE WHO KNOWS ABSOLUTELY NOTHING

**Read this entire document before doing anything.**

---

# PART 1: WHAT THE FUCK ARE WE DOING AND WHY

## 1.1 The Problem We're Solving

You're building a robot that can use a computer. Specifically, it will:
1. Look at a screenshot of Higgsfield AI (a video generation app)
2. Decide what to click/type
3. Do it
4. Repeat until the task is done

**The problem:** The AI model doesn't know what Higgsfield looks like. It's never seen it. It doesn't know where buttons are. It doesn't know what "Generate Video" looks like. It's like asking someone who's never seen a computer to use Photoshop.

**The solution:** We show it examples. "Here's a screenshot. Here's what you should click." We do this hundreds of times until it learns.

## 1.2 Why We Need TWO Types of Models

We're training TWO different models that do TWO different jobs:

### MODEL 1: EMBEDDING MODEL (The Librarian)

**What it does:**
- Takes a screenshot and says "I've seen something like this before"
- Finds similar past situations from your training data
- Returns the top 100 most similar situations

**Analogy:** You walk into a library and say "I need a book about cooking pasta." The librarian doesn't read every book - they know the cooking section is in aisle 3. They quickly narrow down to 100 relevant books.

**Why we need it:**
- It's FAST (10 milliseconds)
- It can search through 10,000 examples quickly
- But it's not super accurate - it gives you 100 options, not 1

### MODEL 2: RERANKER MODEL (The Expert)

**What it does:**
- Takes the 100 options from the embedding model
- Carefully reads each one
- Picks the BEST 5 options

**Analogy:** The librarian gave you 100 cooking books. Now an expert chef looks at each one and says "These 5 are actually about pasta. The others are about pizza, salads, etc."

**Why we need it:**
- It's ACCURATE (much better than embedding)
- But it's SLOW (500 milliseconds for 100 items)
- That's why we need embedding first - to narrow down from 10,000 to 100

### THE PIPELINE

```
User says: "Generate a video of a dancing cat"
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Take screenshot of current Higgsfield screen        │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: EMBEDDING MODEL looks at screenshot                 │
│         "I've seen 100 similar situations before"           │
│         Returns: 100 past examples that look similar        │
│         Time: 10ms                                          │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: RERANKER MODEL examines those 100                   │
│         "Actually, these 5 are the most relevant"           │
│         Returns: Top 5 best matches                         │
│         Time: 500ms                                         │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: GENERATION MODEL (the 3B model) uses top 5          │
│         "Based on these examples, I should click here"      │
│         Outputs: Click coordinates or text to type          │
│         Time: 200ms                                         │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: Actually click/type on the screen                   │
│         Go back to STEP 1                                   │
└─────────────────────────────────────────────────────────────┘
```

## 1.3 What is a 3B Model?

"3B" means 3 billion parameters. Parameters are the "brain cells" of the AI.

- **3B model** = Small brain, fast, needs less data, runs on cheaper GPUs
- **8B model** = Medium brain, moderate speed, needs more data
- **70B model** = Huge brain, slow, needs lots of data, expensive GPUs

**For Higgsfield automation, 3B is enough because:**
- The task is repetitive (same UI, same buttons)
- We're using retrieval (embedding + reranker) to help it
- Speed matters more than raw intelligence

**Exact model we're using:** `Qwen/Qwen3-VL-3B-Instruct` (or MoE variant)

---

# PART 2: EXACT DATA REQUIREMENTS FOR 3B MODEL

## 2.1 Summary Table (MEMORIZE THIS)

| Dataset | Minimum | Recommended | Format | File |
|---------|---------|-------------|--------|------|
| Embedding | **250 triplets** | 500 triplets | JSONL | `embedding_train.jsonl` |
| Reranker | **150 pairs** | 300 pairs | JSONL | `reranker_train.jsonl` |
| SFT | **50 conversations** | 100 conversations | JSONL | `sft_train.jsonl` |
| DPO (optional) | **25 pairs** | 50 pairs | JSONL | `dpo_train.jsonl` |

**Total minimum: 475 data points**
**Total recommended: 950 data points**

## 2.2 What is a "Triplet"? (Embedding Data)

A triplet has THREE parts:

```
TRIPLET = {
    "query": "What to search for",
    "positive": "A correct match",
    "negative": "A wrong match (but looks similar)"
}
```

**Real example for Higgsfield:**

```json
{
    "query": "I see a login screen with email and password fields. I need to log in.",
    "positive": {
        "image": "screenshots/login_screen_01.png",
        "text": "Click the email field, type email, click password field, type password, click Login button"
    },
    "negative": {
        "image": "screenshots/signup_screen_01.png",
        "text": "Click Create Account button to make a new account"
    }
}
```

**Why do we need negatives?**
The model needs to learn what's WRONG, not just what's right. The negative should be:
- Similar looking (to make it hard)
- But actually wrong for this situation

The login screen and signup screen look similar (both have email/password fields), but the actions are different. This teaches the model to distinguish them.

## 2.3 What is a "Pair"? (Reranker Data)

A pair has TWO parts plus a label:

```
PAIR = {
    "query": "Screenshot + what user wants to do",
    "document": "A possible action",
    "label": 1 (correct) or 0 (wrong)
}
```

**Real example for Higgsfield:**

```json
{
    "query_image": "screenshots/model_selection_01.png",
    "query_text": "I need to select the Animate model for my video",
    "document": "Click on the 'Animate' option in the model dropdown menu",
    "label": 1
}
```

```json
{
    "query_image": "screenshots/model_selection_01.png",
    "query_text": "I need to select the Animate model for my video",
    "document": "Click on the 'Settings' gear icon in the top right",
    "label": 0
}
```

**CRITICAL: You need BALANCED labels**
- 50% of pairs should have label=1 (correct)
- 50% of pairs should have label=0 (wrong)

If you have 150 pairs: 75 correct, 75 wrong.

## 2.4 What is a "Conversation"? (SFT Data)

SFT = Supervised Fine-Tuning. This is for the main 3B generation model.

A conversation shows the model what to output:

```json
{
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "screenshots/upload_screen_01.png"},
                {"type": "text", "text": "I want to upload my image and generate a video. What should I do?"}
            ]
        },
        {
            "role": "assistant",
            "content": "I can see the upload screen. To upload your image:\n\n1. Click the 'Upload Image' button in the center of the screen (coordinates: 512, 384)\n2. This will open a file picker dialog\n\nAction: CLICK(512, 384)"
        }
    ]
}
```

## 2.5 What is a "Preference Pair"? (DPO Data)

DPO = Direct Preference Optimization. This teaches the model which response is BETTER.

```json
{
    "prompt": [
        {"type": "image", "image": "screenshots/generate_screen_01.png"},
        {"type": "text", "text": "Generate a video now"}
    ],
    "chosen": "Click the green 'Generate' button at coordinates (640, 500). This will start video generation with your current settings.",
    "rejected": "I can help you generate a video. Would you like me to explain the options first?"
}
```

- **chosen** = The GOOD response (direct, actionable)
- **rejected** = The BAD response (vague, unhelpful)

---

# PART 3: SCREENSHOT SPECIFICATIONS

## 3.1 Technical Requirements

| Property | Requirement | Why |
|----------|-------------|-----|
| Resolution | **1024 x 768** pixels | Model trained on this size |
| Format | **PNG** | Lossless, preserves UI text |
| Color | **RGB** (not RGBA) | Standard format |
| File size | **Under 2MB** | Memory limits |
| DPI | **72 or 96** | Screen standard |

## 3.2 How to Take Screenshots

### On Windows:
```
1. Open Higgsfield in your browser
2. Press Win + Shift + S
3. Select the browser window (just the content, not the browser chrome)
4. Save as PNG
```

### On Mac:
```
1. Open Higgsfield in your browser
2. Press Cmd + Shift + 4
3. Press Space to capture window
4. Click on browser window
5. File saves to Desktop as PNG
```

### Using Python (PREFERRED - consistent results):
```python
from playwright import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1024, "height": 768})
    page.goto("https://higgsfield.ai")
    page.screenshot(path="screenshot_001.png")
    browser.close()
```

## 3.3 What Screenshots to Take

You need screenshots of EVERY screen in Higgsfield. Here's the complete list:

### CATEGORY 1: Authentication (10 screenshots minimum)

| Screenshot | Filename | What to capture |
|------------|----------|-----------------|
| Login page (empty) | `auth_login_empty_01.png` | Email/password fields empty |
| Login page (email filled) | `auth_login_email_01.png` | Email typed, password empty |
| Login page (both filled) | `auth_login_filled_01.png` | Both fields have text |
| Login page (error) | `auth_login_error_01.png` | "Invalid credentials" shown |
| Signup page (empty) | `auth_signup_empty_01.png` | Registration form empty |
| Signup page (filled) | `auth_signup_filled_01.png` | Form filled out |
| Forgot password | `auth_forgot_01.png` | Password reset screen |
| Email verification | `auth_verify_01.png` | "Check your email" screen |
| OAuth options | `auth_oauth_01.png` | Google/Apple sign-in buttons |
| Loading state | `auth_loading_01.png` | Spinner while authenticating |

### CATEGORY 2: Dashboard/Home (15 screenshots minimum)

| Screenshot | Filename | What to capture |
|------------|----------|-----------------|
| Dashboard empty | `dash_empty_01.png` | New user, no projects |
| Dashboard with projects | `dash_projects_01.png` | Has previous videos |
| Dashboard loading | `dash_loading_01.png` | Projects loading |
| Navigation open | `dash_nav_open_01.png` | Side menu expanded |
| Navigation closed | `dash_nav_closed_01.png` | Side menu collapsed |
| Profile dropdown | `dash_profile_01.png` | User menu open |
| Settings page | `dash_settings_01.png` | Account settings |
| Billing page | `dash_billing_01.png` | Subscription info |
| Notification panel | `dash_notif_01.png` | Notifications open |
| Search open | `dash_search_01.png` | Search bar focused |
| Search results | `dash_search_results_01.png` | Search showing results |
| Help/FAQ | `dash_help_01.png` | Help section |
| Project hover | `dash_hover_01.png` | Hovering over a project |
| Project context menu | `dash_context_01.png` | Right-click menu |
| Keyboard shortcuts | `dash_shortcuts_01.png` | Shortcuts modal |

### CATEGORY 3: Model Selection (15 screenshots minimum)

| Screenshot | Filename | What to capture |
|------------|----------|-----------------|
| Model selector closed | `model_closed_01.png` | Dropdown not open |
| Model selector open | `model_open_01.png` | All models visible |
| Animate selected | `model_animate_01.png` | Animate option highlighted |
| Animate info | `model_animate_info_01.png` | Animate description popup |
| Text-to-Video selected | `model_txt2vid_01.png` | Text-to-video highlighted |
| Image-to-Video selected | `model_img2vid_01.png` | Image-to-video highlighted |
| Face Swap selected | `model_faceswap_01.png` | Face swap highlighted |
| Style Transfer | `model_style_01.png` | Style transfer option |
| Model comparison | `model_compare_01.png` | Comparison view |
| Model loading | `model_loading_01.png` | Model initializing |
| Model error | `model_error_01.png` | Model failed to load |
| Premium model locked | `model_locked_01.png` | Upgrade required |
| Model settings | `model_settings_01.png` | Model parameters |
| Model presets | `model_presets_01.png` | Preset configurations |
| Custom model | `model_custom_01.png` | Custom model upload |

### CATEGORY 4: Input/Upload (20 screenshots minimum)

| Screenshot | Filename | What to capture |
|------------|----------|-----------------|
| Upload area empty | `upload_empty_01.png` | Drag-drop zone empty |
| Upload drag hover | `upload_hover_01.png` | File being dragged over |
| Upload in progress | `upload_progress_01.png` | Upload percentage |
| Upload complete | `upload_complete_01.png` | File uploaded |
| Upload error | `upload_error_01.png` | Upload failed |
| Image preview | `upload_preview_01.png` | Uploaded image shown |
| Image crop tool | `upload_crop_01.png` | Cropping interface |
| Image adjust | `upload_adjust_01.png` | Brightness/contrast |
| Multiple images | `upload_multi_01.png` | Multiple files uploaded |
| Text prompt empty | `prompt_empty_01.png` | Text box empty |
| Text prompt filled | `prompt_filled_01.png` | Prompt typed |
| Text prompt long | `prompt_long_01.png` | Long prompt with scroll |
| Prompt suggestions | `prompt_suggest_01.png` | Autocomplete dropdown |
| Prompt templates | `prompt_templates_01.png` | Template library |
| Negative prompt | `prompt_negative_01.png` | Negative prompt field |
| Prompt history | `prompt_history_01.png` | Previous prompts |
| Voice input | `input_voice_01.png` | Voice recording |
| Reference video | `input_ref_01.png` | Reference video upload |
| Style reference | `input_style_01.png` | Style image upload |
| Clear all | `input_clear_01.png` | Clear button hover |

### CATEGORY 5: Generation Settings (15 screenshots minimum)

| Screenshot | Filename | What to capture |
|------------|----------|-----------------|
| Settings panel closed | `gen_closed_01.png` | Settings collapsed |
| Settings panel open | `gen_open_01.png` | All settings visible |
| Duration slider | `gen_duration_01.png` | Video length selector |
| FPS selector | `gen_fps_01.png` | Frame rate options |
| Resolution dropdown | `gen_res_01.png` | Quality options |
| Aspect ratio | `gen_aspect_01.png` | 16:9, 9:16, etc. |
| Motion intensity | `gen_motion_01.png` | Motion slider |
| Seed input | `gen_seed_01.png` | Seed number field |
| Random seed | `gen_random_01.png` | Randomize button |
| Advanced toggle | `gen_advanced_01.png` | Show advanced options |
| CFG scale | `gen_cfg_01.png` | Guidance scale |
| Steps slider | `gen_steps_01.png` | Inference steps |
| Scheduler | `gen_scheduler_01.png` | Scheduler dropdown |
| Preset save | `gen_preset_save_01.png` | Save as preset |
| Preset load | `gen_preset_load_01.png` | Load preset |

### CATEGORY 6: Generation Process (15 screenshots minimum)

| Screenshot | Filename | What to capture |
|------------|----------|-----------------|
| Generate button | `gen_button_01.png` | Green generate button |
| Generate button hover | `gen_button_hover_01.png` | Button highlighted |
| Generation starting | `gen_start_01.png` | "Starting..." message |
| Queue position | `gen_queue_01.png` | "Position 3 in queue" |
| Progress 10% | `gen_10_01.png` | Progress bar at 10% |
| Progress 50% | `gen_50_01.png` | Progress bar at 50% |
| Progress 90% | `gen_90_01.png` | Progress bar at 90% |
| Preview frame | `gen_preview_01.png` | Preview thumbnail |
| Generation complete | `gen_complete_01.png` | Success message |
| Generation error | `gen_error_01.png` | Error message |
| Timeout error | `gen_timeout_01.png` | Took too long |
| Cancel button | `gen_cancel_01.png` | Cancel option |
| Cancel confirm | `gen_cancel_confirm_01.png` | Confirm cancellation |
| Retry button | `gen_retry_01.png` | Try again option |
| Credits remaining | `gen_credits_01.png` | Credit counter |

### CATEGORY 7: Results/Output (15 screenshots minimum)

| Screenshot | Filename | What to capture |
|------------|----------|-----------------|
| Video player | `result_player_01.png` | Video playing |
| Video paused | `result_paused_01.png` | Video paused |
| Video controls | `result_controls_01.png` | Play/pause/seek |
| Fullscreen | `result_fullscreen_01.png` | Fullscreen mode |
| Download button | `result_download_01.png` | Download hover |
| Download options | `result_download_opts_01.png` | Format selection |
| Share button | `result_share_01.png` | Share options |
| Share link | `result_share_link_01.png` | Copy link dialog |
| Social share | `result_social_01.png` | Social media buttons |
| Regenerate | `result_regen_01.png` | Make another |
| Edit/refine | `result_edit_01.png` | Edit options |
| Save to library | `result_save_01.png` | Save button |
| Video info | `result_info_01.png` | Metadata panel |
| Side by side | `result_compare_01.png` | Compare versions |
| Rating/feedback | `result_rate_01.png` | Quality rating |

**TOTAL MINIMUM SCREENSHOTS: 105**

## 3.4 Screenshot Naming Convention

```
{category}_{screen}_{variation}_{number}.png
```

Examples:
- `auth_login_empty_01.png` - First empty login screen
- `auth_login_empty_02.png` - Second empty login screen (different browser width?)
- `model_animate_selected_01.png` - Animate model selected

**NEVER use:**
- Spaces in filenames: `login screen.png` ❌
- Special characters: `login@screen#1.png` ❌
- Uppercase: `Login_Screen_01.PNG` ❌

**ALWAYS use:**
- Lowercase: `login_screen_01.png` ✓
- Underscores: `login_empty_01.png` ✓
- Numbers with leading zero: `01`, `02`, ..., `99` ✓

## 3.5 Folder Structure

Create this EXACT structure:

```
higgsfield_dataset/
├── screenshots/
│   ├── auth/
│   │   ├── auth_login_empty_01.png
│   │   ├── auth_login_empty_02.png
│   │   ├── auth_login_filled_01.png
│   │   └── ... (all auth screenshots)
│   ├── dashboard/
│   │   ├── dash_empty_01.png
│   │   ├── dash_projects_01.png
│   │   └── ... (all dashboard screenshots)
│   ├── model/
│   │   ├── model_closed_01.png
│   │   ├── model_open_01.png
│   │   └── ... (all model screenshots)
│   ├── upload/
│   │   ├── upload_empty_01.png
│   │   ├── upload_hover_01.png
│   │   └── ... (all upload screenshots)
│   ├── generation/
│   │   ├── gen_button_01.png
│   │   ├── gen_start_01.png
│   │   └── ... (all generation screenshots)
│   └── results/
│       ├── result_player_01.png
│       ├── result_download_01.png
│       └── ... (all result screenshots)
├── data/
│   ├── embedding_train.jsonl
│   ├── reranker_train.jsonl
│   ├── sft_train.jsonl
│   └── dpo_train.jsonl
├── validation/
│   ├── embedding_val.jsonl
│   ├── reranker_val.jsonl
│   ├── sft_val.jsonl
│   └── dpo_val.jsonl
└── README.md
```

---

# PART 4: EXACT DATA FORMATS

## 4.1 Embedding Data Format

**File:** `data/embedding_train.jsonl`

**Each line is ONE triplet. 250 lines minimum.**

### EXACT FORMAT:

```json
{"query": {"text": "WHAT THE USER SEES AND WANTS", "image": "PATH_TO_SCREENSHOT"}, "positive": {"text": "CORRECT ACTION DESCRIPTION", "image": "PATH_TO_POSITIVE_SCREENSHOT"}, "negative": {"text": "WRONG BUT SIMILAR ACTION", "image": "PATH_TO_NEGATIVE_SCREENSHOT"}}
```

### COMPLETE REAL EXAMPLES:

**Example 1: Login Flow**
```json
{"query": {"text": "I see the Higgsfield login page with empty email and password fields. I need to enter my credentials.", "image": "screenshots/auth/auth_login_empty_01.png"}, "positive": {"text": "Click the email input field at the center of the form, then type your email address. The field has placeholder text 'Enter your email'.", "image": "screenshots/auth/auth_login_empty_01.png"}, "negative": {"text": "Click the 'Sign Up' link below the login form to create a new account instead of logging in.", "image": "screenshots/auth/auth_signup_empty_01.png"}}
```

**Example 2: Model Selection**
```json
{"query": {"text": "I'm on the main workspace and need to select the Animate model to create a video from my image.", "image": "screenshots/model/model_closed_01.png"}, "positive": {"text": "Click the model selector dropdown button showing 'Select Model' to open the model options menu.", "image": "screenshots/model/model_closed_01.png"}, "negative": {"text": "Click the Settings gear icon in the top right corner to open account settings.", "image": "screenshots/dashboard/dash_settings_01.png"}}
```

**Example 3: Upload Image**
```json
{"query": {"text": "I need to upload an image for video generation. I see the upload area with a drag-and-drop zone.", "image": "screenshots/upload/upload_empty_01.png"}, "positive": {"text": "Click the 'Upload Image' button in the center of the drag-drop zone, or drag your image file directly onto the dotted area.", "image": "screenshots/upload/upload_empty_01.png"}, "negative": {"text": "Click the 'Templates' button in the sidebar to browse pre-made video templates.", "image": "screenshots/dashboard/dash_nav_open_01.png"}}
```

**Example 4: Generation**
```json
{"query": {"text": "My image is uploaded and settings are configured. I want to start generating the video now.", "image": "screenshots/generation/gen_button_01.png"}, "positive": {"text": "Click the green 'Generate Video' button at the bottom right of the settings panel to begin video creation.", "image": "screenshots/generation/gen_button_01.png"}, "negative": {"text": "Click 'Save Settings' to save your current configuration as a preset for later use.", "image": "screenshots/generation/gen_preset_save_01.png"}}
```

**Example 5: Download Result**
```json
{"query": {"text": "The video generation is complete. I can see the video player with my generated video. I want to download it.", "image": "screenshots/results/result_player_01.png"}, "positive": {"text": "Click the 'Download' button below the video player to save the video to your computer.", "image": "screenshots/results/result_download_01.png"}, "negative": {"text": "Click the 'Regenerate' button to create a new version with the same settings.", "image": "screenshots/results/result_regen_01.png"}}
```

### WHAT MAKES A GOOD TRIPLET:

| Component | Good | Bad |
|-----------|------|-----|
| Query text | Describes what user SEES and WANTS | Vague like "I need help" |
| Query image | Matches what query describes | Random screenshot |
| Positive text | Specific action with location | "Click the button" |
| Positive image | Shows the element to click | Different screen |
| Negative text | Similar-sounding but WRONG | Completely unrelated |
| Negative image | Visually similar screen | Totally different |

### HARD NEGATIVES (CRITICAL FOR 3B MODEL):

The negative should be HARD to distinguish from positive. This is crucial for a small 3B model.

**EASY negative (BAD):**
- Positive: "Click Login button"
- Negative: "Close the browser"
- Too obviously wrong. Model learns nothing.

**HARD negative (GOOD):**
- Positive: "Click Login button to sign in"
- Negative: "Click Sign Up button to create account"
- Similar buttons, similar location, but different actions. Model must learn the difference.

### CHECKLIST FOR EACH TRIPLET:

- [ ] Query text describes screenshot accurately
- [ ] Query image file exists
- [ ] Positive text is actionable (has "click", "type", "scroll", etc.)
- [ ] Positive image is the SAME as query image (or shows result)
- [ ] Negative text sounds similar but is WRONG for this situation
- [ ] Negative image is from a SIMILAR-LOOKING screen
- [ ] All image paths are correct (use forward slashes: `screenshots/auth/login.png`)
- [ ] No trailing commas in JSON
- [ ] Entire triplet is on ONE line

## 4.2 Reranker Data Format

**File:** `data/reranker_train.jsonl`

**Each line is ONE pair. 150 lines minimum. 75 with label=1, 75 with label=0.**

### EXACT FORMAT:

```json
{"query": {"text": "DESCRIPTION OF SITUATION", "image": "PATH_TO_SCREENSHOT"}, "document": "POSSIBLE ACTION TO TAKE", "label": 1}
```

### COMPLETE REAL EXAMPLES:

**Positive Examples (label=1):**

```json
{"query": {"text": "Higgsfield login page is displayed with empty form fields. I need to start the login process.", "image": "screenshots/auth/auth_login_empty_01.png"}, "document": "Click on the email input field to begin entering login credentials. The email field is the first input in the login form.", "label": 1}
```

```json
{"query": {"text": "The model selection dropdown is open showing all available models. I want to use Animate.", "image": "screenshots/model/model_open_01.png"}, "document": "Click on 'Animate' option in the dropdown list. It's the first option that converts images to animated videos.", "label": 1}
```

```json
{"query": {"text": "Video generation is complete and the result is showing in the player. I want to save it.", "image": "screenshots/results/result_player_01.png"}, "document": "Click the Download button located below the video player to save the generated video to your local device.", "label": 1}
```

**Negative Examples (label=0):**

```json
{"query": {"text": "Higgsfield login page is displayed with empty form fields. I need to start the login process.", "image": "screenshots/auth/auth_login_empty_01.png"}, "document": "Click on the 'Forgot Password' link to reset your password. This is for users who can't remember their credentials.", "label": 0}
```

```json
{"query": {"text": "The model selection dropdown is open showing all available models. I want to use Animate.", "image": "screenshots/model/model_open_01.png"}, "document": "Click on 'Settings' to configure model parameters before selection.", "label": 0}
```

```json
{"query": {"text": "Video generation is complete and the result is showing in the player. I want to save it.", "image": "screenshots/results/result_player_01.png"}, "document": "Click the Share button to post the video on social media platforms.", "label": 0}
```

### BALANCE CHECKLIST:

Count your labels:
```bash
# On Linux/Mac
grep -c '"label": 1' reranker_train.jsonl  # Should be ~75
grep -c '"label": 0' reranker_train.jsonl  # Should be ~75
```

If unbalanced, add more of the minority class.

## 4.3 SFT Data Format

**File:** `data/sft_train.jsonl`

**Each line is ONE conversation. 50 lines minimum.**

### EXACT FORMAT:

```json
{"messages": [{"role": "user", "content": [{"type": "image", "image": "PATH"}, {"type": "text", "text": "USER REQUEST"}]}, {"role": "assistant", "content": "ASSISTANT RESPONSE WITH ACTION"}]}
```

### COMPLETE REAL EXAMPLES:

**Example 1: Login**
```json
{"messages": [{"role": "user", "content": [{"type": "image", "image": "screenshots/auth/auth_login_empty_01.png"}, {"type": "text", "text": "I need to log into Higgsfield with my email john@example.com"}]}, {"role": "assistant", "content": "I can see the Higgsfield login page. To log in with your email:\n\n1. The email field is in the center of the form\n2. Click the email input field\n3. Type: john@example.com\n\nACTION: CLICK(512, 300)\nACTION: TYPE(john@example.com)"}]}
```

**Example 2: Select Model**
```json
{"messages": [{"role": "user", "content": [{"type": "image", "image": "screenshots/model/model_closed_01.png"}, {"type": "text", "text": "I want to animate my uploaded image. Which model should I use?"}]}, {"role": "assistant", "content": "For animating images, you should use the 'Animate' model. Let me help you select it:\n\n1. First, I'll click the model dropdown to open it\n2. Then select 'Animate' from the list\n\nACTION: CLICK(400, 150)\n\nThis will open the model selector where you can choose Animate."}]}
```

**Example 3: Upload**
```json
{"messages": [{"role": "user", "content": [{"type": "image", "image": "screenshots/upload/upload_empty_01.png"}, {"type": "text", "text": "How do I upload my image?"}]}, {"role": "assistant", "content": "I can see the upload area. You have two options:\n\n1. Click the 'Upload Image' button in the center\n2. Or drag and drop your image onto the dotted area\n\nLet me click the upload button for you:\n\nACTION: CLICK(512, 384)\n\nThis will open a file picker where you can select your image."}]}
```

**Example 4: Generate**
```json
{"messages": [{"role": "user", "content": [{"type": "image", "image": "screenshots/generation/gen_button_01.png"}, {"type": "text", "text": "Everything is set up. Generate the video now."}]}, {"role": "assistant", "content": "Your settings look good. I'll start the generation now.\n\nACTION: CLICK(700, 500)\n\nThe green 'Generate' button will start the process. This typically takes 30-60 seconds depending on your settings."}]}
```

**Example 5: Download**
```json
{"messages": [{"role": "user", "content": [{"type": "image", "image": "screenshots/results/result_player_01.png"}, {"type": "text", "text": "Great! Now download the video for me."}]}, {"role": "assistant", "content": "The video looks great! I'll download it now.\n\nACTION: CLICK(600, 450)\n\nThis clicks the Download button. The video will be saved to your default downloads folder as an MP4 file."}]}
```

### ASSISTANT RESPONSE FORMAT:

Every assistant response should have:
1. **Acknowledgment** - What you see on screen
2. **Explanation** - What you're going to do and why
3. **ACTION** - The specific action in this format:
   - `ACTION: CLICK(x, y)` - Click at coordinates
   - `ACTION: TYPE(text)` - Type text
   - `ACTION: SCROLL(direction, amount)` - Scroll
   - `ACTION: WAIT(seconds)` - Wait

## 4.4 DPO Data Format

**File:** `data/dpo_train.jsonl`

**Each line is ONE preference pair. 25 lines minimum.**

### EXACT FORMAT:

```json
{"prompt": [{"type": "image", "image": "PATH"}, {"type": "text", "text": "USER REQUEST"}], "chosen": "GOOD RESPONSE", "rejected": "BAD RESPONSE"}
```

### COMPLETE REAL EXAMPLES:

```json
{"prompt": [{"type": "image", "image": "screenshots/generation/gen_button_01.png"}, {"type": "text", "text": "Generate video"}], "chosen": "I'll start the generation now.\n\nACTION: CLICK(700, 500)\n\nClicking the green Generate button. This will take about 30 seconds.", "rejected": "I see you want to generate a video. There are several options you could consider. You might want to adjust the duration first, or maybe change the resolution. Let me know what you'd prefer."}
```

```json
{"prompt": [{"type": "image", "image": "screenshots/auth/auth_login_empty_01.png"}, {"type": "text", "text": "Log me in"}], "chosen": "Clicking the email field to enter your credentials.\n\nACTION: CLICK(512, 300)\n\nPlease provide your email address.", "rejected": "I'd be happy to help you log in! First, could you tell me if you have an account? If not, you might need to sign up first. Also, do you remember your password?"}
```

```json
{"prompt": [{"type": "image", "image": "screenshots/results/result_player_01.png"}, {"type": "text", "text": "Download this"}], "chosen": "Downloading your video now.\n\nACTION: CLICK(600, 450)\n\nThe MP4 file will save to your Downloads folder.", "rejected": "Your video is ready! Would you like to download it, share it, or perhaps generate another version? There are also options to adjust the quality."}
```

### WHAT MAKES CHOSEN BETTER THAN REJECTED:

| Chosen (Good) | Rejected (Bad) |
|---------------|----------------|
| Takes action immediately | Asks unnecessary questions |
| Has ACTION command | No ACTION command |
| Specific and concise | Vague and verbose |
| Confident | Uncertain |
| 2-3 sentences | 5+ sentences |

---

# PART 5: CREATING THE DATASET STEP BY STEP

## Step 1: Set Up Your Folder (5 minutes)

Open terminal and run:

```bash
mkdir -p higgsfield_dataset/screenshots/{auth,dashboard,model,upload,generation,results}
mkdir -p higgsfield_dataset/data
mkdir -p higgsfield_dataset/validation
```

## Step 2: Take All Screenshots (2-3 hours)

Go through Higgsfield and take EVERY screenshot listed in Part 3.3.

**Pro tip:** Use this checklist script:

```bash
#!/bin/bash
# Save as check_screenshots.sh

REQUIRED=(
    "screenshots/auth/auth_login_empty_01.png"
    "screenshots/auth/auth_login_filled_01.png"
    "screenshots/auth/auth_signup_empty_01.png"
    "screenshots/model/model_closed_01.png"
    "screenshots/model/model_open_01.png"
    "screenshots/model/model_animate_01.png"
    "screenshots/upload/upload_empty_01.png"
    "screenshots/upload/upload_complete_01.png"
    "screenshots/generation/gen_button_01.png"
    "screenshots/generation/gen_50_01.png"
    "screenshots/results/result_player_01.png"
    "screenshots/results/result_download_01.png"
)

for file in "${REQUIRED[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ MISSING: $file"
    fi
done
```

## Step 3: Create Embedding Data (2-3 hours)

Open a text editor (VS Code, Sublime, etc.)

Create file: `higgsfield_dataset/data/embedding_train.jsonl`

Write 250 triplets, ONE PER LINE.

**Strategy for 250 triplets:**
- 50 triplets for authentication flows
- 50 triplets for dashboard/navigation
- 50 triplets for model selection
- 50 triplets for upload/input
- 50 triplets for generation/results

## Step 4: Create Reranker Data (1-2 hours)

Create file: `higgsfield_dataset/data/reranker_train.jsonl`

Write 150 pairs (75 positive, 75 negative), ONE PER LINE.

**Strategy:**
- For each screenshot, create 1-2 positive pairs and 1-2 negative pairs
- Make sure negatives are HARD (similar but wrong)

## Step 5: Create SFT Data (1-2 hours)

Create file: `higgsfield_dataset/data/sft_train.jsonl`

Write 50 conversations, ONE PER LINE.

**Strategy:**
- Cover the main workflow: Login → Select Model → Upload → Configure → Generate → Download
- Include error recovery scenarios
- Include variations in user requests

## Step 6: Create DPO Data (30 minutes)

Create file: `higgsfield_dataset/data/dpo_train.jsonl`

Write 25 preference pairs, ONE PER LINE.

**Strategy:**
- Take your best SFT responses as "chosen"
- Write vague/unhelpful versions as "rejected"

## Step 7: Create Validation Sets (30 minutes)

Copy 10% of each dataset to validation:

```bash
# For each file, take last 10% as validation
head -n 225 data/embedding_train.jsonl > data/embedding_train_final.jsonl
tail -n 25 data/embedding_train.jsonl > validation/embedding_val.jsonl
mv data/embedding_train_final.jsonl data/embedding_train.jsonl

head -n 135 data/reranker_train.jsonl > data/reranker_train_final.jsonl
tail -n 15 data/reranker_train.jsonl > validation/reranker_val.jsonl
mv data/reranker_train_final.jsonl data/reranker_train.jsonl

head -n 45 data/sft_train.jsonl > data/sft_train_final.jsonl
tail -n 5 data/sft_train.jsonl > validation/sft_val.jsonl
mv data/sft_train_final.jsonl data/sft_train.jsonl

head -n 22 data/dpo_train.jsonl > data/dpo_train_final.jsonl
tail -n 3 data/dpo_train.jsonl > validation/dpo_val.jsonl
mv data/dpo_train_final.jsonl data/dpo_train.jsonl
```

## Step 8: Validate Your Data (15 minutes)

Run this Python script to check for errors:

```python
#!/usr/bin/env python3
"""Save as validate_dataset.py"""

import json
import os
from pathlib import Path

def validate_jsonl(filepath, required_fields):
    """Validate a JSONL file."""
    errors = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                for field in required_fields:
                    if field not in data:
                        errors.append(f"Line {i}: Missing field '{field}'")

                # Check image paths exist
                def check_images(obj, path=""):
                    if isinstance(obj, dict):
                        if 'image' in obj:
                            img_path = obj['image']
                            if not os.path.exists(img_path):
                                errors.append(f"Line {i}: Image not found: {img_path}")
                        for k, v in obj.items():
                            check_images(v, f"{path}.{k}")
                    elif isinstance(obj, list):
                        for item in obj:
                            check_images(item, path)

                check_images(data)

            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: Invalid JSON - {e}")

    return errors

# Validate each file
files = {
    'data/embedding_train.jsonl': ['query', 'positive', 'negative'],
    'data/reranker_train.jsonl': ['query', 'document', 'label'],
    'data/sft_train.jsonl': ['messages'],
    'data/dpo_train.jsonl': ['prompt', 'chosen', 'rejected'],
}

print("=" * 60)
print("DATASET VALIDATION REPORT")
print("=" * 60)

for filepath, fields in files.items():
    if not os.path.exists(filepath):
        print(f"\n❌ {filepath}: FILE NOT FOUND")
        continue

    with open(filepath) as f:
        line_count = sum(1 for _ in f)

    errors = validate_jsonl(filepath, fields)

    if errors:
        print(f"\n❌ {filepath}: {line_count} lines, {len(errors)} errors")
        for error in errors[:5]:  # Show first 5 errors
            print(f"   - {error}")
        if len(errors) > 5:
            print(f"   ... and {len(errors) - 5} more errors")
    else:
        print(f"\n✓ {filepath}: {line_count} lines, 0 errors")

# Check label balance for reranker
if os.path.exists('data/reranker_train.jsonl'):
    pos = neg = 0
    with open('data/reranker_train.jsonl') as f:
        for line in f:
            data = json.loads(line)
            if data.get('label') == 1:
                pos += 1
            else:
                neg += 1
    print(f"\nReranker label balance: {pos} positive, {neg} negative")
    if abs(pos - neg) > 10:
        print("⚠️  WARNING: Labels are unbalanced! Add more of the minority class.")

print("\n" + "=" * 60)
```

Run it:
```bash
cd higgsfield_dataset
python validate_dataset.py
```

---

# PART 6: COMMON MISTAKES AND HOW TO AVOID THEM

## Mistake 1: Wrong JSON Format

**WRONG:**
```json
{
    "query": "text",
    "positive": "text",
    "negative": "text"
}
```
This spans multiple lines. JSONL requires ONE line per entry.

**CORRECT:**
```json
{"query": "text", "positive": "text", "negative": "text"}
```

## Mistake 2: Image Path Doesn't Exist

**WRONG:**
```json
{"query": {"image": "login.png"}}
```
Relative path might not resolve correctly.

**CORRECT:**
```json
{"query": {"image": "screenshots/auth/auth_login_empty_01.png"}}
```
Full path from dataset root.

## Mistake 3: Trailing Comma

**WRONG:**
```json
{"query": "text", "positive": "text",}
```
That comma after "text" breaks JSON.

**CORRECT:**
```json
{"query": "text", "positive": "text"}
```

## Mistake 4: Unbalanced Reranker Labels

**WRONG:**
- 120 examples with label=1
- 30 examples with label=0

Model will be biased toward predicting 1.

**CORRECT:**
- 75 examples with label=1
- 75 examples with label=0

## Mistake 5: Easy Negatives

**WRONG:**
- Query: "Login to the website"
- Negative: "Read the documentation about API"

Too easy. Model learns nothing useful.

**CORRECT:**
- Query: "Login to the website"
- Negative: "Create a new account on the website"

Similar context, wrong action.

## Mistake 6: Vague Action Descriptions

**WRONG:**
```
"Click the button"
```
Which button? Where?

**CORRECT:**
```
"Click the green 'Generate' button in the bottom right corner of the settings panel, at approximate coordinates (700, 500)"
```

## Mistake 7: Screenshot Resolution Mismatch

**WRONG:** Taking screenshots at different resolutions
- login.png: 1920x1080
- dashboard.png: 1024x768
- generate.png: 1440x900

Model gets confused.

**CORRECT:** All screenshots at 1024x768.

## Mistake 8: Forgetting Validation Split

**WRONG:** Using all data for training.

You can't measure if your model is learning or memorizing.

**CORRECT:** Keep 10% for validation. Never train on validation data.

---

# PART 7: NUMBERS SUMMARY FOR 3B MODEL

## Minimum Viable Dataset:

| Dataset | Count | Time to Create |
|---------|-------|----------------|
| Screenshots | 105 | 2-3 hours |
| Embedding triplets | 250 | 2-3 hours |
| Reranker pairs | 150 | 1-2 hours |
| SFT conversations | 50 | 1-2 hours |
| DPO pairs | 25 | 30 minutes |
| **TOTAL** | **475 data points** | **7-10 hours** |

## Training Time (on single A100 GPU):

| Model | Time | VRAM Needed |
|-------|------|-------------|
| Embedding (3B) | 1-2 hours | 24GB |
| Reranker (3B) | 30-60 minutes | 24GB |
| SFT (3B) | 2-4 hours | 24GB |
| DPO (3B) | 1-2 hours | 24GB |
| **TOTAL** | **5-9 hours** | 24GB |

## Expected Performance:

With minimum data on 3B model:
- Task success rate: ~60-70%
- Action accuracy: ~75-85%

With recommended data (2x minimum):
- Task success rate: ~75-85%
- Action accuracy: ~85-92%

---

# PART 8: QUICK REFERENCE CARD

Print this and keep it at your desk:

```
┌────────────────────────────────────────────────────────────────┐
│                    DATASET QUICK REFERENCE                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  SCREENSHOTS: 1024x768 PNG, 105 minimum                        │
│                                                                │
│  EMBEDDING DATA (embedding_train.jsonl):                       │
│  - 250 triplets minimum                                        │
│  - Format: {"query": {...}, "positive": {...}, "negative": {...}} │
│  - One line per triplet                                        │
│  - Hard negatives only                                         │
│                                                                │
│  RERANKER DATA (reranker_train.jsonl):                         │
│  - 150 pairs minimum (75 pos, 75 neg)                          │
│  - Format: {"query": {...}, "document": "...", "label": 1}     │
│  - One line per pair                                           │
│  - Must be balanced                                            │
│                                                                │
│  SFT DATA (sft_train.jsonl):                                   │
│  - 50 conversations minimum                                    │
│  - Format: {"messages": [...]}                                 │
│  - Include ACTION: CLICK(x,y) in responses                     │
│                                                                │
│  DPO DATA (dpo_train.jsonl):                                   │
│  - 25 pairs minimum                                            │
│  - Format: {"prompt": [...], "chosen": "...", "rejected": "..."}│
│  - Chosen: direct action, Rejected: vague response             │
│                                                                │
│  VALIDATION: 10% of each dataset                               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

# PART 9: FINAL CHECKLIST BEFORE TRAINING

Run through this checklist before handing off the dataset:

## Screenshots
- [ ] All 105+ screenshots taken
- [ ] All screenshots are 1024x768 PNG
- [ ] All screenshots are in correct folders
- [ ] No missing or corrupted files

## Embedding Data
- [ ] 250+ triplets in embedding_train.jsonl
- [ ] Each triplet on one line
- [ ] All image paths valid
- [ ] Hard negatives (not easy ones)
- [ ] 25 triplets in embedding_val.jsonl

## Reranker Data
- [ ] 150+ pairs in reranker_train.jsonl
- [ ] ~75 with label=1, ~75 with label=0
- [ ] Each pair on one line
- [ ] All image paths valid
- [ ] 15 pairs in reranker_val.jsonl

## SFT Data
- [ ] 50+ conversations in sft_train.jsonl
- [ ] Each conversation on one line
- [ ] All have ACTION commands in responses
- [ ] All image paths valid
- [ ] 5 conversations in sft_val.jsonl

## DPO Data
- [ ] 25+ pairs in dpo_train.jsonl
- [ ] Chosen is always better than rejected
- [ ] Chosen has ACTION, rejected is vague
- [ ] All image paths valid
- [ ] 3 pairs in dpo_val.jsonl

## Validation
- [ ] Run validate_dataset.py with 0 errors
- [ ] Reranker labels are balanced
- [ ] No duplicate entries
- [ ] All paths use forward slashes

If all boxes are checked, the dataset is ready for training.

---

**END OF INTERN GUIDE**

Questions? Something unclear? Ask before you start. It's easier to clarify now than to redo 10 hours of work.
