# DATA CURATOR BIBLE
## Complete Guide for Someone Who Knows NOTHING

**READ THIS ENTIRE DOCUMENT BEFORE DOING ANYTHING.**

---

# PART 0: WHAT THE FUCK ARE WE DOING?

## The Goal
We're teaching a robot (AI model) to use a computer like a human. Specifically, to use Higgsfield AI (a video generation website).

## Why We Need Data
The AI learns by looking at examples. More examples = smarter AI. Better examples = MUCH smarter AI.

## What You'll Create
You will create 4 datasets (collections of examples):

| Dataset | What It Teaches | Min Examples | Your Target |
|---------|-----------------|--------------|-------------|
| Embedding | "Find similar things" | 500 | 2,000 |
| Reranker | "Pick the best match" | 300 | 1,500 |
| SFT | "Do the right action" | 100 | 500 |
| DPO | "This is better than that" | 50 | 200 |

**TOTAL: You need to create ~4,200 examples**

This sounds like a lot. It's not. You'll get fast. Budget 2-3 weeks.

---

# PART 1: SETUP YOUR WORKSPACE

## Step 1.1: Create Folder Structure

Open your terminal (or file explorer) and create these EXACT folders:

```
higgsfield_data/
├── screenshots/
│   ├── raw/                    # Your original screenshots go here
│   ├── login/                  # Login page screenshots
│   ├── dashboard/              # Dashboard screenshots
│   ├── video_workspace/        # Video creation screenshots
│   ├── model_selection/        # Model dropdown screenshots
│   ├── motion_presets/         # Camera motion screenshots
│   └── generation/             # Generate button, results
├── embedding/
│   └── train.jsonl             # You'll create this file
├── reranker/
│   └── train.jsonl             # You'll create this file
├── sft/
│   └── train.jsonl             # You'll create this file
├── dpo/
│   └── train.jsonl             # You'll create this file
└── logs/
    └── curation_log.md         # Track your progress here
```

**DO THIS NOW.** Create every single folder. Don't skip any.

## Step 1.2: Install Screenshot Tool

You need a tool to take screenshots with coordinates.

**On Mac:**
- Press Cmd+Shift+4, then hover over elements
- The coordinates show in the corner
- Or use: `brew install flameshot`

**On Windows:**
- Download ShareX (free): https://getsharex.com/
- It shows coordinates when you hover

**On Linux:**
- `sudo apt install flameshot`

## Step 1.3: Set Up Browser

1. Open Chrome (use Chrome, not Firefox/Safari)
2. Set window size to EXACTLY 1920x1080
   - Right-click > Inspect > Click three dots > More tools > Device toolbar
   - Set to "Responsive" and type 1920 x 1080
3. Go to https://higgsfield.ai
4. Create an account if you don't have one

---

# PART 2: TAKING SCREENSHOTS

## THE GOLDEN RULES OF SCREENSHOTS

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SCREENSHOT RULES                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. ALWAYS use 1920x1080 resolution. ALWAYS.                               │
│                                                                             │
│  2. ALWAYS capture the FULL browser window (not just a piece)              │
│                                                                             │
│  3. ALWAYS save as PNG (not JPG, not WebP, not anything else)              │
│                                                                             │
│  4. ALWAYS use lowercase filenames with underscores                        │
│     GOOD: login_page_google_button.png                                     │
│     BAD:  Login Page Google Button.png                                     │
│     BAD:  loginPageGoogleButton.png                                        │
│                                                                             │
│  5. ALWAYS record the coordinates of the element you want to click         │
│                                                                             │
│  6. NEVER crop the screenshots                                             │
│                                                                             │
│  7. NEVER edit the screenshots (no arrows, circles, text)                  │
│                                                                             │
│  8. NEVER take screenshots of error pages or loading screens               │
│     (unless specifically documenting errors)                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Screenshot Naming Convention

Use this EXACT format:
```
[page]_[state]_[element].png
```

Examples:
```
login_default_full.png              # Login page, nothing clicked
login_email_focused.png             # Login page, email field clicked
login_google_hover.png              # Login page, hovering over Google button
dashboard_default_full.png          # Dashboard, just loaded
dashboard_create_hover.png          # Dashboard, hovering over Create
dashboard_create_expanded.png       # Dashboard, Create menu open
workspace_default_full.png          # Video workspace, just loaded
workspace_model_dropdown_open.png   # Model selector expanded
workspace_model_wan26_hover.png     # Hovering over WAN 2.6
workspace_motion_panel_open.png     # Motion presets visible
workspace_motion_pushin_hover.png   # Hovering over Push In preset
workspace_prompt_focused.png        # Clicked in prompt field
workspace_ready_to_generate.png     # Everything filled, ready to go
workspace_generate_hover.png        # Hovering over Generate button
```

## What Screenshots to Take

### LOGIN PAGE (Take 15 screenshots)

Go to: https://higgsfield.ai/auth

```
Screenshot 1: login_default_full.png
- Just the page, nothing clicked or hovered
- Record: Nothing

Screenshot 2: login_google_hover.png
- Hover your mouse over "Sign in with Google"
- Record: Coordinates of button center (e.g., x=640, y=400)

Screenshot 3: login_google_clicked.png
- Click the Google button (let the popup appear)
- Record: Same coordinates

Screenshot 4: login_email_hover.png
- Hover over email input field
- Record: Coordinates

Screenshot 5: login_email_focused.png
- Click the email field (cursor should blink inside)
- Record: Coordinates

Screenshot 6: login_email_filled.png
- Type "test@example.com" in email field
- Record: Coordinates

Screenshot 7: login_password_hover.png
- Hover over password field
- Record: Coordinates

Screenshot 8: login_password_focused.png
- Click password field
- Record: Coordinates

Screenshot 9: login_password_filled.png
- Type "********" (or actual dots that appear)
- Record: Coordinates

Screenshot 10: login_submit_hover.png
- Hover over login/submit button
- Record: Coordinates

Screenshot 11: login_submit_clicked.png
- Click submit (capture moment of click)
- Record: Coordinates

Screenshot 12: login_signup_hover.png
- Hover over "Sign up" link
- Record: Coordinates

Screenshot 13: login_forgot_hover.png
- Hover over "Forgot password" if it exists
- Record: Coordinates

Screenshot 14: login_error_invalid.png
- Submit with wrong credentials, capture error
- Record: Nothing (error state)

Screenshot 15: login_loading.png
- Capture loading spinner if visible
- Record: Nothing (loading state)
```

### DASHBOARD (Take 15 screenshots)

Log in successfully first, then go to dashboard.

```
Screenshot 1: dashboard_default_full.png
- Dashboard just loaded, nothing clicked
- Record: Nothing

Screenshot 2: dashboard_create_hover.png
- Hover over "Create" button (usually top-left)
- Record: Coordinates (e.g., x=100, y=50)

Screenshot 3: dashboard_create_expanded.png
- Click Create, menu expands showing Video/Image options
- Record: Coordinates of Create button

Screenshot 4: dashboard_create_video_hover.png
- Hover over "Video" option in dropdown
- Record: Coordinates of Video option

Screenshot 5: dashboard_create_image_hover.png
- Hover over "Image" option in dropdown
- Record: Coordinates of Image option

Screenshot 6: dashboard_myvideos_hover.png
- Hover over "My Videos" or similar tab
- Record: Coordinates

Screenshot 7: dashboard_credits_visible.png
- Make sure credits display is visible
- Record: Coordinates of credits area

Screenshot 8: dashboard_profile_hover.png
- Hover over profile/account icon
- Record: Coordinates

Screenshot 9: dashboard_profile_expanded.png
- Click profile to expand menu
- Record: Coordinates

Screenshot 10: dashboard_logout_hover.png
- Hover over logout option
- Record: Coordinates

Screenshot 11-15: (Additional states)
- Any other buttons, tabs, or states you see
```

### VIDEO WORKSPACE (Take 40 screenshots)

Click Create > Video to enter workspace.

```
MODEL SELECTION (10 screenshots):

Screenshot 1: workspace_default_full.png
- Workspace just loaded
- Record: Nothing

Screenshot 2: workspace_model_selector_hover.png
- Hover over model selector dropdown
- Record: Coordinates

Screenshot 3: workspace_model_dropdown_open.png
- Click to open dropdown
- Record: Coordinates of dropdown

Screenshot 4: workspace_model_wan26_hover.png
- Hover over "WAN 2.6" option
- Record: Coordinates

Screenshot 5: workspace_model_wan26_selected.png
- Click WAN 2.6, it's now selected
- Record: Coordinates

Screenshot 6: workspace_model_kling_hover.png
- Open dropdown again, hover over Kling 2.6
- Record: Coordinates

Screenshot 7: workspace_model_kling_selected.png
- Select Kling 2.6
- Record: Coordinates

Screenshot 8: workspace_model_minimax_hover.png
- Hover over Minimax Hailuo
- Record: Coordinates

Screenshot 9: workspace_model_minimax_selected.png
- Select Minimax
- Record: Coordinates

Screenshot 10: workspace_model_other_hover.png
- Any other model options
- Record: Coordinates

---

MOTION PRESETS (15 screenshots):

Screenshot 11: workspace_motion_panel_visible.png
- Motion control panel visible
- Record: Coordinates of panel

Screenshot 12: workspace_motion_change_hover.png
- Hover over "Change" button for motion
- Record: Coordinates

Screenshot 13: workspace_motion_mix_hover.png
- Hover over "Mix" button
- Record: Coordinates

Screenshot 14: workspace_motion_presets_open.png
- Click to open motion preset gallery
- Record: Coordinates

Screenshot 15: workspace_motion_pushin_hover.png
- Hover over "Push In" preset
- Record: Coordinates

Screenshot 16: workspace_motion_pushin_selected.png
- Select Push In
- Record: Coordinates

Screenshot 17: workspace_motion_pullback_hover.png
- Hover over "Pull Back"
- Record: Coordinates

Screenshot 18: workspace_motion_panleft_hover.png
- Hover over "Pan Left"
- Record: Coordinates

Screenshot 19: workspace_motion_panright_hover.png
- Hover over "Pan Right"
- Record: Coordinates

Screenshot 20: workspace_motion_tiltup_hover.png
- Hover over "Tilt Up"
- Record: Coordinates

Screenshot 21: workspace_motion_tiltdown_hover.png
- Hover over "Tilt Down"
- Record: Coordinates

Screenshot 22: workspace_motion_orbit_hover.png
- Hover over "Orbit"
- Record: Coordinates

Screenshot 23: workspace_motion_dolly_hover.png
- Hover over "Dolly Zoom"
- Record: Coordinates

Screenshot 24: workspace_motion_static_hover.png
- Hover over "Static" or no motion
- Record: Coordinates

Screenshot 25: workspace_motion_other_hover.png
- Any other presets
- Record: Coordinates

---

UPLOAD & INPUT (10 screenshots):

Screenshot 26: workspace_upload_hover.png
- Hover over "Upload Image" button
- Record: Coordinates

Screenshot 27: workspace_upload_clicked.png
- Click upload (file picker appears)
- Record: Coordinates

Screenshot 28: workspace_image_uploaded.png
- After uploading an image
- Record: Coordinates of image preview

Screenshot 29: workspace_prompt_hover.png
- Hover over prompt input field
- Record: Coordinates

Screenshot 30: workspace_prompt_focused.png
- Click prompt field (cursor inside)
- Record: Coordinates

Screenshot 31: workspace_prompt_filled.png
- Type a prompt: "A woman walking in garden"
- Record: Coordinates

Screenshot 32: workspace_duration_hover.png
- Hover over duration slider
- Record: Coordinates

Screenshot 33: workspace_duration_changed.png
- Change duration value
- Record: Coordinates

Screenshot 34: workspace_settings_hover.png
- Hover over any settings/options
- Record: Coordinates

Screenshot 35: workspace_settings_expanded.png
- Expand settings if collapsible
- Record: Coordinates

---

GENERATION (5 screenshots):

Screenshot 36: workspace_ready_to_generate.png
- Everything filled in, ready to generate
- Record: Nothing (overview state)

Screenshot 37: workspace_generate_hover.png
- Hover over Generate button
- Record: Coordinates

Screenshot 38: workspace_generate_clicked.png
- Click Generate
- Record: Coordinates

Screenshot 39: workspace_generating_progress.png
- Generation in progress (loading)
- Record: Nothing (progress state)

Screenshot 40: workspace_generation_complete.png
- Video finished generating
- Record: Coordinates of result
```

## Recording Coordinates

For EVERY screenshot where you hover/click something, write down:

```
File: workspace_model_wan26_hover.png
Element: WAN 2.6 model option
Action: Click
X: 300
Y: 200
Notes: Third option in dropdown
```

Save this in a spreadsheet or text file called `coordinates.csv`:

```csv
filename,element,action,x,y,notes
login_google_hover.png,Google Sign In,click,640,400,Blue OAuth button
login_email_focused.png,Email input,click,640,300,Text input field
login_password_focused.png,Password input,click,640,360,Password field
dashboard_create_hover.png,Create button,click,100,50,Top left navigation
dashboard_create_video_hover.png,Video option,click,100,100,In dropdown menu
workspace_model_wan26_hover.png,WAN 2.6,click,300,200,Model selector
workspace_model_kling_hover.png,Kling 2.6,click,300,240,Model selector
workspace_model_minimax_hover.png,Minimax,click,300,280,Model selector
workspace_motion_pushin_hover.png,Push In,click,200,200,Motion preset
workspace_upload_hover.png,Upload Image,click,640,400,Upload button
workspace_prompt_focused.png,Prompt input,click,640,500,Text area
workspace_generate_hover.png,Generate,click,640,600,Purple CTA button
```

---

# PART 3: CREATING THE EMBEDDING DATASET

## What Is The Embedding Dataset?

The embedding model learns: "These two things are similar" and "These two things are different."

You give it:
- A **query**: What the user wants to do
- A **positive**: An example that MATCHES the query
- **Negatives**: Examples that DON'T match (but might look similar)

## The File Format

Create a file called `embedding/train.jsonl`

**JSONL means:** Each line is a separate JSON object. One example per line. No commas between lines.

**CORRECT:**
```
{"query": {...}, "positive": {...}, "negatives": [...]}
{"query": {...}, "positive": {...}, "negatives": [...]}
{"query": {...}, "positive": {...}, "negatives": [...]}
```

**WRONG:**
```
[
  {"query": {...}, "positive": {...}, "negatives": [...]},
  {"query": {...}, "positive": {...}, "negatives": [...]}
]
```

**WRONG:**
```
{"query": {...}, "positive": {...}, "negatives": [...]},
{"query": {...}, "positive": {...}, "negatives": [...]},
```

## The Exact Structure

Here is ONE example, formatted nicely so you can see it:

```json
{
  "query": {
    "image": "screenshots/login/login_default_full.png",
    "text": "Click the Google sign in button"
  },
  "positive": {
    "image": "screenshots/login/login_google_hover.png",
    "text": "Google Sign In OAuth button, blue button with Google logo, coordinates (640, 400)"
  },
  "negatives": [
    {
      "image": "screenshots/login/login_email_focused.png",
      "text": "Email input field for typing email address",
      "negative_type": "same_page_wrong_element"
    },
    {
      "image": "screenshots/login/login_password_focused.png",
      "text": "Password input field for typing password",
      "negative_type": "same_page_wrong_element"
    },
    {
      "image": "screenshots/login/login_signup_hover.png",
      "text": "Sign up link to create new account",
      "negative_type": "same_page_wrong_element"
    }
  ]
}
```

**But in the actual file, it must be ONE LINE:**

```
{"query":{"image":"screenshots/login/login_default_full.png","text":"Click the Google sign in button"},"positive":{"image":"screenshots/login/login_google_hover.png","text":"Google Sign In OAuth button, blue button with Google logo, coordinates (640, 400)"},"negatives":[{"image":"screenshots/login/login_email_focused.png","text":"Email input field for typing email address","negative_type":"same_page_wrong_element"},{"image":"screenshots/login/login_password_focused.png","text":"Password input field for typing password","negative_type":"same_page_wrong_element"},{"image":"screenshots/login/login_signup_hover.png","text":"Sign up link to create new account","negative_type":"same_page_wrong_element"}]}
```

## Field Descriptions

### query (REQUIRED)
What the user wants to do.

```json
"query": {
  "image": "path/to/screenshot.png",   // REQUIRED: Screenshot showing current state
  "text": "Natural language instruction"  // REQUIRED: What user wants to do
}
```

**Good query texts:**
- "Click the Google sign in button"
- "Sign in with my Google account"
- "Press the Google login button"
- "Authenticate using Google"
- "Use Google to log in"

**Bad query texts:**
- "Click" (too vague)
- "Google" (not an instruction)
- "The blue button" (which blue button?)
- "Login" (login how? where?)

### positive (REQUIRED)
An example that CORRECTLY matches the query.

```json
"positive": {
  "image": "path/to/screenshot.png",   // Screenshot showing the target element
  "text": "Description with coordinates"  // What it is + where it is
}
```

**Good positive texts:**
- "Google Sign In OAuth button, blue button with Google logo, coordinates (640, 400)"
- "Blue Google authentication button in center of login form at position (640, 400)"

**Bad positive texts:**
- "Google button" (too vague, no coordinates)
- "Click here" (not descriptive)
- "Button" (which button?)

### negatives (REQUIRED, minimum 1, recommended 3-5)
Examples that DON'T match but might be confused.

```json
"negatives": [
  {
    "image": "path/to/screenshot.png",
    "text": "Description of wrong element",
    "negative_type": "category of why it's wrong"
  }
]
```

**negative_type values (use these EXACT strings):**
- `same_page_wrong_element` - Same page but wrong thing to click
- `same_element_wrong_page` - Similar element but different page
- `similar_action_wrong_target` - Right action type but wrong target
- `wrong_action_type` - Wrong type of action entirely
- `different_app` - Similar element in different application
- `visually_similar` - Looks similar but is wrong
- `semantically_similar` - Sounds similar but is wrong

## Complete Examples For Each Workflow

### LOGIN WORKFLOW EXAMPLES

**Example 1: Click Google Sign In**
```
{"query":{"image":"screenshots/login/login_default_full.png","text":"Click the Google sign in button"},"positive":{"image":"screenshots/login/login_google_hover.png","text":"Google Sign In OAuth button, blue button with Google logo at center of form, coordinates (640, 400)"},"negatives":[{"image":"screenshots/login/login_email_focused.png","text":"Email input field for manual login","negative_type":"same_page_wrong_element"},{"image":"screenshots/login/login_signup_hover.png","text":"Sign up link for new account creation","negative_type":"same_page_wrong_element"},{"image":"screenshots/dashboard/dashboard_default_full.png","text":"Dashboard page with no login elements","negative_type":"wrong_page"}]}
```

**Example 2: Enter email address**
```
{"query":{"image":"screenshots/login/login_default_full.png","text":"Type my email address"},"positive":{"image":"screenshots/login/login_email_focused.png","text":"Email text input field with placeholder text, coordinates (640, 300)"},"negatives":[{"image":"screenshots/login/login_password_focused.png","text":"Password input field below email","negative_type":"same_page_wrong_element"},{"image":"screenshots/login/login_google_hover.png","text":"Google sign in button not a text field","negative_type":"wrong_action_type"}]}
```

**Example 3: Submit login form**
```
{"query":{"image":"screenshots/login/login_email_filled.png","text":"Submit the login form"},"positive":{"image":"screenshots/login/login_submit_hover.png","text":"Login submit button below form fields, coordinates (640, 450)"},"negatives":[{"image":"screenshots/login/login_signup_hover.png","text":"Sign up link not submit button","negative_type":"same_page_wrong_element"},{"image":"screenshots/login/login_forgot_hover.png","text":"Forgot password link","negative_type":"same_page_wrong_element"}]}
```

**Example 4-10: Paraphrases of Example 1**

You need VARIATIONS of the same action with different wording:

```
{"query":{"image":"screenshots/login/login_default_full.png","text":"Sign in with Google"},"positive":{"image":"screenshots/login/login_google_hover.png","text":"Google Sign In OAuth button, blue button with Google logo at center of form, coordinates (640, 400)"},"negatives":[{"image":"screenshots/login/login_email_focused.png","text":"Email input field for manual login","negative_type":"same_page_wrong_element"}]}
```

```
{"query":{"image":"screenshots/login/login_default_full.png","text":"Use Google to authenticate"},"positive":{"image":"screenshots/login/login_google_hover.png","text":"Google Sign In OAuth button, blue button with Google logo at center of form, coordinates (640, 400)"},"negatives":[{"image":"screenshots/login/login_email_focused.png","text":"Email input field for manual login","negative_type":"same_page_wrong_element"}]}
```

```
{"query":{"image":"screenshots/login/login_default_full.png","text":"Press the Google login button"},"positive":{"image":"screenshots/login/login_google_hover.png","text":"Google Sign In OAuth button, blue button with Google logo at center of form, coordinates (640, 400)"},"negatives":[{"image":"screenshots/login/login_email_focused.png","text":"Email input field for manual login","negative_type":"same_page_wrong_element"}]}
```

```
{"query":{"image":"screenshots/login/login_default_full.png","text":"Log in using my Google account"},"positive":{"image":"screenshots/login/login_google_hover.png","text":"Google Sign In OAuth button, blue button with Google logo at center of form, coordinates (640, 400)"},"negatives":[{"image":"screenshots/login/login_email_focused.png","text":"Email input field for manual login","negative_type":"same_page_wrong_element"}]}
```

```
{"query":{"image":"screenshots/login/login_default_full.png","text":"Authenticate with Google OAuth"},"positive":{"image":"screenshots/login/login_google_hover.png","text":"Google Sign In OAuth button, blue button with Google logo at center of form, coordinates (640, 400)"},"negatives":[{"image":"screenshots/login/login_email_focused.png","text":"Email input field for manual login","negative_type":"same_page_wrong_element"}]}
```

### MODEL SELECTION EXAMPLES

**Example 11: Select WAN 2.6**
```
{"query":{"image":"screenshots/video_workspace/workspace_model_dropdown_open.png","text":"Select the WAN 2.6 model"},"positive":{"image":"screenshots/video_workspace/workspace_model_wan26_hover.png","text":"WAN 2.6 model option, best for cinematic continuity and multi-shot generation, coordinates (300, 200)"},"negatives":[{"image":"screenshots/video_workspace/workspace_model_kling_hover.png","text":"Kling 2.6 model for lip sync","negative_type":"same_page_wrong_element"},{"image":"screenshots/video_workspace/workspace_model_minimax_hover.png","text":"Minimax Hailuo 02 for fast iteration","negative_type":"same_page_wrong_element"}]}
```

**Example 12: Select fastest model**
```
{"query":{"image":"screenshots/video_workspace/workspace_model_dropdown_open.png","text":"Choose the fastest model for quick testing"},"positive":{"image":"screenshots/video_workspace/workspace_model_minimax_hover.png","text":"Minimax Hailuo 02 model, optimized for fast iteration and quick previews, coordinates (300, 280)"},"negatives":[{"image":"screenshots/video_workspace/workspace_model_wan26_hover.png","text":"WAN 2.6 slower but higher quality","negative_type":"same_page_wrong_element"},{"image":"screenshots/video_workspace/workspace_model_kling_hover.png","text":"Kling 2.6 for dialogue not speed","negative_type":"same_page_wrong_element"}]}
```

**Example 13: Select lip sync model**
```
{"query":{"image":"screenshots/video_workspace/workspace_model_dropdown_open.png","text":"Pick the model with best lip sync for dialogue"},"positive":{"image":"screenshots/video_workspace/workspace_model_kling_hover.png","text":"Kling 2.6 model with strong lip-sync and voice alignment capabilities, coordinates (300, 240)"},"negatives":[{"image":"screenshots/video_workspace/workspace_model_wan26_hover.png","text":"WAN 2.6 for cinematic not dialogue","negative_type":"same_page_wrong_element"},{"image":"screenshots/video_workspace/workspace_model_minimax_hover.png","text":"Minimax for speed not lip sync","negative_type":"same_page_wrong_element"}]}
```

### MOTION PRESET EXAMPLES

**Example 14: Push In motion**
```
{"query":{"image":"screenshots/motion_presets/workspace_motion_presets_open.png","text":"Apply a push in camera motion"},"positive":{"image":"screenshots/motion_presets/workspace_motion_pushin_hover.png","text":"Push In motion preset, camera moves toward subject creating dramatic effect, coordinates (200, 200)"},"negatives":[{"image":"screenshots/motion_presets/workspace_motion_pullback_hover.png","text":"Pull Back motion moves camera away opposite direction","negative_type":"same_page_wrong_element"},{"image":"screenshots/motion_presets/workspace_motion_orbit_hover.png","text":"Orbit motion rotates around subject different movement type","negative_type":"same_page_wrong_element"}]}
```

**Example 15: Pull Back motion**
```
{"query":{"image":"screenshots/motion_presets/workspace_motion_presets_open.png","text":"Create a reveal shot that pulls back"},"positive":{"image":"screenshots/motion_presets/workspace_motion_pullback_hover.png","text":"Pull Back motion preset, camera moves away from subject revealing wider scene, coordinates (350, 200)"},"negatives":[{"image":"screenshots/motion_presets/workspace_motion_pushin_hover.png","text":"Push In moves toward not away","negative_type":"same_page_wrong_element"},{"image":"screenshots/motion_presets/workspace_motion_panleft_hover.png","text":"Pan Left horizontal movement not backward","negative_type":"same_page_wrong_element"}]}
```

**Example 16: Orbit motion**
```
{"query":{"image":"screenshots/motion_presets/workspace_motion_presets_open.png","text":"Add an orbit camera movement around the subject"},"positive":{"image":"screenshots/motion_presets/workspace_motion_orbit_hover.png","text":"Orbit motion preset, 360 degree camera rotation around subject, coordinates (650, 300)"},"negatives":[{"image":"screenshots/motion_presets/workspace_motion_dolly_hover.png","text":"Dolly zoom is zoom effect not rotation","negative_type":"same_page_wrong_element"},{"image":"screenshots/motion_presets/workspace_motion_panleft_hover.png","text":"Pan is horizontal slide not orbit","negative_type":"same_page_wrong_element"}]}
```

### GENERATION WORKFLOW EXAMPLES

**Example 17: Upload image**
```
{"query":{"image":"screenshots/video_workspace/workspace_default_full.png","text":"Upload an image for video generation"},"positive":{"image":"screenshots/video_workspace/workspace_upload_hover.png","text":"Upload Image button to select reference image from file system, coordinates (640, 400)"},"negatives":[{"image":"screenshots/video_workspace/workspace_generate_hover.png","text":"Generate button starts creation not upload","negative_type":"same_page_wrong_element"},{"image":"screenshots/video_workspace/workspace_prompt_focused.png","text":"Prompt field for text not images","negative_type":"wrong_action_type"}]}
```

**Example 18: Enter prompt**
```
{"query":{"image":"screenshots/video_workspace/workspace_image_uploaded.png","text":"Enter a prompt describing the video"},"positive":{"image":"screenshots/video_workspace/workspace_prompt_focused.png","text":"Prompt text input field for describing desired video content, coordinates (640, 500)"},"negatives":[{"image":"screenshots/video_workspace/workspace_upload_hover.png","text":"Upload button for images not text","negative_type":"wrong_action_type"},{"image":"screenshots/video_workspace/workspace_generate_hover.png","text":"Generate button not text input","negative_type":"same_page_wrong_element"}]}
```

**Example 19: Generate video**
```
{"query":{"image":"screenshots/video_workspace/workspace_ready_to_generate.png","text":"Generate the video"},"positive":{"image":"screenshots/video_workspace/workspace_generate_hover.png","text":"Generate button, purple CTA button that starts video creation process, coordinates (640, 600)"},"negatives":[{"image":"screenshots/video_workspace/workspace_upload_hover.png","text":"Upload button already used","negative_type":"same_page_wrong_element"},{"image":"screenshots/video_workspace/workspace_prompt_focused.png","text":"Prompt field already filled","negative_type":"same_page_wrong_element"}]}
```

---

# PART 4: CREATING THE RERANKER DATASET

## What Is The Reranker Dataset?

The reranker learns: "Is this specific match relevant to the query? Yes or No."

It's simpler than embedding - just pairs with a 0 or 1 label.

## The File Format

File: `reranker/train.jsonl`

Each line:
```
{"query":{"image":"...","text":"..."},"document":{"image":"...","text":"..."},"label":1}
```

- `label: 1` means the document IS relevant to the query
- `label: 0` means the document is NOT relevant

## The Exact Structure

**Positive example (label=1):**
```json
{
  "query": {
    "image": "screenshots/login/login_default_full.png",
    "text": "Click the Google sign in button"
  },
  "document": {
    "image": "screenshots/login/login_google_hover.png",
    "text": "Google Sign In OAuth button, coordinates (640, 400)"
  },
  "label": 1
}
```

**Negative example (label=0):**
```json
{
  "query": {
    "image": "screenshots/login/login_default_full.png",
    "text": "Click the Google sign in button"
  },
  "document": {
    "image": "screenshots/login/login_email_focused.png",
    "text": "Email input field, coordinates (640, 300)"
  },
  "label": 0
}
```

## Label Distribution Rule

**IMPORTANT: You need MORE negatives than positives!**

Target ratio: 1 positive : 3 negatives

If you have 100 positive examples, you need ~300 negative examples.

**Why?** In real usage, most retrieved candidates are WRONG. The model needs to learn to reject bad matches confidently.

## Complete Reranker Examples

### Positive Examples (label=1)

```
{"query":{"image":"screenshots/login/login_default_full.png","text":"Click Google sign in"},"document":{"image":"screenshots/login/login_google_hover.png","text":"Google Sign In OAuth button at (640, 400)"},"label":1}
```

```
{"query":{"image":"screenshots/video_workspace/workspace_model_dropdown_open.png","text":"Select WAN 2.6 model"},"document":{"image":"screenshots/video_workspace/workspace_model_wan26_hover.png","text":"WAN 2.6 model card at (300, 200)"},"label":1}
```

```
{"query":{"image":"screenshots/video_workspace/workspace_model_dropdown_open.png","text":"Choose fastest model"},"document":{"image":"screenshots/video_workspace/workspace_model_minimax_hover.png","text":"Minimax Hailuo 02 fast model at (300, 280)"},"label":1}
```

```
{"query":{"image":"screenshots/motion_presets/workspace_motion_presets_open.png","text":"Apply push in motion"},"document":{"image":"screenshots/motion_presets/workspace_motion_pushin_hover.png","text":"Push In camera preset at (200, 200)"},"label":1}
```

```
{"query":{"image":"screenshots/video_workspace/workspace_ready_to_generate.png","text":"Start generating video"},"document":{"image":"screenshots/video_workspace/workspace_generate_hover.png","text":"Generate button at (640, 600)"},"label":1}
```

### Negative Examples (label=0)

**Same page, wrong element:**
```
{"query":{"image":"screenshots/login/login_default_full.png","text":"Click Google sign in"},"document":{"image":"screenshots/login/login_email_focused.png","text":"Email input field at (640, 300)"},"label":0}
```

```
{"query":{"image":"screenshots/login/login_default_full.png","text":"Click Google sign in"},"document":{"image":"screenshots/login/login_signup_hover.png","text":"Sign up link at (640, 480)"},"label":0}
```

```
{"query":{"image":"screenshots/login/login_default_full.png","text":"Click Google sign in"},"document":{"image":"screenshots/login/login_password_focused.png","text":"Password field at (640, 360)"},"label":0}
```

**Wrong model selected:**
```
{"query":{"image":"screenshots/video_workspace/workspace_model_dropdown_open.png","text":"Select WAN 2.6 model"},"document":{"image":"screenshots/video_workspace/workspace_model_kling_hover.png","text":"Kling 2.6 model at (300, 240)"},"label":0}
```

```
{"query":{"image":"screenshots/video_workspace/workspace_model_dropdown_open.png","text":"Select WAN 2.6 model"},"document":{"image":"screenshots/video_workspace/workspace_model_minimax_hover.png","text":"Minimax model at (300, 280)"},"label":0}
```

**Wrong motion preset:**
```
{"query":{"image":"screenshots/motion_presets/workspace_motion_presets_open.png","text":"Apply push in motion"},"document":{"image":"screenshots/motion_presets/workspace_motion_pullback_hover.png","text":"Pull Back preset at (350, 200)"},"label":0}
```

```
{"query":{"image":"screenshots/motion_presets/workspace_motion_presets_open.png","text":"Apply push in motion"},"document":{"image":"screenshots/motion_presets/workspace_motion_orbit_hover.png","text":"Orbit preset at (650, 300)"},"label":0}
```

**Wrong action entirely:**
```
{"query":{"image":"screenshots/video_workspace/workspace_ready_to_generate.png","text":"Start generating video"},"document":{"image":"screenshots/video_workspace/workspace_upload_hover.png","text":"Upload button at (640, 400)"},"label":0}
```

```
{"query":{"image":"screenshots/video_workspace/workspace_ready_to_generate.png","text":"Start generating video"},"document":{"image":"screenshots/video_workspace/workspace_prompt_focused.png","text":"Prompt text field at (640, 500)"},"label":0}
```

---

# PART 5: CREATING THE SFT DATASET

## What Is The SFT Dataset?

SFT (Supervised Fine-Tuning) teaches the model to generate the CORRECT action given a screenshot and instruction.

This is the main model that actually DOES things.

## The File Format

File: `sft/train.jsonl`

This uses a "messages" format with roles: system, user, assistant.

## The Exact Structure

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a computer use agent that automates tasks on Higgsfield AI. You see screenshots and execute actions. Respond with JSON: {\"action\": \"click|type|scroll\", \"x\": number, \"y\": number, \"text\": \"for type only\", \"reasoning\": \"why this action\"}"
    },
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "screenshots/login/login_default_full.png"},
        {"type": "text", "text": "Sign in with Google"}
      ]
    },
    {
      "role": "assistant",
      "content": "{\"action\": \"click\", \"x\": 640, \"y\": 400, \"reasoning\": \"Clicking the Google Sign In button to authenticate with OAuth\"}"
    }
  ]
}
```

**As one line:**
```
{"messages":[{"role":"system","content":"You are a computer use agent that automates tasks on Higgsfield AI. You see screenshots and execute actions. Respond with JSON: {\"action\": \"click|type|scroll\", \"x\": number, \"y\": number, \"text\": \"for type only\", \"reasoning\": \"why this action\"}"},{"role":"user","content":[{"type":"image","image":"screenshots/login/login_default_full.png"},{"type":"text","text":"Sign in with Google"}]},{"role":"assistant","content":"{\"action\": \"click\", \"x\": 640, \"y\": 400, \"reasoning\": \"Clicking the Google Sign In button to authenticate with OAuth\"}"}]}
```

## The System Prompt

Use this EXACT system prompt for ALL examples:

```
You are a computer use agent that automates tasks on Higgsfield AI. You see screenshots and execute actions. Respond with JSON: {"action": "click|type|scroll", "x": number, "y": number, "text": "for type only", "reasoning": "why this action"}
```

## Action Types

**click** - Click at coordinates
```json
{"action": "click", "x": 640, "y": 400, "reasoning": "Clicking the button"}
```

**type** - Type text (after clicking a text field)
```json
{"action": "type", "x": 640, "y": 300, "text": "hello@email.com", "reasoning": "Typing email address"}
```

**scroll** - Scroll up or down
```json
{"action": "scroll", "x": 640, "y": 400, "direction": "down", "amount": 300, "reasoning": "Scrolling to see more options"}
```

## Complete SFT Examples

### Login Examples

```
{"messages":[{"role":"system","content":"You are a computer use agent that automates tasks on Higgsfield AI. You see screenshots and execute actions. Respond with JSON: {\"action\": \"click|type|scroll\", \"x\": number, \"y\": number, \"text\": \"for type only\", \"reasoning\": \"why this action\"}"},{"role":"user","content":[{"type":"image","image":"screenshots/login/login_default_full.png"},{"type":"text","text":"Sign in with Google"}]},{"role":"assistant","content":"{\"action\": \"click\", \"x\": 640, \"y\": 400, \"reasoning\": \"Clicking the Google Sign In OAuth button to authenticate\"}"}]}
```

```
{"messages":[{"role":"system","content":"You are a computer use agent that automates tasks on Higgsfield AI. You see screenshots and execute actions. Respond with JSON: {\"action\": \"click|type|scroll\", \"x\": number, \"y\": number, \"text\": \"for type only\", \"reasoning\": \"why this action\"}"},{"role":"user","content":[{"type":"image","image":"screenshots/login/login_default_full.png"},{"type":"text","text":"I want to login with email instead"}]},{"role":"assistant","content":"{\"action\": \"click\", \"x\": 640, \"y\": 300, \"reasoning\": \"Clicking the email input field to enter credentials manually\"}"}]}
```

```
{"messages":[{"role":"system","content":"You are a computer use agent that automates tasks on Higgsfield AI. You see screenshots and execute actions. Respond with JSON: {\"action\": \"click|type|scroll\", \"x\": number, \"y\": number, \"text\": \"for type only\", \"reasoning\": \"why this action\"}"},{"role":"user","content":[{"type":"image","image":"screenshots/login/login_email_focused.png"},{"type":"text","text":"Enter email: john@example.com"}]},{"role":"assistant","content":"{\"action\": \"type\", \"x\": 640, \"y\": 300, \"text\": \"john@example.com\", \"reasoning\": \"Typing the email address into the focused email field\"}"}]}
```

### Model Selection Examples

```
{"messages":[{"role":"system","content":"You are a computer use agent that automates tasks on Higgsfield AI. You see screenshots and execute actions. Respond with JSON: {\"action\": \"click|type|scroll\", \"x\": number, \"y\": number, \"text\": \"for type only\", \"reasoning\": \"why this action\"}"},{"role":"user","content":[{"type":"image","image":"screenshots/video_workspace/workspace_model_dropdown_open.png"},{"type":"text","text":"Select WAN 2.6 for cinematic quality"}]},{"role":"assistant","content":"{\"action\": \"click\", \"x\": 300, \"y\": 200, \"reasoning\": \"Selecting WAN 2.6 model which is best for cinematic continuity and scene coverage\"}"}]}
```

```
{"messages":[{"role":"system","content":"You are a computer use agent that automates tasks on Higgsfield AI. You see screenshots and execute actions. Respond with JSON: {\"action\": \"click|type|scroll\", \"x\": number, \"y\": number, \"text\": \"for type only\", \"reasoning\": \"why this action\"}"},{"role":"user","content":[{"type":"image","image":"screenshots/video_workspace/workspace_model_dropdown_open.png"},{"type":"text","text":"I need the fastest model for quick iteration"}]},{"role":"assistant","content":"{\"action\": \"click\", \"x\": 300, \"y\": 280, \"reasoning\": \"Selecting Minimax Hailuo 02 which is optimized for fast generation and quick previews\"}"}]}
```

```
{"messages":[{"role":"system","content":"You are a computer use agent that automates tasks on Higgsfield AI. You see screenshots and execute actions. Respond with JSON: {\"action\": \"click|type|scroll\", \"x\": number, \"y\": number, \"text\": \"for type only\", \"reasoning\": \"why this action\"}"},{"role":"user","content":[{"type":"image","image":"screenshots/video_workspace/workspace_model_dropdown_open.png"},{"type":"text","text":"Choose model with best lip sync"}]},{"role":"assistant","content":"{\"action\": \"click\", \"x\": 300, \"y\": 240, \"reasoning\": \"Selecting Kling 2.6 which has strong lip-sync and voice alignment for dialogue\"}"}]}
```

### Motion Preset Examples

```
{"messages":[{"role":"system","content":"You are a computer use agent that automates tasks on Higgsfield AI. You see screenshots and execute actions. Respond with JSON: {\"action\": \"click|type|scroll\", \"x\": number, \"y\": number, \"text\": \"for type only\", \"reasoning\": \"why this action\"}"},{"role":"user","content":[{"type":"image","image":"screenshots/motion_presets/workspace_motion_presets_open.png"},{"type":"text","text":"Apply dramatic push in effect"}]},{"role":"assistant","content":"{\"action\": \"click\", \"x\": 200, \"y\": 200, \"reasoning\": \"Clicking Push In preset for dramatic camera movement toward subject\"}"}]}
```

```
{"messages":[{"role":"system","content":"You are a computer use agent that automates tasks on Higgsfield AI. You see screenshots and execute actions. Respond with JSON: {\"action\": \"click|type|scroll\", \"x\": number, \"y\": number, \"text\": \"for type only\", \"reasoning\": \"why this action\"}"},{"role":"user","content":[{"type":"image","image":"screenshots/motion_presets/workspace_motion_presets_open.png"},{"type":"text","text":"Add an establishing shot that reveals the scene"}]},{"role":"assistant","content":"{\"action\": \"click\", \"x\": 350, \"y\": 200, \"reasoning\": \"Clicking Pull Back preset to create a reveal effect showing wider context\"}"}]}
```

### Generation Examples

```
{"messages":[{"role":"system","content":"You are a computer use agent that automates tasks on Higgsfield AI. You see screenshots and execute actions. Respond with JSON: {\"action\": \"click|type|scroll\", \"x\": number, \"y\": number, \"text\": \"for type only\", \"reasoning\": \"why this action\"}"},{"role":"user","content":[{"type":"image","image":"screenshots/video_workspace/workspace_default_full.png"},{"type":"text","text":"Upload my reference image"}]},{"role":"assistant","content":"{\"action\": \"click\", \"x\": 640, \"y\": 400, \"reasoning\": \"Clicking Upload Image button to open file picker for reference image\"}"}]}
```

```
{"messages":[{"role":"system","content":"You are a computer use agent that automates tasks on Higgsfield AI. You see screenshots and execute actions. Respond with JSON: {\"action\": \"click|type|scroll\", \"x\": number, \"y\": number, \"text\": \"for type only\", \"reasoning\": \"why this action\"}"},{"role":"user","content":[{"type":"image","image":"screenshots/video_workspace/workspace_prompt_focused.png"},{"type":"text","text":"Type: A woman walking through a sunlit garden"}]},{"role":"assistant","content":"{\"action\": \"type\", \"x\": 640, \"y\": 500, \"text\": \"A woman walking through a sunlit garden\", \"reasoning\": \"Typing the prompt describing the desired video scene\"}"}]}
```

```
{"messages":[{"role":"system","content":"You are a computer use agent that automates tasks on Higgsfield AI. You see screenshots and execute actions. Respond with JSON: {\"action\": \"click|type|scroll\", \"x\": number, \"y\": number, \"text\": \"for type only\", \"reasoning\": \"why this action\"}"},{"role":"user","content":[{"type":"image","image":"screenshots/video_workspace/workspace_ready_to_generate.png"},{"type":"text","text":"Generate the video now"}]},{"role":"assistant","content":"{\"action\": \"click\", \"x\": 640, \"y\": 600, \"reasoning\": \"Clicking Generate button to start video creation process\"}"}]}
```

---

# PART 6: CREATING THE DPO DATASET

## What Is The DPO Dataset?

DPO (Direct Preference Optimization) teaches: "This response is BETTER than that response."

It's like grading homework - you show a good answer and a bad answer.

## The File Format

File: `dpo/train.jsonl`

```json
{
  "prompt": {
    "system": "System prompt here",
    "user_image": "path/to/screenshot.png",
    "user_text": "User instruction"
  },
  "chosen": "The BETTER response (correct action)",
  "rejected": "The WORSE response (wrong action)"
}
```

## What Makes "chosen" Better Than "rejected"?

1. **Correct element** vs wrong element
2. **Precise coordinates** vs vague/wrong coordinates
3. **Correct action type** vs wrong action type
4. **Good reasoning** vs poor/no reasoning
5. **Correct model choice** vs suboptimal model choice

## Complete DPO Examples

### Precision Examples

**Better coordinates:**
```
{"prompt":{"system":"You are a computer use agent for Higgsfield AI.","user_image":"screenshots/video_workspace/workspace_model_dropdown_open.png","user_text":"Select WAN 2.6 model"},"chosen":"{\"action\": \"click\", \"x\": 300, \"y\": 200, \"reasoning\": \"Clicking WAN 2.6 model card at its exact center position\"}","rejected":"{\"action\": \"click\", \"x\": 350, \"y\": 250, \"reasoning\": \"Clicking somewhere in the model area\"}"}
```

### Correct Element Examples

**Right button vs wrong button:**
```
{"prompt":{"system":"You are a computer use agent for Higgsfield AI.","user_image":"screenshots/login/login_default_full.png","user_text":"Sign in with Google"},"chosen":"{\"action\": \"click\", \"x\": 640, \"y\": 400, \"reasoning\": \"Clicking Google Sign In OAuth button\"}","rejected":"{\"action\": \"click\", \"x\": 640, \"y\": 480, \"reasoning\": \"Clicking Sign Up link\"}"}
```

### Correct Model Selection Examples

**Fast model when speed requested:**
```
{"prompt":{"system":"You are a computer use agent for Higgsfield AI.","user_image":"screenshots/video_workspace/workspace_model_dropdown_open.png","user_text":"I need fast iteration to test my idea quickly"},"chosen":"{\"action\": \"click\", \"x\": 300, \"y\": 280, \"reasoning\": \"Selecting Minimax Hailuo 02 which is optimized for fastest generation speed\"}","rejected":"{\"action\": \"click\", \"x\": 300, \"y\": 200, \"reasoning\": \"Selecting WAN 2.6 for high quality\"}"}
```

**Quality model when quality requested:**
```
{"prompt":{"system":"You are a computer use agent for Higgsfield AI.","user_image":"screenshots/video_workspace/workspace_model_dropdown_open.png","user_text":"I want the highest quality cinematic video"},"chosen":"{\"action\": \"click\", \"x\": 300, \"y\": 200, \"reasoning\": \"Selecting WAN 2.6 which is best for cinematic continuity and professional quality\"}","rejected":"{\"action\": \"click\", \"x\": 300, \"y\": 280, \"reasoning\": \"Selecting Minimax for fast generation\"}"}
```

### Correct Action Type Examples

**Click vs type:**
```
{"prompt":{"system":"You are a computer use agent for Higgsfield AI.","user_image":"screenshots/motion_presets/workspace_motion_presets_open.png","user_text":"Select push in motion"},"chosen":"{\"action\": \"click\", \"x\": 200, \"y\": 200, \"reasoning\": \"Clicking the Push In preset tile\"}","rejected":"{\"action\": \"type\", \"x\": 200, \"y\": 200, \"text\": \"push in\", \"reasoning\": \"Typing push in to search\"}"}
```

### Reasoning Quality Examples

**Detailed vs vague reasoning:**
```
{"prompt":{"system":"You are a computer use agent for Higgsfield AI.","user_image":"screenshots/video_workspace/workspace_model_dropdown_open.png","user_text":"Select best model for cinematic shots"},"chosen":"{\"action\": \"click\", \"x\": 300, \"y\": 200, \"reasoning\": \"Selecting WAN 2.6 because it excels at cinematic continuity, multi-shot generation, and professional camera movements - ideal for high-quality cinematic content\"}","rejected":"{\"action\": \"click\", \"x\": 300, \"y\": 200, \"reasoning\": \"Clicking model\"}"}
```

### Correct Sequence Examples

**Generate after setup vs premature generate:**
```
{"prompt":{"system":"You are a computer use agent for Higgsfield AI.","user_image":"screenshots/video_workspace/workspace_ready_to_generate.png","user_text":"Create the video"},"chosen":"{\"action\": \"click\", \"x\": 640, \"y\": 600, \"reasoning\": \"Clicking Generate button to create video - image is uploaded, prompt is filled, model is selected, ready to generate\"}","rejected":"{\"action\": \"click\", \"x\": 640, \"y\": 400, \"reasoning\": \"Clicking Upload Image again even though image is already uploaded\"}"}
```

---

# PART 7: VALIDATION CHECKLIST

Before you consider yourself DONE, check every single item:

## Screenshots Checklist

```
□ All screenshots are 1920x1080 resolution
□ All screenshots are PNG format
□ All screenshots are full browser window (not cropped)
□ All screenshots have lowercase filenames with underscores
□ All screenshots are in the correct subfolder
□ Coordinates recorded for every interactive element
□ Coordinates saved in coordinates.csv

Login screenshots:
□ login_default_full.png exists
□ login_google_hover.png exists
□ login_email_focused.png exists
□ login_password_focused.png exists
□ login_submit_hover.png exists
□ (at least 10 more variations)

Dashboard screenshots:
□ dashboard_default_full.png exists
□ dashboard_create_hover.png exists
□ dashboard_create_expanded.png exists
□ (at least 10 more variations)

Workspace screenshots:
□ workspace_default_full.png exists
□ workspace_model_dropdown_open.png exists
□ workspace_model_wan26_hover.png exists
□ workspace_model_kling_hover.png exists
□ workspace_model_minimax_hover.png exists
□ workspace_motion_presets_open.png exists
□ workspace_motion_pushin_hover.png exists
□ workspace_upload_hover.png exists
□ workspace_prompt_focused.png exists
□ workspace_generate_hover.png exists
□ (at least 25 more variations)
```

## Embedding Dataset Checklist

```
□ File exists: embedding/train.jsonl
□ File is valid JSONL (one JSON object per line)
□ Every line parses as valid JSON
□ Every example has "query" field
□ Every example has "positive" field
□ Every example has "negatives" field (array with at least 1 item)
□ Every query has "image" and "text" fields
□ Every positive has "image" and "text" fields
□ Every negative has "image", "text", and "negative_type" fields
□ All image paths point to existing files
□ At least 500 examples total
□ Each core action has at least 5 paraphrase variations
□ Negatives include different negative_types
```

## Reranker Dataset Checklist

```
□ File exists: reranker/train.jsonl
□ File is valid JSONL
□ Every example has "query", "document", "label" fields
□ Labels are only 0 or 1 (not strings, not other numbers)
□ All image paths point to existing files
□ At least 300 examples total
□ Ratio is approximately 1:3 (positive:negative)
□ Count of label=1: ____
□ Count of label=0: ____
□ Ratio check: negatives should be ~3x positives
```

## SFT Dataset Checklist

```
□ File exists: sft/train.jsonl
□ File is valid JSONL
□ Every example has "messages" array
□ Every messages array has exactly 3 items
□ First message is role="system"
□ Second message is role="user"
□ Third message is role="assistant"
□ User content is array with image and text
□ Assistant content is valid JSON string
□ Assistant JSON has "action", "x", "y", "reasoning"
□ Coordinates in assistant response match actual element position
□ All image paths point to existing files
□ At least 100 examples total
```

## DPO Dataset Checklist

```
□ File exists: dpo/train.jsonl
□ File is valid JSONL
□ Every example has "prompt", "chosen", "rejected" fields
□ Prompt has "system", "user_image", "user_text" fields
□ Chosen and rejected are both valid JSON strings
□ Chosen is actually BETTER than rejected (verify manually!)
□ All image paths point to existing files
□ At least 50 examples total
```

---

# PART 8: COMMON MISTAKES AND HOW TO AVOID THEM

## Mistake 1: Invalid JSON

**WRONG:**
```
{"query": {"image": "test.png", "text": "Click button",}}
```
(Extra comma before closing brace)

**RIGHT:**
```
{"query": {"image": "test.png", "text": "Click button"}}
```

## Mistake 2: Missing Quotes

**WRONG:**
```
{"query": {"image": test.png, "text": "Click"}}
```

**RIGHT:**
```
{"query": {"image": "test.png", "text": "Click"}}
```

## Mistake 3: Wrong File Path

**WRONG:**
```
{"image": "C:\\Users\\John\\Desktop\\screenshot.png"}
```

**RIGHT:**
```
{"image": "screenshots/login/login_default_full.png"}
```
(Use relative paths, forward slashes, no Windows paths)

## Mistake 4: Label as String

**WRONG:**
```
{"label": "1"}
```

**RIGHT:**
```
{"label": 1}
```

## Mistake 5: Coordinates as Strings

**WRONG:**
```
{"x": "640", "y": "400"}
```

**RIGHT:**
```
{"x": 640, "y": 400}
```

## Mistake 6: Newlines in JSONL

**WRONG:**
```
{
  "query": {...},
  "positive": {...}
}
```

**RIGHT:**
```
{"query":{...},"positive":{...}}
```
(Everything on ONE line)

## Mistake 7: Not Enough Variations

**WRONG:** Only one way to say each action
- "Click Google sign in" (only this one)

**RIGHT:** Multiple paraphrases
- "Click Google sign in"
- "Sign in with Google"
- "Use Google to log in"
- "Authenticate using Google"
- "Press the Google login button"

## Mistake 8: Wrong Negative Types

**WRONG:** Random unrelated negatives
- Query: "Click login button"
- Negative: Screenshot of a cat

**RIGHT:** Hard negatives that could be confused
- Query: "Click login button"
- Negative: "Sign up button on same page"
- Negative: "Cancel button near login"
- Negative: "Login button on different site"

---

# PART 9: HOW TO VALIDATE YOUR DATA

## Python Validation Script

Save this as `validate_data.py` and run it:

```python
import json
import os
from pathlib import Path

def validate_jsonl(file_path, required_fields, name):
    """Validate a JSONL file."""
    print(f"\n{'='*60}")
    print(f"Validating: {name}")
    print(f"File: {file_path}")
    print('='*60)

    if not os.path.exists(file_path):
        print(f"❌ ERROR: File does not exist!")
        return False

    errors = []
    warnings = []
    line_count = 0

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            line_count += 1

            # Try to parse JSON
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
                continue

            # Check required fields
            for field in required_fields:
                if field not in data:
                    errors.append(f"Line {line_num}: Missing field '{field}'")

    # Report
    print(f"Total examples: {line_count}")

    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for err in errors[:10]:  # Show first 10
            print(f"  {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors)-10} more errors")
        return False
    else:
        print("✅ All validations passed!")
        return True

def check_image_paths(file_path, image_fields):
    """Check if all referenced images exist."""
    print(f"\nChecking image paths in {file_path}...")

    missing = []
    checked = 0

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except:
                continue

            # Recursively find image paths
            def find_images(obj, path=""):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k == "image" and isinstance(v, str):
                            checked += 1
                            if not os.path.exists(v):
                                missing.append((line_num, v))
                        else:
                            find_images(v, f"{path}.{k}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        find_images(item, f"{path}[{i}]")

            find_images(data)

    print(f"Checked {checked} image references")
    if missing:
        print(f"❌ Missing images ({len(missing)}):")
        for line_num, path in missing[:10]:
            print(f"  Line {line_num}: {path}")
    else:
        print("✅ All images exist!")

# Run validations
print("DATA VALIDATION REPORT")
print("=" * 60)

# Embedding
validate_jsonl(
    "embedding/train.jsonl",
    ["query", "positive", "negatives"],
    "Embedding Dataset"
)

# Reranker
validate_jsonl(
    "reranker/train.jsonl",
    ["query", "document", "label"],
    "Reranker Dataset"
)

# SFT
validate_jsonl(
    "sft/train.jsonl",
    ["messages"],
    "SFT Dataset"
)

# DPO
validate_jsonl(
    "dpo/train.jsonl",
    ["prompt", "chosen", "rejected"],
    "DPO Dataset"
)

print("\n" + "=" * 60)
print("DONE")
```

Run it:
```bash
cd higgsfield_data
python validate_data.py
```

---

# PART 10: DAILY WORKFLOW

## Day 1-3: Screenshots
- Take ALL screenshots
- Record ALL coordinates
- Organize into folders
- Validate: Every screenshot exists, named correctly

## Day 4-7: Embedding Data
- Create 500+ triplets
- Focus on paraphrasing (5+ ways to say each action)
- Focus on hard negatives (similar but wrong)
- Validate: Run validation script

## Day 8-10: Reranker Data
- Create 300+ pairs
- Maintain 1:3 positive:negative ratio
- Reuse screenshots from embedding
- Validate: Check ratio, run validation

## Day 11-13: SFT Data
- Create 100+ examples
- Every example must have correct coordinates
- Test: Does the action make sense for the screenshot?
- Validate: JSON structure correct

## Day 14: DPO Data
- Create 50+ preference pairs
- Each pair: chosen is CLEARLY better than rejected
- Validate: Manually review each pair

## Day 15: Final Validation
- Run validation script on all datasets
- Fix any errors
- Double-check image paths
- Backup everything

---

# QUICK REFERENCE CARD

```
┌─────────────────────────────────────────────────────────────┐
│                    QUICK REFERENCE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  EMBEDDING (embedding/train.jsonl)                          │
│  ─────────────────────────────────                          │
│  {"query":{"image":"...","text":"..."},                    │
│   "positive":{"image":"...","text":"..."},                 │
│   "negatives":[{"image":"...","text":"...",                │
│                 "negative_type":"..."}]}                    │
│                                                             │
│  Min: 500 examples                                          │
│  Key: 5+ paraphrases per action, hard negatives            │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  RERANKER (reranker/train.jsonl)                           │
│  ─────────────────────────────────                          │
│  {"query":{"image":"...","text":"..."},                    │
│   "document":{"image":"...","text":"..."},                 │
│   "label":1}                                                │
│                                                             │
│  Min: 300 examples                                          │
│  Key: 1:3 positive:negative ratio                          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  SFT (sft/train.jsonl)                                     │
│  ─────────────────────                                      │
│  {"messages":[                                              │
│    {"role":"system","content":"..."},                      │
│    {"role":"user","content":[                              │
│      {"type":"image","image":"..."},                       │
│      {"type":"text","text":"..."}]},                       │
│    {"role":"assistant","content":"{\"action\":...}"}]}     │
│                                                             │
│  Min: 100 examples                                          │
│  Key: Exact coordinates, good reasoning                    │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  DPO (dpo/train.jsonl)                                     │
│  ─────────────────────                                      │
│  {"prompt":{"system":"...","user_image":"...",             │
│             "user_text":"..."},                            │
│   "chosen":"...",                                          │
│   "rejected":"..."}                                         │
│                                                             │
│  Min: 50 examples                                           │
│  Key: Chosen MUST be better than rejected                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

**NOW GO DO IT. EXACTLY AS I SAID. NO IMPROVISING.**
