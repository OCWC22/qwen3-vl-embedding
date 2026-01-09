#!/usr/bin/env python3
"""
Create Sample Training Data for Higgsfield AI Automation
=========================================================

This script generates the MINIMUM viable training data structure
for all 4 fine-tuning tasks. Replace placeholder images with real screenshots.

Run: python create_sample_data.py
"""

import json
import os
from pathlib import Path

# Create directories
DATA_DIR = Path("data")
SCREENSHOTS_DIR = DATA_DIR / "screenshots"
DATA_DIR.mkdir(exist_ok=True)
SCREENSHOTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# HIGGSFIELD UI ELEMENTS (Based on research)
# =============================================================================

HIGGSFIELD_ELEMENTS = {
    "login_page": {
        "url": "higgsfield.ai/auth",
        "elements": [
            {"name": "google_signin_button", "text": "Sign in with Google", "coords": (640, 400)},
            {"name": "email_input", "text": "Email address", "coords": (640, 300)},
            {"name": "password_input", "text": "Password", "coords": (640, 360)},
            {"name": "login_button", "text": "Log in", "coords": (640, 420)},
            {"name": "signup_link", "text": "Sign up", "coords": (640, 480)},
        ]
    },
    "dashboard": {
        "url": "higgsfield.ai/dashboard",
        "elements": [
            {"name": "create_button", "text": "Create", "coords": (100, 50)},
            {"name": "create_video_option", "text": "Video", "coords": (100, 100)},
            {"name": "create_image_option", "text": "Image", "coords": (100, 140)},
            {"name": "my_videos_tab", "text": "My Videos", "coords": (200, 50)},
            {"name": "credits_display", "text": "40 Credits", "coords": (1100, 50)},
        ]
    },
    "video_creation": {
        "url": "higgsfield.ai/create/video",
        "elements": [
            {"name": "model_selector", "text": "Select Model", "coords": (300, 150)},
            {"name": "model_wan26", "text": "WAN 2.6", "coords": (300, 200)},
            {"name": "model_kling26", "text": "Kling 2.6", "coords": (300, 240)},
            {"name": "model_minimax", "text": "Minimax Hailuo 02", "coords": (300, 280)},
            {"name": "motion_control", "text": "Motion Control", "coords": (600, 150)},
            {"name": "motion_change", "text": "Change", "coords": (700, 150)},
            {"name": "motion_mix", "text": "Mix", "coords": (760, 150)},
            {"name": "upload_image", "text": "Upload Image", "coords": (640, 400)},
            {"name": "prompt_input", "text": "Enter prompt...", "coords": (640, 500)},
            {"name": "generate_button", "text": "Generate", "coords": (640, 600)},
            {"name": "duration_slider", "text": "Duration", "coords": (900, 300)},
        ]
    },
    "motion_presets": {
        "url": "higgsfield.ai/create/video#motion",
        "elements": [
            {"name": "push_in", "text": "Push In", "coords": (200, 200)},
            {"name": "pull_back", "text": "Pull Back", "coords": (350, 200)},
            {"name": "pan_left", "text": "Pan Left", "coords": (500, 200)},
            {"name": "pan_right", "text": "Pan Right", "coords": (650, 200)},
            {"name": "tilt_up", "text": "Tilt Up", "coords": (200, 300)},
            {"name": "tilt_down", "text": "Tilt Down", "coords": (350, 300)},
            {"name": "dolly_zoom", "text": "Dolly Zoom", "coords": (500, 300)},
            {"name": "orbit", "text": "Orbit", "coords": (650, 300)},
        ]
    }
}

# =============================================================================
# 1. EMBEDDING TRAINING DATA
# =============================================================================

def create_embedding_data():
    """
    Create contrastive learning triplets for embedding model.

    Format: {"query": {...}, "positive": {...}, "negatives": [...]}

    MINIMUM: 500 triplets
    RECOMMENDED: 5,000 triplets
    """

    examples = []

    # ----- LOGIN WORKFLOW -----
    examples.extend([
        {
            "query": {
                "image": "screenshots/higgsfield_login_page.png",
                "text": "Click the sign in with Google button"
            },
            "positive": {
                "image": "screenshots/higgsfield_login_google_highlighted.png",
                "text": "Google Sign In button, blue OAuth button at center of login form"
            },
            "negatives": [
                {
                    "image": "screenshots/higgsfield_login_page.png",
                    "text": "Email input field for manual login",
                    "negative_type": "same_view_wrong_element"
                },
                {
                    "image": "screenshots/higgsfield_login_page.png",
                    "text": "Sign up link to create new account",
                    "negative_type": "same_view_wrong_element"
                },
                {
                    "image": "screenshots/google_signin_different_site.png",
                    "text": "Google Sign In on different website",
                    "negative_type": "similar_element_different_context"
                }
            ]
        },
        {
            "query": {
                "image": "screenshots/higgsfield_login_page.png",
                "text": "Enter my email address"
            },
            "positive": {
                "image": "screenshots/higgsfield_login_email_field.png",
                "text": "Email input field in Higgsfield login form"
            },
            "negatives": [
                {
                    "image": "screenshots/higgsfield_login_page.png",
                    "text": "Password input field",
                    "negative_type": "same_view_wrong_element"
                }
            ]
        },
    ])

    # ----- NAVIGATION WORKFLOW -----
    examples.extend([
        {
            "query": {
                "image": "screenshots/higgsfield_dashboard.png",
                "text": "Click Create to make a new video"
            },
            "positive": {
                "image": "screenshots/higgsfield_create_button.png",
                "text": "Create button in top navigation bar"
            },
            "negatives": [
                {
                    "image": "screenshots/higgsfield_dashboard.png",
                    "text": "My Videos tab showing previous creations",
                    "negative_type": "same_view_wrong_element"
                },
                {
                    "image": "screenshots/higgsfield_dashboard.png",
                    "text": "Credits display showing remaining balance",
                    "negative_type": "same_view_wrong_element"
                }
            ]
        },
        {
            "query": {
                "image": "screenshots/higgsfield_create_menu.png",
                "text": "Select Video from the create menu"
            },
            "positive": {
                "image": "screenshots/higgsfield_create_video_option.png",
                "text": "Video option in Create dropdown menu"
            },
            "negatives": [
                {
                    "image": "screenshots/higgsfield_create_menu.png",
                    "text": "Image option in Create dropdown",
                    "negative_type": "same_view_wrong_element"
                }
            ]
        },
    ])

    # ----- MODEL SELECTION WORKFLOW -----
    examples.extend([
        {
            "query": {
                "image": "screenshots/higgsfield_video_workspace.png",
                "text": "Select the WAN 2.6 model for video generation"
            },
            "positive": {
                "image": "screenshots/higgsfield_model_wan26.png",
                "text": "WAN 2.6 model option - best for cinematic continuity"
            },
            "negatives": [
                {
                    "image": "screenshots/higgsfield_model_kling.png",
                    "text": "Kling 2.6 model - best for lip sync",
                    "negative_type": "same_view_wrong_model"
                },
                {
                    "image": "screenshots/higgsfield_model_minimax.png",
                    "text": "Minimax Hailuo 02 - fastest iteration",
                    "negative_type": "same_view_wrong_model"
                }
            ]
        },
        {
            "query": {
                "image": "screenshots/higgsfield_video_workspace.png",
                "text": "Choose Kling model for better dialogue and lip sync"
            },
            "positive": {
                "image": "screenshots/higgsfield_model_kling.png",
                "text": "Kling 2.6 model option with voice alignment"
            },
            "negatives": [
                {
                    "image": "screenshots/higgsfield_model_wan26.png",
                    "text": "WAN 2.6 model",
                    "negative_type": "same_view_wrong_model"
                }
            ]
        },
        {
            "query": {
                "image": "screenshots/higgsfield_video_workspace.png",
                "text": "Pick the fastest model for quick iteration"
            },
            "positive": {
                "image": "screenshots/higgsfield_model_minimax.png",
                "text": "Minimax Hailuo 02 - optimized for speed"
            },
            "negatives": [
                {
                    "image": "screenshots/higgsfield_model_wan26.png",
                    "text": "WAN 2.6 - slower but higher quality",
                    "negative_type": "same_view_wrong_model"
                }
            ]
        },
    ])

    # ----- MOTION PRESET WORKFLOW -----
    examples.extend([
        {
            "query": {
                "image": "screenshots/higgsfield_motion_panel.png",
                "text": "Select push in camera motion"
            },
            "positive": {
                "image": "screenshots/higgsfield_motion_push_in.png",
                "text": "Push In motion preset - camera moves toward subject"
            },
            "negatives": [
                {
                    "image": "screenshots/higgsfield_motion_pull_back.png",
                    "text": "Pull Back motion - camera moves away",
                    "negative_type": "opposite_motion"
                },
                {
                    "image": "screenshots/higgsfield_motion_pan.png",
                    "text": "Pan motion - horizontal movement",
                    "negative_type": "different_motion_type"
                }
            ]
        },
        {
            "query": {
                "image": "screenshots/higgsfield_motion_panel.png",
                "text": "Apply orbit camera movement around subject"
            },
            "positive": {
                "image": "screenshots/higgsfield_motion_orbit.png",
                "text": "Orbit motion preset - 360 degree rotation"
            },
            "negatives": [
                {
                    "image": "screenshots/higgsfield_motion_dolly.png",
                    "text": "Dolly zoom effect",
                    "negative_type": "different_motion_type"
                }
            ]
        },
    ])

    # ----- IMAGE UPLOAD & GENERATION -----
    examples.extend([
        {
            "query": {
                "image": "screenshots/higgsfield_video_workspace.png",
                "text": "Upload an image for video generation"
            },
            "positive": {
                "image": "screenshots/higgsfield_upload_button.png",
                "text": "Upload Image button in workspace"
            },
            "negatives": [
                {
                    "image": "screenshots/higgsfield_video_workspace.png",
                    "text": "Generate button to start creation",
                    "negative_type": "same_view_wrong_element"
                }
            ]
        },
        {
            "query": {
                "image": "screenshots/higgsfield_video_ready.png",
                "text": "Click generate to create the video"
            },
            "positive": {
                "image": "screenshots/higgsfield_generate_button.png",
                "text": "Generate button - starts video creation"
            },
            "negatives": [
                {
                    "image": "screenshots/higgsfield_video_workspace.png",
                    "text": "Upload Image button",
                    "negative_type": "same_view_wrong_element"
                }
            ]
        },
    ])

    # Save to file
    output_path = DATA_DIR / "embedding_train.jsonl"
    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    print(f"Created {len(examples)} embedding examples -> {output_path}")
    print(f"  MINIMUM NEEDED: 500 (you have {len(examples)}, need {max(0, 500-len(examples))} more)")

    return examples


# =============================================================================
# 2. RERANKER TRAINING DATA
# =============================================================================

def create_reranker_data():
    """
    Create binary classification pairs for reranker model.

    Format: {"query": {...}, "document": {...}, "label": 0|1}

    MINIMUM: 300 pairs (1:3 positive:negative ratio)
    RECOMMENDED: 3,000 pairs
    """

    examples = []

    # ----- POSITIVE EXAMPLES (label=1) -----
    positive_examples = [
        # Login
        {
            "query": {
                "image": "screenshots/higgsfield_login_page.png",
                "text": "Click Google sign in button"
            },
            "document": {
                "image": "screenshots/higgsfield_google_button.png",
                "text": "Google OAuth button, click at (640, 400)"
            },
            "label": 1
        },
        # Model selection
        {
            "query": {
                "image": "screenshots/higgsfield_model_selector.png",
                "text": "Select WAN 2.6 model"
            },
            "document": {
                "image": "screenshots/higgsfield_wan26_option.png",
                "text": "WAN 2.6 model card, click at (300, 200)"
            },
            "label": 1
        },
        {
            "query": {
                "image": "screenshots/higgsfield_model_selector.png",
                "text": "Choose fastest model for quick preview"
            },
            "document": {
                "image": "screenshots/higgsfield_minimax_option.png",
                "text": "Minimax Hailuo 02 - fast iteration, click at (300, 280)"
            },
            "label": 1
        },
        # Motion
        {
            "query": {
                "image": "screenshots/higgsfield_motion_panel.png",
                "text": "Apply push in camera motion"
            },
            "document": {
                "image": "screenshots/higgsfield_push_in_preset.png",
                "text": "Push In motion preset tile, click at (200, 200)"
            },
            "label": 1
        },
        # Generate
        {
            "query": {
                "image": "screenshots/higgsfield_ready_to_generate.png",
                "text": "Start generating the video"
            },
            "document": {
                "image": "screenshots/higgsfield_generate_btn.png",
                "text": "Generate button, purple CTA, click at (640, 600)"
            },
            "label": 1
        },
    ]

    # ----- NEGATIVE EXAMPLES (label=0) -----
    negative_examples = [
        # Wrong element same view
        {
            "query": {
                "image": "screenshots/higgsfield_login_page.png",
                "text": "Click Google sign in button"
            },
            "document": {
                "image": "screenshots/higgsfield_email_field.png",
                "text": "Email input field, not a button"
            },
            "label": 0
        },
        {
            "query": {
                "image": "screenshots/higgsfield_login_page.png",
                "text": "Click Google sign in button"
            },
            "document": {
                "image": "screenshots/higgsfield_signup_link.png",
                "text": "Sign up link for new accounts"
            },
            "label": 0
        },
        # Wrong model
        {
            "query": {
                "image": "screenshots/higgsfield_model_selector.png",
                "text": "Select WAN 2.6 model"
            },
            "document": {
                "image": "screenshots/higgsfield_kling_option.png",
                "text": "Kling 2.6 model - wrong model selected"
            },
            "label": 0
        },
        {
            "query": {
                "image": "screenshots/higgsfield_model_selector.png",
                "text": "Select WAN 2.6 model"
            },
            "document": {
                "image": "screenshots/higgsfield_minimax_option.png",
                "text": "Minimax model - wrong model selected"
            },
            "label": 0
        },
        # Wrong motion
        {
            "query": {
                "image": "screenshots/higgsfield_motion_panel.png",
                "text": "Apply push in camera motion"
            },
            "document": {
                "image": "screenshots/higgsfield_pull_back_preset.png",
                "text": "Pull Back motion - opposite direction"
            },
            "label": 0
        },
        {
            "query": {
                "image": "screenshots/higgsfield_motion_panel.png",
                "text": "Apply push in camera motion"
            },
            "document": {
                "image": "screenshots/higgsfield_pan_preset.png",
                "text": "Pan motion - horizontal not forward"
            },
            "label": 0
        },
        # Wrong action
        {
            "query": {
                "image": "screenshots/higgsfield_ready_to_generate.png",
                "text": "Start generating the video"
            },
            "document": {
                "image": "screenshots/higgsfield_upload_btn.png",
                "text": "Upload button - wrong action"
            },
            "label": 0
        },
        # Similar but different site
        {
            "query": {
                "image": "screenshots/higgsfield_login_page.png",
                "text": "Click Google sign in button"
            },
            "document": {
                "image": "screenshots/runway_google_signin.png",
                "text": "Google signin on Runway - wrong site"
            },
            "label": 0
        },
    ]

    examples = positive_examples + negative_examples

    # Save to file
    output_path = DATA_DIR / "reranker_train.jsonl"
    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    pos_count = sum(1 for e in examples if e['label'] == 1)
    neg_count = len(examples) - pos_count

    print(f"Created {len(examples)} reranker examples -> {output_path}")
    print(f"  Positive: {pos_count}, Negative: {neg_count}")
    print(f"  Ratio: 1:{neg_count/pos_count:.1f}")
    print(f"  MINIMUM NEEDED: 300 (you have {len(examples)}, need {max(0, 300-len(examples))} more)")

    return examples


# =============================================================================
# 3. SFT TRAINING DATA
# =============================================================================

def create_sft_data():
    """
    Create supervised fine-tuning data for base generation model.

    Format: {"messages": [{"role": "system/user/assistant", "content": [...]}]}

    MINIMUM: 100 examples
    RECOMMENDED: 1,000 examples
    """

    examples = []

    SYSTEM_PROMPT = """You are a computer use agent that helps users automate tasks on Higgsfield AI video generator.
You can see the user's screen and execute actions like click, type, and scroll.
Always respond with a JSON action in the following format:
{"action": "click|type|scroll", "x": int, "y": int, "text": "optional for type", "reasoning": "brief explanation"}"""

    # ----- LOGIN WORKFLOW -----
    examples.extend([
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "screenshots/higgsfield_login_page.png"},
                        {"type": "text", "text": "Sign in to Higgsfield using Google"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": '{"action": "click", "x": 640, "y": 400, "reasoning": "Clicking the Google Sign In button to authenticate with OAuth"}'
                }
            ]
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "screenshots/higgsfield_login_page.png"},
                        {"type": "text", "text": "I want to login with email instead"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": '{"action": "click", "x": 640, "y": 300, "reasoning": "Clicking the email input field to enter credentials manually"}'
                }
            ]
        },
    ])

    # ----- NAVIGATION -----
    examples.extend([
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "screenshots/higgsfield_dashboard.png"},
                        {"type": "text", "text": "I want to create a new video"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": '{"action": "click", "x": 100, "y": 50, "reasoning": "Clicking Create button in top nav to open creation menu"}'
                }
            ]
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "screenshots/higgsfield_create_menu_open.png"},
                        {"type": "text", "text": "Select video option"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": '{"action": "click", "x": 100, "y": 100, "reasoning": "Clicking Video option from dropdown to enter video workspace"}'
                }
            ]
        },
    ])

    # ----- MODEL SELECTION -----
    examples.extend([
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "screenshots/higgsfield_video_workspace.png"},
                        {"type": "text", "text": "Select WAN 2.6 model for cinematic quality"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": '{"action": "click", "x": 300, "y": 200, "reasoning": "Selecting WAN 2.6 model which is best for cinematic continuity and scene coverage"}'
                }
            ]
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "screenshots/higgsfield_video_workspace.png"},
                        {"type": "text", "text": "I need the fastest model to test my idea quickly"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": '{"action": "click", "x": 300, "y": 280, "reasoning": "Selecting Minimax Hailuo 02 which is optimized for fast iteration and quick previews"}'
                }
            ]
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "screenshots/higgsfield_video_workspace.png"},
                        {"type": "text", "text": "Choose model with best lip sync for dialogue"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": '{"action": "click", "x": 300, "y": 240, "reasoning": "Selecting Kling 2.6 which has strong lip-sync and voice alignment capabilities"}'
                }
            ]
        },
    ])

    # ----- MOTION PRESETS -----
    examples.extend([
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "screenshots/higgsfield_motion_panel.png"},
                        {"type": "text", "text": "Apply a dramatic push in effect"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": '{"action": "click", "x": 200, "y": 200, "reasoning": "Clicking Push In preset for dramatic camera movement toward subject"}'
                }
            ]
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "screenshots/higgsfield_motion_panel.png"},
                        {"type": "text", "text": "I want an establishing shot that reveals the scene"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": '{"action": "click", "x": 350, "y": 200, "reasoning": "Clicking Pull Back preset to create a reveal effect showing wider context"}'
                }
            ]
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "screenshots/higgsfield_motion_panel.png"},
                        {"type": "text", "text": "Add an orbit movement to show all angles"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": '{"action": "click", "x": 650, "y": 300, "reasoning": "Clicking Orbit preset for 360-degree rotation around subject"}'
                }
            ]
        },
    ])

    # ----- IMAGE UPLOAD & GENERATION -----
    examples.extend([
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "screenshots/higgsfield_video_workspace.png"},
                        {"type": "text", "text": "Upload my reference image"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": '{"action": "click", "x": 640, "y": 400, "reasoning": "Clicking Upload Image button to select reference image from file system"}'
                }
            ]
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "screenshots/higgsfield_prompt_area.png"},
                        {"type": "text", "text": "Enter prompt: A woman walking through a garden"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": '{"action": "click", "x": 640, "y": 500, "reasoning": "Clicking prompt input field to enter the generation prompt"}'
                }
            ]
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "screenshots/higgsfield_prompt_focused.png"},
                        {"type": "text", "text": "Type: A woman walking through a garden, golden hour lighting, cinematic"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": '{"action": "type", "x": 640, "y": 500, "text": "A woman walking through a garden, golden hour lighting, cinematic", "reasoning": "Typing the prompt describing the desired video scene"}'
                }
            ]
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "screenshots/higgsfield_ready_to_generate.png"},
                        {"type": "text", "text": "Generate the video now"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": '{"action": "click", "x": 640, "y": 600, "reasoning": "Clicking Generate button to start video creation process"}'
                }
            ]
        },
    ])

    # Save to file
    output_path = DATA_DIR / "sft_train.jsonl"
    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    print(f"Created {len(examples)} SFT examples -> {output_path}")
    print(f"  MINIMUM NEEDED: 100 (you have {len(examples)}, need {max(0, 100-len(examples))} more)")

    return examples


# =============================================================================
# 4. DPO/RL TRAINING DATA
# =============================================================================

def create_dpo_data():
    """
    Create preference pairs for DPO (Direct Preference Optimization).
    This is a simpler alternative to online RL like PPO.

    Format: {"prompt": {...}, "chosen": "...", "rejected": "..."}

    MINIMUM: 50 pairs
    RECOMMENDED: 500 pairs
    """

    examples = []

    SYSTEM_PROMPT = """You are a computer use agent for Higgsfield AI."""

    # ----- PRECISION PREFERENCES -----
    examples.extend([
        # Prefer precise coordinates over vague ones
        {
            "prompt": {
                "system": SYSTEM_PROMPT,
                "user_image": "screenshots/higgsfield_video_workspace.png",
                "user_text": "Click the WAN 2.6 model option"
            },
            "chosen": '{"action": "click", "x": 300, "y": 200, "reasoning": "Clicking WAN 2.6 model card at its center position"}',
            "rejected": '{"action": "click", "x": 350, "y": 250, "reasoning": "Clicking somewhere in the model selection area"}'
        },
        # Prefer correct model for task
        {
            "prompt": {
                "system": SYSTEM_PROMPT,
                "user_image": "screenshots/higgsfield_model_selector.png",
                "user_text": "I need fast iteration to test my idea"
            },
            "chosen": '{"action": "click", "x": 300, "y": 280, "reasoning": "Selecting Minimax Hailuo 02 for fastest generation speed"}',
            "rejected": '{"action": "click", "x": 300, "y": 200, "reasoning": "Selecting WAN 2.6 for high quality"}'
        },
    ])

    # ----- CORRECT ACTION TYPE PREFERENCES -----
    examples.extend([
        # Prefer click over type when selecting
        {
            "prompt": {
                "system": SYSTEM_PROMPT,
                "user_image": "screenshots/higgsfield_motion_panel.png",
                "user_text": "Select push in motion"
            },
            "chosen": '{"action": "click", "x": 200, "y": 200, "reasoning": "Clicking the Push In preset tile"}',
            "rejected": '{"action": "type", "x": 200, "y": 200, "text": "push in", "reasoning": "Typing push in to search"}'
        },
        # Prefer type for text input
        {
            "prompt": {
                "system": SYSTEM_PROMPT,
                "user_image": "screenshots/higgsfield_prompt_focused.png",
                "user_text": "Enter the prompt for video generation"
            },
            "chosen": '{"action": "type", "x": 640, "y": 500, "text": "A cinematic scene", "reasoning": "Typing in the focused prompt field"}',
            "rejected": '{"action": "click", "x": 640, "y": 600, "reasoning": "Clicking generate without entering prompt"}'
        },
    ])

    # ----- CORRECT ELEMENT PREFERENCES -----
    examples.extend([
        # Prefer correct button over similar-looking one
        {
            "prompt": {
                "system": SYSTEM_PROMPT,
                "user_image": "screenshots/higgsfield_login_page.png",
                "user_text": "Login with Google"
            },
            "chosen": '{"action": "click", "x": 640, "y": 400, "reasoning": "Clicking Google Sign In OAuth button"}',
            "rejected": '{"action": "click", "x": 640, "y": 480, "reasoning": "Clicking Sign Up link"}'
        },
        # Prefer generate over upload when ready
        {
            "prompt": {
                "system": SYSTEM_PROMPT,
                "user_image": "screenshots/higgsfield_ready_to_generate.png",
                "user_text": "Create the video"
            },
            "chosen": '{"action": "click", "x": 640, "y": 600, "reasoning": "Clicking Generate to create video"}',
            "rejected": '{"action": "click", "x": 640, "y": 400, "reasoning": "Clicking Upload Image again"}'
        },
    ])

    # ----- REASONING QUALITY PREFERENCES -----
    examples.extend([
        # Prefer detailed reasoning
        {
            "prompt": {
                "system": SYSTEM_PROMPT,
                "user_image": "screenshots/higgsfield_model_selector.png",
                "user_text": "Select best model for cinematic shots"
            },
            "chosen": '{"action": "click", "x": 300, "y": 200, "reasoning": "Selecting WAN 2.6 because it excels at cinematic continuity, multi-shot generation, and professional camera movements"}',
            "rejected": '{"action": "click", "x": 300, "y": 200, "reasoning": "Clicking model"}'
        },
    ])

    # Save to file
    output_path = DATA_DIR / "dpo_train.jsonl"
    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    print(f"Created {len(examples)} DPO examples -> {output_path}")
    print(f"  MINIMUM NEEDED: 50 (you have {len(examples)}, need {max(0, 50-len(examples))} more)")

    return examples


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("CREATING MINIMAL TRAINING DATA FOR HIGGSFIELD AUTOMATION")
    print("=" * 60)
    print()

    # Create all datasets
    create_embedding_data()
    print()
    create_reranker_data()
    print()
    create_sft_data()
    print()
    create_dpo_data()

    print()
    print("=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("""
1. CAPTURE REAL SCREENSHOTS:
   - Take screenshots of each Higgsfield page/state
   - Save to data/screenshots/ with matching filenames
   - Use consistent resolution (1920x1080 recommended)

2. EXPAND DATASET:
   - Copy examples and modify for variations
   - Add edge cases (errors, loading states, popups)
   - Use LLM to paraphrase queries

3. TRAIN MODELS:
   python train_embedding.py    # ~2-4 hours
   python train_reranker.py     # ~1-2 hours
   python train_sft.py          # ~4-8 hours
   python train_dpo.py          # ~2-4 hours (optional)
""")


if __name__ == "__main__":
    main()
