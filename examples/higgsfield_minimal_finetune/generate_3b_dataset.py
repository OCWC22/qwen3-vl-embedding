#!/usr/bin/env python3
"""
COMPLETE DATASET GENERATOR FOR 3B MODEL
======================================

This script generates sample training data in the EXACT format required.
Run this to see what your data should look like.

For production: Replace the placeholder screenshots with real ones.
"""

import json
import os
from pathlib import Path

# =============================================================================
# CONFIGURATION - These are the EXACT minimums for 3B model
# =============================================================================

MINIMUM_EMBEDDING_TRIPLETS = 250
MINIMUM_RERANKER_PAIRS = 150  # Must be 75 positive + 75 negative
MINIMUM_SFT_CONVERSATIONS = 50
MINIMUM_DPO_PAIRS = 25

# Validation split (10%)
VALIDATION_RATIO = 0.1

# =============================================================================
# HIGGSFIELD UI KNOWLEDGE BASE
# This is what the intern needs to know about the Higgsfield UI
# =============================================================================

HIGGSFIELD_UI = {
    "auth": {
        "screens": [
            {
                "name": "login_empty",
                "filename": "auth_login_empty_01.png",
                "description": "Login page with empty email and password fields",
                "elements": {
                    "email_field": {"x": 512, "y": 300, "type": "input", "placeholder": "Enter your email"},
                    "password_field": {"x": 512, "y": 360, "type": "input", "placeholder": "Enter your password"},
                    "login_button": {"x": 512, "y": 420, "type": "button", "text": "Log In"},
                    "signup_link": {"x": 512, "y": 480, "type": "link", "text": "Sign Up"},
                    "forgot_password": {"x": 512, "y": 500, "type": "link", "text": "Forgot Password?"},
                    "google_oauth": {"x": 512, "y": 540, "type": "button", "text": "Continue with Google"},
                }
            },
            {
                "name": "login_filled",
                "filename": "auth_login_filled_01.png",
                "description": "Login page with email and password entered",
                "elements": {
                    "email_field": {"x": 512, "y": 300, "type": "input", "value": "user@example.com"},
                    "password_field": {"x": 512, "y": 360, "type": "input", "value": "••••••••"},
                    "login_button": {"x": 512, "y": 420, "type": "button", "text": "Log In"},
                }
            },
            {
                "name": "signup_empty",
                "filename": "auth_signup_empty_01.png",
                "description": "Sign up page with empty registration form",
                "elements": {
                    "name_field": {"x": 512, "y": 260, "type": "input", "placeholder": "Full Name"},
                    "email_field": {"x": 512, "y": 320, "type": "input", "placeholder": "Email"},
                    "password_field": {"x": 512, "y": 380, "type": "input", "placeholder": "Password"},
                    "confirm_field": {"x": 512, "y": 440, "type": "input", "placeholder": "Confirm Password"},
                    "signup_button": {"x": 512, "y": 500, "type": "button", "text": "Create Account"},
                    "login_link": {"x": 512, "y": 540, "type": "link", "text": "Already have an account?"},
                }
            },
            {
                "name": "login_error",
                "filename": "auth_login_error_01.png",
                "description": "Login page showing invalid credentials error",
                "elements": {
                    "error_message": {"x": 512, "y": 250, "type": "text", "text": "Invalid email or password"},
                    "email_field": {"x": 512, "y": 300, "type": "input"},
                    "password_field": {"x": 512, "y": 360, "type": "input"},
                    "login_button": {"x": 512, "y": 420, "type": "button", "text": "Try Again"},
                }
            },
        ]
    },
    "dashboard": {
        "screens": [
            {
                "name": "empty",
                "filename": "dash_empty_01.png",
                "description": "Dashboard with no projects, shows welcome message",
                "elements": {
                    "welcome_text": {"x": 512, "y": 200, "type": "text", "text": "Welcome to Higgsfield!"},
                    "new_project_button": {"x": 512, "y": 350, "type": "button", "text": "Create Your First Video"},
                    "nav_menu": {"x": 50, "y": 300, "type": "nav"},
                    "profile_icon": {"x": 950, "y": 40, "type": "button"},
                    "settings_gear": {"x": 900, "y": 40, "type": "button"},
                }
            },
            {
                "name": "with_projects",
                "filename": "dash_projects_01.png",
                "description": "Dashboard showing previous video projects",
                "elements": {
                    "project_1": {"x": 250, "y": 250, "type": "card", "title": "Cat Video"},
                    "project_2": {"x": 500, "y": 250, "type": "card", "title": "Product Demo"},
                    "project_3": {"x": 750, "y": 250, "type": "card", "title": "Marketing Clip"},
                    "new_project_button": {"x": 900, "y": 150, "type": "button", "text": "+ New"},
                    "search_bar": {"x": 512, "y": 100, "type": "input", "placeholder": "Search projects"},
                }
            },
            {
                "name": "nav_open",
                "filename": "dash_nav_open_01.png",
                "description": "Dashboard with side navigation expanded",
                "elements": {
                    "nav_home": {"x": 100, "y": 150, "type": "nav_item", "text": "Home"},
                    "nav_projects": {"x": 100, "y": 200, "type": "nav_item", "text": "My Projects"},
                    "nav_templates": {"x": 100, "y": 250, "type": "nav_item", "text": "Templates"},
                    "nav_settings": {"x": 100, "y": 300, "type": "nav_item", "text": "Settings"},
                    "nav_help": {"x": 100, "y": 350, "type": "nav_item", "text": "Help"},
                }
            },
        ]
    },
    "model": {
        "screens": [
            {
                "name": "selector_closed",
                "filename": "model_closed_01.png",
                "description": "Model selector dropdown is closed",
                "elements": {
                    "model_dropdown": {"x": 400, "y": 150, "type": "dropdown", "text": "Select Model ▼"},
                    "workspace_area": {"x": 512, "y": 400, "type": "area"},
                }
            },
            {
                "name": "selector_open",
                "filename": "model_open_01.png",
                "description": "Model selector dropdown is open showing all models",
                "elements": {
                    "model_dropdown": {"x": 400, "y": 150, "type": "dropdown", "text": "Select Model ▲"},
                    "option_animate": {"x": 400, "y": 200, "type": "option", "text": "Animate"},
                    "option_img2vid": {"x": 400, "y": 240, "type": "option", "text": "Image to Video"},
                    "option_txt2vid": {"x": 400, "y": 280, "type": "option", "text": "Text to Video"},
                    "option_faceswap": {"x": 400, "y": 320, "type": "option", "text": "Face Swap"},
                    "option_style": {"x": 400, "y": 360, "type": "option", "text": "Style Transfer"},
                }
            },
            {
                "name": "animate_selected",
                "filename": "model_animate_01.png",
                "description": "Animate model is selected and ready",
                "elements": {
                    "model_dropdown": {"x": 400, "y": 150, "type": "dropdown", "text": "Animate ▼"},
                    "model_description": {"x": 600, "y": 150, "type": "text", "text": "Transform still images into animated videos"},
                    "upload_area": {"x": 512, "y": 400, "type": "dropzone"},
                }
            },
        ]
    },
    "upload": {
        "screens": [
            {
                "name": "empty",
                "filename": "upload_empty_01.png",
                "description": "Upload area showing drag-drop zone with no file",
                "elements": {
                    "dropzone": {"x": 512, "y": 350, "type": "dropzone", "text": "Drag image here or click to upload"},
                    "upload_button": {"x": 512, "y": 420, "type": "button", "text": "Upload Image"},
                    "supported_formats": {"x": 512, "y": 480, "type": "text", "text": "PNG, JPG, WebP up to 10MB"},
                }
            },
            {
                "name": "drag_hover",
                "filename": "upload_hover_01.png",
                "description": "File is being dragged over the upload area",
                "elements": {
                    "dropzone_active": {"x": 512, "y": 350, "type": "dropzone", "text": "Release to upload", "state": "active"},
                }
            },
            {
                "name": "uploading",
                "filename": "upload_progress_01.png",
                "description": "File is currently being uploaded",
                "elements": {
                    "progress_bar": {"x": 512, "y": 350, "type": "progress", "value": 65},
                    "progress_text": {"x": 512, "y": 400, "type": "text", "text": "Uploading... 65%"},
                    "cancel_button": {"x": 512, "y": 450, "type": "button", "text": "Cancel"},
                }
            },
            {
                "name": "complete",
                "filename": "upload_complete_01.png",
                "description": "Image successfully uploaded and showing preview",
                "elements": {
                    "image_preview": {"x": 512, "y": 300, "type": "image", "alt": "Uploaded image preview"},
                    "remove_button": {"x": 700, "y": 200, "type": "button", "text": "✕"},
                    "replace_button": {"x": 512, "y": 500, "type": "button", "text": "Replace Image"},
                    "continue_button": {"x": 512, "y": 550, "type": "button", "text": "Continue"},
                }
            },
        ]
    },
    "generation": {
        "screens": [
            {
                "name": "settings_panel",
                "filename": "gen_settings_01.png",
                "description": "Generation settings panel with all options",
                "elements": {
                    "duration_slider": {"x": 700, "y": 200, "type": "slider", "label": "Duration", "value": "5s"},
                    "fps_dropdown": {"x": 700, "y": 260, "type": "dropdown", "label": "FPS", "value": "24"},
                    "resolution_dropdown": {"x": 700, "y": 320, "type": "dropdown", "label": "Resolution", "value": "720p"},
                    "motion_slider": {"x": 700, "y": 380, "type": "slider", "label": "Motion", "value": 5},
                    "seed_input": {"x": 700, "y": 440, "type": "input", "label": "Seed", "value": "random"},
                    "generate_button": {"x": 700, "y": 520, "type": "button", "text": "Generate Video"},
                }
            },
            {
                "name": "ready",
                "filename": "gen_button_01.png",
                "description": "Everything configured, generate button prominent",
                "elements": {
                    "image_preview": {"x": 300, "y": 350, "type": "image"},
                    "settings_summary": {"x": 700, "y": 300, "type": "text", "text": "5s • 24fps • 720p"},
                    "generate_button": {"x": 700, "y": 500, "type": "button", "text": "Generate Video", "style": "primary"},
                    "save_preset": {"x": 700, "y": 560, "type": "button", "text": "Save as Preset"},
                }
            },
            {
                "name": "generating_25",
                "filename": "gen_25_01.png",
                "description": "Video generation at 25% progress",
                "elements": {
                    "progress_bar": {"x": 512, "y": 350, "type": "progress", "value": 25},
                    "progress_text": {"x": 512, "y": 400, "type": "text", "text": "Generating... 25%"},
                    "eta_text": {"x": 512, "y": 430, "type": "text", "text": "Estimated: 45 seconds"},
                    "cancel_button": {"x": 512, "y": 500, "type": "button", "text": "Cancel"},
                }
            },
            {
                "name": "generating_75",
                "filename": "gen_75_01.png",
                "description": "Video generation at 75% progress",
                "elements": {
                    "progress_bar": {"x": 512, "y": 350, "type": "progress", "value": 75},
                    "progress_text": {"x": 512, "y": 400, "type": "text", "text": "Generating... 75%"},
                    "preview_frame": {"x": 512, "y": 250, "type": "image", "alt": "Preview frame"},
                    "eta_text": {"x": 512, "y": 430, "type": "text", "text": "Almost done..."},
                }
            },
            {
                "name": "complete",
                "filename": "gen_complete_01.png",
                "description": "Generation complete, success message shown",
                "elements": {
                    "success_icon": {"x": 512, "y": 200, "type": "icon", "name": "checkmark"},
                    "success_text": {"x": 512, "y": 250, "type": "text", "text": "Video Generated Successfully!"},
                    "view_button": {"x": 450, "y": 350, "type": "button", "text": "View Video"},
                    "download_button": {"x": 574, "y": 350, "type": "button", "text": "Download"},
                    "new_video_button": {"x": 512, "y": 420, "type": "button", "text": "Create Another"},
                }
            },
            {
                "name": "error",
                "filename": "gen_error_01.png",
                "description": "Generation failed with error message",
                "elements": {
                    "error_icon": {"x": 512, "y": 200, "type": "icon", "name": "error"},
                    "error_text": {"x": 512, "y": 250, "type": "text", "text": "Generation Failed"},
                    "error_details": {"x": 512, "y": 300, "type": "text", "text": "The server timed out. Please try again."},
                    "retry_button": {"x": 512, "y": 400, "type": "button", "text": "Try Again"},
                    "report_button": {"x": 512, "y": 450, "type": "button", "text": "Report Issue"},
                }
            },
        ]
    },
    "results": {
        "screens": [
            {
                "name": "player",
                "filename": "result_player_01.png",
                "description": "Video player showing generated video",
                "elements": {
                    "video_player": {"x": 512, "y": 300, "type": "video"},
                    "play_button": {"x": 512, "y": 300, "type": "button", "text": "▶"},
                    "progress_bar": {"x": 512, "y": 450, "type": "slider"},
                    "time_display": {"x": 100, "y": 450, "type": "text", "text": "0:00 / 0:05"},
                    "volume_button": {"x": 850, "y": 450, "type": "button"},
                    "fullscreen_button": {"x": 920, "y": 450, "type": "button"},
                    "download_button": {"x": 600, "y": 520, "type": "button", "text": "Download"},
                    "share_button": {"x": 700, "y": 520, "type": "button", "text": "Share"},
                    "regenerate_button": {"x": 800, "y": 520, "type": "button", "text": "Regenerate"},
                }
            },
            {
                "name": "download_options",
                "filename": "result_download_opts_01.png",
                "description": "Download options modal showing format selection",
                "elements": {
                    "modal_title": {"x": 512, "y": 200, "type": "text", "text": "Download Video"},
                    "format_mp4": {"x": 400, "y": 280, "type": "radio", "text": "MP4 (recommended)", "selected": True},
                    "format_webm": {"x": 400, "y": 320, "type": "radio", "text": "WebM"},
                    "format_gif": {"x": 400, "y": 360, "type": "radio", "text": "GIF (no audio)"},
                    "quality_high": {"x": 600, "y": 280, "type": "radio", "text": "High (720p)"},
                    "quality_low": {"x": 600, "y": 320, "type": "radio", "text": "Low (480p)"},
                    "download_button": {"x": 512, "y": 450, "type": "button", "text": "Download Now"},
                    "cancel_button": {"x": 512, "y": 500, "type": "button", "text": "Cancel"},
                }
            },
            {
                "name": "share_modal",
                "filename": "result_share_01.png",
                "description": "Share options showing link and social media",
                "elements": {
                    "modal_title": {"x": 512, "y": 200, "type": "text", "text": "Share Video"},
                    "link_input": {"x": 512, "y": 280, "type": "input", "value": "https://higgsfield.ai/v/abc123"},
                    "copy_button": {"x": 700, "y": 280, "type": "button", "text": "Copy"},
                    "twitter_button": {"x": 400, "y": 360, "type": "button", "text": "Twitter"},
                    "facebook_button": {"x": 512, "y": 360, "type": "button", "text": "Facebook"},
                    "linkedin_button": {"x": 624, "y": 360, "type": "button", "text": "LinkedIn"},
                    "close_button": {"x": 512, "y": 450, "type": "button", "text": "Close"},
                }
            },
        ]
    },
}

# =============================================================================
# DATA GENERATORS
# =============================================================================

def generate_embedding_triplets(count: int) -> list:
    """
    Generate embedding triplets for training.

    Each triplet has:
    - query: What the user sees and wants to do
    - positive: The correct action for this situation
    - negative: A wrong but similar-looking action (HARD negative)
    """
    triplets = []

    # Define triplet templates for each screen
    templates = [
        # AUTH - Login
        {
            "query_screen": "auth_login_empty_01.png",
            "query_text": "I'm on the Higgsfield login page. The email and password fields are empty. I need to log in with my credentials.",
            "positive_text": "Click on the email input field in the center of the form. It has placeholder text 'Enter your email'. After clicking, you can type your email address.",
            "positive_screen": "auth_login_empty_01.png",
            "negative_text": "Click on the 'Sign Up' link below the login form to create a new account. This is for users who don't have an account yet.",
            "negative_screen": "auth_signup_empty_01.png",
        },
        {
            "query_screen": "auth_login_filled_01.png",
            "query_text": "I've entered my email and password on the login page. Both fields are filled in. I want to complete the login.",
            "positive_text": "Click the blue 'Log In' button below the password field to submit your credentials and access your account.",
            "positive_screen": "auth_login_filled_01.png",
            "negative_text": "Click 'Forgot Password?' link to reset your password. This opens the password recovery flow.",
            "negative_screen": "auth_login_error_01.png",
        },
        {
            "query_screen": "auth_login_error_01.png",
            "query_text": "The login failed and I see an error message saying 'Invalid email or password'. I need to try again.",
            "positive_text": "Clear the password field and re-enter your password carefully, then click 'Try Again' button.",
            "positive_screen": "auth_login_error_01.png",
            "negative_text": "Click 'Create Account' to sign up for a new account instead of logging into an existing one.",
            "negative_screen": "auth_signup_empty_01.png",
        },
        # AUTH - Signup
        {
            "query_screen": "auth_signup_empty_01.png",
            "query_text": "I want to create a new Higgsfield account. I'm on the signup page with empty fields.",
            "positive_text": "Start by clicking the 'Full Name' input field at the top of the form and enter your name.",
            "positive_screen": "auth_signup_empty_01.png",
            "negative_text": "Click 'Already have an account?' link at the bottom to go to the login page instead.",
            "negative_screen": "auth_login_empty_01.png",
        },
        # DASHBOARD
        {
            "query_screen": "dash_empty_01.png",
            "query_text": "I just logged in and see the dashboard with no projects. I want to create my first video.",
            "positive_text": "Click the large 'Create Your First Video' button in the center of the page to start a new project.",
            "positive_screen": "dash_empty_01.png",
            "negative_text": "Click the profile icon in the top right to view your account settings.",
            "negative_screen": "dash_nav_open_01.png",
        },
        {
            "query_screen": "dash_projects_01.png",
            "query_text": "I'm on the dashboard and can see my previous projects. I want to create a new video.",
            "positive_text": "Click the '+ New' button in the top right area to start a new video project.",
            "positive_screen": "dash_projects_01.png",
            "negative_text": "Click on an existing project card to open and edit that previous video.",
            "negative_screen": "dash_projects_01.png",
        },
        {
            "query_screen": "dash_projects_01.png",
            "query_text": "I want to find a specific project I worked on before. I see the dashboard with multiple projects.",
            "positive_text": "Click the search bar at the top and type the project name to filter your projects.",
            "positive_screen": "dash_projects_01.png",
            "negative_text": "Click '+ New' button to create a brand new project instead of finding an existing one.",
            "negative_screen": "dash_empty_01.png",
        },
        {
            "query_screen": "dash_nav_open_01.png",
            "query_text": "The navigation menu is open. I want to go to my account settings.",
            "positive_text": "Click 'Settings' in the navigation menu on the left side of the screen.",
            "positive_screen": "dash_nav_open_01.png",
            "negative_text": "Click 'Home' in the navigation to go back to the main dashboard.",
            "negative_screen": "dash_empty_01.png",
        },
        # MODEL SELECTION
        {
            "query_screen": "model_closed_01.png",
            "query_text": "I'm ready to select a model for my video. The model dropdown shows 'Select Model' and is closed.",
            "positive_text": "Click the 'Select Model' dropdown button to open the list of available AI models.",
            "positive_screen": "model_closed_01.png",
            "negative_text": "Click directly in the workspace area below to start uploading without selecting a model first.",
            "negative_screen": "upload_empty_01.png",
        },
        {
            "query_screen": "model_open_01.png",
            "query_text": "The model dropdown is open and I can see all options. I want to animate a still image.",
            "positive_text": "Click on 'Animate' option in the dropdown list. It's designed to transform still images into animated videos.",
            "positive_screen": "model_open_01.png",
            "negative_text": "Click 'Text to Video' which creates videos from text descriptions, not from images.",
            "negative_screen": "model_open_01.png",
        },
        {
            "query_screen": "model_open_01.png",
            "query_text": "I want to create a video from a text prompt, not an image. The model dropdown is open.",
            "positive_text": "Click on 'Text to Video' option which generates videos directly from text descriptions.",
            "positive_screen": "model_open_01.png",
            "negative_text": "Click 'Image to Video' which requires uploading an image first.",
            "negative_screen": "model_open_01.png",
        },
        {
            "query_screen": "model_open_01.png",
            "query_text": "I want to swap a face in my video. The model dropdown is showing all options.",
            "positive_text": "Click on 'Face Swap' option to use the face replacement model.",
            "positive_screen": "model_open_01.png",
            "negative_text": "Click 'Style Transfer' which changes the artistic style, not faces.",
            "negative_screen": "model_open_01.png",
        },
        {
            "query_screen": "model_animate_01.png",
            "query_text": "I've selected the Animate model. Now I need to upload my image.",
            "positive_text": "The upload area is now visible. Click in the dropzone area or the 'Upload Image' button to select your file.",
            "positive_screen": "upload_empty_01.png",
            "negative_text": "Click the model dropdown again to change your model selection.",
            "negative_screen": "model_open_01.png",
        },
        # UPLOAD
        {
            "query_screen": "upload_empty_01.png",
            "query_text": "I see the upload area with a drag-drop zone. I want to upload my image file.",
            "positive_text": "Click the 'Upload Image' button in the center of the dropzone to open the file picker and select your image.",
            "positive_screen": "upload_empty_01.png",
            "negative_text": "Look at the 'Supported formats' text at the bottom for information but this doesn't upload anything.",
            "negative_screen": "upload_empty_01.png",
        },
        {
            "query_screen": "upload_progress_01.png",
            "query_text": "My image is uploading. I see a progress bar at 65%. I changed my mind and want to cancel.",
            "positive_text": "Click the 'Cancel' button below the progress bar to stop the upload.",
            "positive_screen": "upload_progress_01.png",
            "negative_text": "Wait for the upload to complete - there's nothing to click right now to speed it up.",
            "negative_screen": "upload_complete_01.png",
        },
        {
            "query_screen": "upload_complete_01.png",
            "query_text": "My image uploaded successfully. I can see the preview. I want to proceed to generation.",
            "positive_text": "Click the 'Continue' button below the image preview to move to the generation settings.",
            "positive_screen": "upload_complete_01.png",
            "negative_text": "Click 'Replace Image' to upload a different image instead.",
            "negative_screen": "upload_empty_01.png",
        },
        {
            "query_screen": "upload_complete_01.png",
            "query_text": "I uploaded the wrong image. I need to change it before generating.",
            "positive_text": "Click the 'Replace Image' button to select a different image file.",
            "positive_screen": "upload_complete_01.png",
            "negative_text": "Click 'Continue' which would proceed with the wrong image.",
            "negative_screen": "gen_settings_01.png",
        },
        # GENERATION SETTINGS
        {
            "query_screen": "gen_settings_01.png",
            "query_text": "I'm on the generation settings page. I want to make a longer video.",
            "positive_text": "Adjust the 'Duration' slider to increase the video length. Drag it to the right for longer videos.",
            "positive_screen": "gen_settings_01.png",
            "negative_text": "Click 'Generate Video' button immediately without changing duration.",
            "negative_screen": "gen_button_01.png",
        },
        {
            "query_screen": "gen_settings_01.png",
            "query_text": "I want to change the video quality to a higher resolution.",
            "positive_text": "Click the 'Resolution' dropdown and select a higher option like 1080p.",
            "positive_screen": "gen_settings_01.png",
            "negative_text": "Adjust the 'Motion' slider which controls animation intensity, not quality.",
            "negative_screen": "gen_settings_01.png",
        },
        {
            "query_screen": "gen_button_01.png",
            "query_text": "All settings are configured. I can see my image preview and settings summary. Time to generate.",
            "positive_text": "Click the green 'Generate Video' button to start the video creation process.",
            "positive_screen": "gen_button_01.png",
            "negative_text": "Click 'Save as Preset' to save settings for later, but this won't generate anything.",
            "negative_screen": "gen_settings_01.png",
        },
        # GENERATION PROGRESS
        {
            "query_screen": "gen_25_01.png",
            "query_text": "Video is generating. Progress shows 25%. I want to cancel and try different settings.",
            "positive_text": "Click the 'Cancel' button to stop the current generation. You can then adjust settings and try again.",
            "positive_screen": "gen_25_01.png",
            "negative_text": "Wait for generation to complete - it will finish automatically.",
            "negative_screen": "gen_complete_01.png",
        },
        {
            "query_screen": "gen_75_01.png",
            "query_text": "Generation is 75% done. I see a preview frame. Should I wait or can I do something?",
            "positive_text": "The generation is almost complete. Wait for it to finish - you can see the progress and preview.",
            "positive_screen": "gen_75_01.png",
            "negative_text": "Click 'Cancel' even though you're almost done, wasting the progress.",
            "negative_screen": "gen_settings_01.png",
        },
        {
            "query_screen": "gen_complete_01.png",
            "query_text": "Generation completed! I see a success message. I want to see my video.",
            "positive_text": "Click 'View Video' button to watch your generated video in the player.",
            "positive_screen": "gen_complete_01.png",
            "negative_text": "Click 'Create Another' to start over without viewing this video.",
            "negative_screen": "model_closed_01.png",
        },
        {
            "query_screen": "gen_complete_01.png",
            "query_text": "Video generated successfully. I want to save it to my computer immediately.",
            "positive_text": "Click 'Download' button to save the video file directly to your downloads folder.",
            "positive_screen": "gen_complete_01.png",
            "negative_text": "Click 'View Video' which only plays it but doesn't download.",
            "negative_screen": "result_player_01.png",
        },
        {
            "query_screen": "gen_error_01.png",
            "query_text": "Generation failed with an error. The message says server timed out.",
            "positive_text": "Click 'Try Again' button to retry the generation with the same settings.",
            "positive_screen": "gen_error_01.png",
            "negative_text": "Click 'Report Issue' which opens support but doesn't retry.",
            "negative_screen": "gen_error_01.png",
        },
        # RESULTS
        {
            "query_screen": "result_player_01.png",
            "query_text": "I'm viewing my generated video. I want to download it.",
            "positive_text": "Click the 'Download' button below the video player to save the video to your device.",
            "positive_screen": "result_player_01.png",
            "negative_text": "Click 'Regenerate' which creates a new video instead of downloading this one.",
            "negative_screen": "gen_25_01.png",
        },
        {
            "query_screen": "result_player_01.png",
            "query_text": "I want to share this video with someone. I'm looking at the player.",
            "positive_text": "Click the 'Share' button to get a shareable link or post to social media.",
            "positive_screen": "result_player_01.png",
            "negative_text": "Click 'Download' which saves locally but doesn't create a shareable link.",
            "negative_screen": "result_download_opts_01.png",
        },
        {
            "query_screen": "result_player_01.png",
            "query_text": "I don't like this result. I want to generate a different version.",
            "positive_text": "Click 'Regenerate' button to create a new version with the same or modified settings.",
            "positive_screen": "result_player_01.png",
            "negative_text": "Click 'Download' to save a video you don't like.",
            "negative_screen": "result_download_opts_01.png",
        },
        {
            "query_screen": "result_download_opts_01.png",
            "query_text": "Download options modal appeared. I want to download as MP4.",
            "positive_text": "MP4 is already selected (recommended). Click 'Download Now' to start the download.",
            "positive_screen": "result_download_opts_01.png",
            "negative_text": "Click 'Cancel' which closes the modal without downloading.",
            "negative_screen": "result_player_01.png",
        },
        {
            "query_screen": "result_download_opts_01.png",
            "query_text": "I want to download as GIF format instead of MP4.",
            "positive_text": "Click the 'GIF' radio button to select GIF format, then click 'Download Now'.",
            "positive_screen": "result_download_opts_01.png",
            "negative_text": "Click 'Download Now' immediately, which downloads as MP4 not GIF.",
            "negative_screen": "result_player_01.png",
        },
        {
            "query_screen": "result_share_01.png",
            "query_text": "The share modal is open. I want to copy the link to send manually.",
            "positive_text": "Click the 'Copy' button next to the link input field to copy the URL to your clipboard.",
            "positive_screen": "result_share_01.png",
            "negative_text": "Click 'Twitter' button which opens Twitter instead of just copying the link.",
            "negative_screen": "result_share_01.png",
        },
        {
            "query_screen": "result_share_01.png",
            "query_text": "I want to post my video on Twitter directly.",
            "positive_text": "Click the 'Twitter' button to open Twitter with your video link pre-filled.",
            "positive_screen": "result_share_01.png",
            "negative_text": "Click 'Copy' which only copies the link without opening Twitter.",
            "negative_screen": "result_share_01.png",
        },
    ]

    # Generate triplets by cycling through templates with variations
    triplet_id = 0
    while len(triplets) < count:
        template = templates[triplet_id % len(templates)]

        # Create variations by slightly modifying the text
        variation = triplet_id // len(templates)

        query_prefix = [
            "",
            "Looking at the screen, ",
            "I can see that ",
            "Currently, ",
            "On this page, ",
        ][variation % 5]

        triplet = {
            "query": {
                "text": query_prefix + template["query_text"],
                "image": f"screenshots/{template['query_screen']}"
            },
            "positive": {
                "text": template["positive_text"],
                "image": f"screenshots/{template['positive_screen']}"
            },
            "negative": {
                "text": template["negative_text"],
                "image": f"screenshots/{template['negative_screen']}"
            }
        }

        triplets.append(triplet)
        triplet_id += 1

    return triplets


def generate_reranker_pairs(count: int) -> list:
    """
    Generate reranker pairs for training.

    Each pair has:
    - query: Screenshot + description of what user wants
    - document: A possible action
    - label: 1 (correct) or 0 (wrong)

    CRITICAL: Must be balanced - 50% label=1, 50% label=0
    """
    pairs = []

    # Templates for positive (label=1) and negative (label=0) pairs
    templates = [
        # AUTH
        {
            "screen": "auth_login_empty_01.png",
            "query_text": "Login page with empty fields. Need to start entering credentials.",
            "positive_docs": [
                "Click on the email input field to begin entering your login credentials.",
                "Select the email field and type your email address.",
            ],
            "negative_docs": [
                "Click 'Sign Up' to create a new account.",
                "Click 'Forgot Password' to reset credentials.",
            ],
        },
        {
            "screen": "auth_login_filled_01.png",
            "query_text": "Login form filled with email and password. Ready to submit.",
            "positive_docs": [
                "Click the 'Log In' button to submit credentials and sign in.",
                "Press the login button to authenticate.",
            ],
            "negative_docs": [
                "Clear the form and start over.",
                "Click 'Sign Up' to create an account.",
            ],
        },
        # DASHBOARD
        {
            "screen": "dash_empty_01.png",
            "query_text": "Empty dashboard. Want to create first video project.",
            "positive_docs": [
                "Click 'Create Your First Video' button in the center.",
                "Select the new project button to start creating.",
            ],
            "negative_docs": [
                "Click profile icon to view settings.",
                "Click navigation menu to explore options.",
            ],
        },
        {
            "screen": "dash_projects_01.png",
            "query_text": "Dashboard with existing projects. Want to create new video.",
            "positive_docs": [
                "Click the '+ New' button in the top right.",
                "Select the new project option to begin.",
            ],
            "negative_docs": [
                "Click on an existing project to edit it.",
                "Use the search bar to find old projects.",
            ],
        },
        # MODEL
        {
            "screen": "model_closed_01.png",
            "query_text": "Model selector dropdown is closed. Need to choose a model.",
            "positive_docs": [
                "Click the 'Select Model' dropdown to see available options.",
                "Open the model dropdown menu to choose an AI model.",
            ],
            "negative_docs": [
                "Start uploading without selecting a model.",
                "Go back to dashboard.",
            ],
        },
        {
            "screen": "model_open_01.png",
            "query_text": "Model dropdown open. Want to animate an image.",
            "positive_docs": [
                "Click 'Animate' option to select image animation model.",
                "Select the Animate model for image-to-video conversion.",
            ],
            "negative_docs": [
                "Click 'Text to Video' for text-based generation.",
                "Click 'Face Swap' for face replacement.",
            ],
        },
        # UPLOAD
        {
            "screen": "upload_empty_01.png",
            "query_text": "Upload area visible. Need to upload an image.",
            "positive_docs": [
                "Click 'Upload Image' button to open file picker.",
                "Drag and drop an image onto the dropzone area.",
            ],
            "negative_docs": [
                "Read the supported formats text.",
                "Go back to model selection.",
            ],
        },
        {
            "screen": "upload_complete_01.png",
            "query_text": "Image uploaded successfully. Ready to proceed.",
            "positive_docs": [
                "Click 'Continue' to move to generation settings.",
                "Proceed to the next step with the uploaded image.",
            ],
            "negative_docs": [
                "Click 'Replace Image' to change the file.",
                "Click the X button to remove the image.",
            ],
        },
        # GENERATION
        {
            "screen": "gen_button_01.png",
            "query_text": "Settings configured. Ready to generate video.",
            "positive_docs": [
                "Click the green 'Generate Video' button.",
                "Start video generation with current settings.",
            ],
            "negative_docs": [
                "Save settings as a preset first.",
                "Go back to change the uploaded image.",
            ],
        },
        {
            "screen": "gen_complete_01.png",
            "query_text": "Video generation completed. Want to download result.",
            "positive_docs": [
                "Click 'Download' button to save the video.",
                "Download the generated video to your device.",
            ],
            "negative_docs": [
                "Click 'Create Another' to start new project.",
                "Click 'View Video' to only watch it.",
            ],
        },
        # RESULTS
        {
            "screen": "result_player_01.png",
            "query_text": "Viewing generated video. Want to share it.",
            "positive_docs": [
                "Click 'Share' button to get shareable link.",
                "Open share options to post on social media.",
            ],
            "negative_docs": [
                "Click 'Download' to save locally.",
                "Click 'Regenerate' to make new version.",
            ],
        },
        {
            "screen": "result_share_01.png",
            "query_text": "Share modal open. Want to copy link.",
            "positive_docs": [
                "Click 'Copy' button to copy link to clipboard.",
                "Copy the shareable URL for manual sharing.",
            ],
            "negative_docs": [
                "Click 'Twitter' to post there instead.",
                "Click 'Close' without copying.",
            ],
        },
    ]

    # Generate pairs - alternating positive and negative to ensure balance
    pair_id = 0
    positive_count = 0
    negative_count = 0
    target_each = count // 2

    while positive_count + negative_count < count:
        template = templates[pair_id % len(templates)]

        # Alternate between positive and negative
        if positive_count < target_each and (negative_count >= target_each or pair_id % 2 == 0):
            # Generate positive pair
            doc_idx = positive_count % len(template["positive_docs"])
            pair = {
                "query": {
                    "text": template["query_text"],
                    "image": f"screenshots/{template['screen']}"
                },
                "document": template["positive_docs"][doc_idx],
                "label": 1
            }
            positive_count += 1
        else:
            # Generate negative pair
            doc_idx = negative_count % len(template["negative_docs"])
            pair = {
                "query": {
                    "text": template["query_text"],
                    "image": f"screenshots/{template['screen']}"
                },
                "document": template["negative_docs"][doc_idx],
                "label": 0
            }
            negative_count += 1

        pairs.append(pair)
        pair_id += 1

    print(f"Generated {positive_count} positive and {negative_count} negative pairs")
    return pairs


def generate_sft_conversations(count: int) -> list:
    """
    Generate SFT conversations for training.

    Each conversation shows the model how to respond with actions.
    """
    conversations = []

    templates = [
        {
            "screen": "auth_login_empty_01.png",
            "user_text": "I need to log into Higgsfield with my email user@example.com",
            "assistant_text": """I can see the Higgsfield login page with empty email and password fields.

To log in:
1. Click the email input field
2. Enter your email address

ACTION: CLICK(512, 300)
ACTION: TYPE(user@example.com)

After typing your email, you'll need to enter your password in the next field."""
        },
        {
            "screen": "auth_login_filled_01.png",
            "user_text": "My credentials are entered. Log me in now.",
            "assistant_text": """I can see your email and password are filled in.

Clicking the Login button to sign you in.

ACTION: CLICK(512, 420)

You should be redirected to the dashboard after authentication."""
        },
        {
            "screen": "dash_empty_01.png",
            "user_text": "I want to create my first video",
            "assistant_text": """Welcome to Higgsfield! I can see this is your first time here - the dashboard is empty.

Let me help you create your first video.

ACTION: CLICK(512, 350)

This clicks the 'Create Your First Video' button which will take you to the project creation flow."""
        },
        {
            "screen": "dash_projects_01.png",
            "user_text": "Start a new project for me",
            "assistant_text": """I can see your dashboard with existing projects. I'll start a new one for you.

ACTION: CLICK(900, 150)

Clicking the '+ New' button in the top right. This will open the model selection screen."""
        },
        {
            "screen": "model_closed_01.png",
            "user_text": "I want to animate an image",
            "assistant_text": """To animate an image, we need to select the Animate model first.

I'll open the model selector dropdown.

ACTION: CLICK(400, 150)

This opens the model menu where we can choose 'Animate' for image animation."""
        },
        {
            "screen": "model_open_01.png",
            "user_text": "Select Animate for image to video",
            "assistant_text": """The model dropdown is open. I can see all the available options.

Selecting the Animate model for you.

ACTION: CLICK(400, 200)

'Animate' transforms still images into animated videos. After this, you'll be able to upload your image."""
        },
        {
            "screen": "upload_empty_01.png",
            "user_text": "Upload my image",
            "assistant_text": """I can see the upload area. You can either drag and drop or click to upload.

I'll click the upload button to open the file picker.

ACTION: CLICK(512, 420)

Select your image file from the dialog that opens. Supported formats are PNG, JPG, and WebP up to 10MB."""
        },
        {
            "screen": "upload_complete_01.png",
            "user_text": "Image is uploaded, continue to next step",
            "assistant_text": """Your image is uploaded and I can see the preview. It looks good.

Moving to the generation settings.

ACTION: CLICK(512, 550)

This takes us to the settings panel where you can configure duration, quality, and other options."""
        },
        {
            "screen": "gen_settings_01.png",
            "user_text": "Make the video 10 seconds long",
            "assistant_text": """I'll adjust the duration slider to 10 seconds.

ACTION: CLICK(750, 200)

Dragging the duration slider to set 10 second video length. The default was 5 seconds."""
        },
        {
            "screen": "gen_button_01.png",
            "user_text": "Generate the video now",
            "assistant_text": """Everything is configured. I can see your image preview and the settings summary.

Starting generation now.

ACTION: CLICK(700, 500)

Clicking the green 'Generate Video' button. This typically takes 30-60 seconds depending on settings."""
        },
        {
            "screen": "gen_25_01.png",
            "user_text": "Cancel the generation",
            "assistant_text": """I'll cancel the current generation for you.

ACTION: CLICK(512, 500)

This stops the generation process. You can adjust settings and try again."""
        },
        {
            "screen": "gen_complete_01.png",
            "user_text": "Great! Download the video",
            "assistant_text": """Video generated successfully! I'll download it for you.

ACTION: CLICK(574, 350)

The video will be saved to your Downloads folder as an MP4 file."""
        },
        {
            "screen": "result_player_01.png",
            "user_text": "Share this video",
            "assistant_text": """I'll open the share options for you.

ACTION: CLICK(700, 520)

This opens the share modal where you can copy a link or post to social media."""
        },
        {
            "screen": "result_share_01.png",
            "user_text": "Copy the share link",
            "assistant_text": """Copying the shareable link to your clipboard.

ACTION: CLICK(700, 280)

The link is now copied. You can paste it anywhere to share your video."""
        },
        {
            "screen": "result_player_01.png",
            "user_text": "I don't like this, regenerate it",
            "assistant_text": """I'll start a new generation for you.

ACTION: CLICK(800, 520)

This opens the regeneration options where you can adjust settings and create a new version."""
        },
        {
            "screen": "gen_error_01.png",
            "user_text": "Generation failed, try again",
            "assistant_text": """I see the generation failed due to a timeout. Let me retry for you.

ACTION: CLICK(512, 400)

Clicking 'Try Again' to restart the generation with the same settings."""
        },
        {
            "screen": "dash_nav_open_01.png",
            "user_text": "Go to settings",
            "assistant_text": """I'll navigate to your account settings.

ACTION: CLICK(100, 300)

Clicking 'Settings' in the navigation menu. This opens your account preferences."""
        },
    ]

    # Generate conversations with variations
    conv_id = 0
    while len(conversations) < count:
        template = templates[conv_id % len(templates)]

        # Add variation to user message
        variation = conv_id // len(templates)
        prefixes = ["", "Please ", "Can you ", "I want to ", "Help me "]
        prefix = prefixes[variation % len(prefixes)]

        conversation = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"screenshots/{template['screen']}"},
                        {"type": "text", "text": prefix + template["user_text"]}
                    ]
                },
                {
                    "role": "assistant",
                    "content": template["assistant_text"]
                }
            ]
        }

        conversations.append(conversation)
        conv_id += 1

    return conversations


def generate_dpo_pairs(count: int) -> list:
    """
    Generate DPO preference pairs.

    Each pair has:
    - prompt: User request with screenshot
    - chosen: Good, direct response with ACTION
    - rejected: Bad, vague response without ACTION
    """
    pairs = []

    templates = [
        {
            "screen": "gen_button_01.png",
            "prompt_text": "Generate the video",
            "chosen": """Starting video generation now.

ACTION: CLICK(700, 500)

The green button initiates generation. This will take about 30 seconds.""",
            "rejected": """I'd be happy to help you generate a video! Before we proceed, would you like to review your settings? There are options for duration, quality, and motion intensity that you might want to adjust. Let me know what you'd prefer and I can guide you through the options available."""
        },
        {
            "screen": "auth_login_empty_01.png",
            "prompt_text": "Log me in",
            "chosen": """Clicking the email field to start login.

ACTION: CLICK(512, 300)

Enter your email address when the field is focused.""",
            "rejected": """Logging in is an important step! First, you'll need to have an account. Do you have an account already? If not, you might want to sign up first. The login process involves entering your email and password. Would you like me to explain the different authentication options available?"""
        },
        {
            "screen": "result_player_01.png",
            "prompt_text": "Download this",
            "chosen": """Downloading your video now.

ACTION: CLICK(600, 520)

The video will save to your Downloads folder as MP4.""",
            "rejected": """Great question! There are several ways to save your video. You could download it in different formats like MP4, WebM, or even GIF. Each format has pros and cons. MP4 is most compatible, WebM is smaller, and GIF works for short clips but has no audio. What format would you prefer?"""
        },
        {
            "screen": "model_open_01.png",
            "prompt_text": "Select Animate",
            "chosen": """Selecting Animate model.

ACTION: CLICK(400, 200)

This model converts still images to animated videos.""",
            "rejected": """The Animate model is a great choice! It's designed specifically for transforming static images into dynamic animations. Before selecting, you might want to know that it works best with high-quality source images. There are also other options like Text-to-Video if you don't have an image. Should I explain the differences?"""
        },
        {
            "screen": "upload_empty_01.png",
            "prompt_text": "Upload image",
            "chosen": """Opening file picker.

ACTION: CLICK(512, 420)

Select your image (PNG, JPG, or WebP up to 10MB).""",
            "rejected": """Uploading an image is the next step! You have a few options here. You can click the upload button, or you can drag and drop your file. Make sure your image is in a supported format - we accept PNG, JPG, and WebP files. The file should be under 10MB. Do you have an image ready, or would you like tips on choosing a good source image?"""
        },
        {
            "screen": "result_share_01.png",
            "prompt_text": "Copy link",
            "chosen": """Copying link to clipboard.

ACTION: CLICK(700, 280)

Link copied. Paste anywhere to share.""",
            "rejected": """Sure! There are multiple ways to share your video. You can copy the direct link, which is great for messaging apps. Or you could share directly to social platforms like Twitter, Facebook, or LinkedIn. The direct link is the most versatile option. Which sharing method would work best for your needs?"""
        },
    ]

    # Generate pairs
    pair_id = 0
    while len(pairs) < count:
        template = templates[pair_id % len(templates)]

        pair = {
            "prompt": [
                {"type": "image", "image": f"screenshots/{template['screen']}"},
                {"type": "text", "text": template["prompt_text"]}
            ],
            "chosen": template["chosen"],
            "rejected": template["rejected"]
        }

        pairs.append(pair)
        pair_id += 1

    return pairs


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    """Generate complete dataset for 3B model training."""

    print("=" * 70)
    print("DATASET GENERATOR FOR 3B MODEL")
    print("=" * 70)

    # Create directory structure
    os.makedirs("higgsfield_dataset/screenshots", exist_ok=True)
    os.makedirs("higgsfield_dataset/data", exist_ok=True)
    os.makedirs("higgsfield_dataset/validation", exist_ok=True)

    # Generate datasets
    print("\n[1/4] Generating embedding triplets...")
    embedding_data = generate_embedding_triplets(MINIMUM_EMBEDDING_TRIPLETS)
    print(f"      Generated {len(embedding_data)} triplets")

    print("\n[2/4] Generating reranker pairs...")
    reranker_data = generate_reranker_pairs(MINIMUM_RERANKER_PAIRS)
    print(f"      Generated {len(reranker_data)} pairs")

    print("\n[3/4] Generating SFT conversations...")
    sft_data = generate_sft_conversations(MINIMUM_SFT_CONVERSATIONS)
    print(f"      Generated {len(sft_data)} conversations")

    print("\n[4/4] Generating DPO pairs...")
    dpo_data = generate_dpo_pairs(MINIMUM_DPO_PAIRS)
    print(f"      Generated {len(dpo_data)} pairs")

    # Split into train/validation
    def split_data(data, val_ratio=0.1):
        split_idx = int(len(data) * (1 - val_ratio))
        return data[:split_idx], data[split_idx:]

    embedding_train, embedding_val = split_data(embedding_data)
    reranker_train, reranker_val = split_data(reranker_data)
    sft_train, sft_val = split_data(sft_data)
    dpo_train, dpo_val = split_data(dpo_data)

    # Write files
    def write_jsonl(filepath, data):
        with open(filepath, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

    print("\nWriting training files...")
    write_jsonl("higgsfield_dataset/data/embedding_train.jsonl", embedding_train)
    write_jsonl("higgsfield_dataset/data/reranker_train.jsonl", reranker_train)
    write_jsonl("higgsfield_dataset/data/sft_train.jsonl", sft_train)
    write_jsonl("higgsfield_dataset/data/dpo_train.jsonl", dpo_train)

    print("Writing validation files...")
    write_jsonl("higgsfield_dataset/validation/embedding_val.jsonl", embedding_val)
    write_jsonl("higgsfield_dataset/validation/reranker_val.jsonl", reranker_val)
    write_jsonl("higgsfield_dataset/validation/sft_val.jsonl", sft_val)
    write_jsonl("higgsfield_dataset/validation/dpo_val.jsonl", dpo_val)

    # Create placeholder screenshots info
    screenshots_readme = """# Screenshots Required

This folder needs actual screenshots from Higgsfield AI.

## Required Screenshots:

### Auth (in auth/ subfolder):
- auth_login_empty_01.png
- auth_login_filled_01.png
- auth_signup_empty_01.png
- auth_login_error_01.png

### Dashboard (in dashboard/ subfolder):
- dash_empty_01.png
- dash_projects_01.png
- dash_nav_open_01.png

### Model Selection (in model/ subfolder):
- model_closed_01.png
- model_open_01.png
- model_animate_01.png

### Upload (in upload/ subfolder):
- upload_empty_01.png
- upload_hover_01.png
- upload_progress_01.png
- upload_complete_01.png

### Generation (in generation/ subfolder):
- gen_settings_01.png
- gen_button_01.png
- gen_25_01.png
- gen_75_01.png
- gen_complete_01.png
- gen_error_01.png

### Results (in results/ subfolder):
- result_player_01.png
- result_download_opts_01.png
- result_share_01.png

## Screenshot Requirements:
- Resolution: 1024x768 pixels
- Format: PNG
- Color: RGB
"""

    with open("higgsfield_dataset/screenshots/README.md", 'w') as f:
        f.write(screenshots_readme)

    # Summary
    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"""
Files created in higgsfield_dataset/:

  data/embedding_train.jsonl    : {len(embedding_train)} triplets
  data/reranker_train.jsonl     : {len(reranker_train)} pairs
  data/sft_train.jsonl          : {len(sft_train)} conversations
  data/dpo_train.jsonl          : {len(dpo_train)} pairs

  validation/embedding_val.jsonl: {len(embedding_val)} triplets
  validation/reranker_val.jsonl : {len(reranker_val)} pairs
  validation/sft_val.jsonl      : {len(sft_val)} conversations
  validation/dpo_val.jsonl      : {len(dpo_val)} pairs

NEXT STEPS:
1. Take the required screenshots from Higgsfield AI
2. Place screenshots in higgsfield_dataset/screenshots/
3. Verify paths match the JSONL files
4. Run validation: python validate_dataset.py

NOTE: The generated data uses placeholder screenshot paths.
      You MUST replace these with real screenshots!
""")


if __name__ == "__main__":
    main()
