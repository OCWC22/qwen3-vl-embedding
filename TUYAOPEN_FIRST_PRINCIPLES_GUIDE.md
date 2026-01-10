# TuyaOpen First Principles Guide: Building AI IoT from Scratch

## For a Software Engineer with No Hardware Background

---

# PART 1: PHYSICS FUNDAMENTALS - Why Any of This Works

## 1.1 Electricity: The Foundation of Everything

**What is electricity?**
Electrons flowing through conductive materials (metals like copper). Think of it like water flowing through pipes:
- **Voltage (V)** = Water pressure (how hard electrons are pushed)
- **Current (A)** = Flow rate (how many electrons per second)
- **Resistance (Ω)** = Pipe diameter (how much the material resists flow)

**Ohm's Law**: `V = I × R`

Your T5 chip operates at **3.3V logic** - meaning:
- A "1" (HIGH) = 3.3 volts
- A "0" (LOW) = 0 volts (ground)

This is how ALL digital computing works - billions of tiny switches representing 1s and 0s.

## 1.2 Semiconductors: The Magic Material

**Why silicon?**
Silicon is a "semiconductor" - it can be either conductive OR insulating depending on how you treat it:

```
Pure Silicon → Insulator (no free electrons)
     ↓
Add Phosphorus (N-type) → Extra electrons → Conducts
Add Boron (P-type) → "Holes" for electrons → Conducts differently
     ↓
Combine N + P → PN Junction → DIODE (one-way valve)
     ↓
Add control gate → TRANSISTOR (electronic switch)
```

**Transistors are switches controlled by electricity:**
- Apply voltage to gate → Switch ON → Current flows
- Remove voltage → Switch OFF → Current stops

The T5 chip has **~100 million transistors** switching on/off billions of times per second.

## 1.3 How a CPU Actually Works

```
                    ┌─────────────────────────────────────────┐
                    │              T5 CHIP DIE                │
                    │                                         │
  ┌─────────┐       │   ┌───────────┐    ┌──────────────┐    │
  │  Flash  │◄─────►│   │  ARM CPU  │◄──►│    SRAM      │    │
  │ (Code)  │       │   │ Cortex-M  │    │  (Fast RAM)  │    │
  └─────────┘       │   └───────────┘    └──────────────┘    │
                    │         ▲                              │
                    │         │ Bus                          │
                    │         ▼                              │
                    │   ┌───────────────────────────────┐    │
                    │   │        PERIPHERALS            │    │
                    │   │  ┌─────┐ ┌─────┐ ┌────────┐   │    │
                    │   │  │GPIO │ │UART │ │ Camera │   │    │
                    │   │  │Pins │ │/SPI │ │ Interface│  │    │
                    │   │  └─────┘ └─────┘ └────────┘   │    │
                    │   └───────────────────────────────┘    │
                    └─────────────────────────────────────────┘
```

**Clock Speed**: The chip has a crystal oscillator that "ticks" at a specific frequency (e.g., 240 MHz = 240 million ticks/second). Each tick, instructions execute.

**Instruction Cycle**:
1. FETCH: Read instruction from memory
2. DECODE: Figure out what instruction means
3. EXECUTE: Do the operation
4. WRITEBACK: Store result

This happens 240 million times per second on the T5.

## 1.4 How Cameras Work (Image Sensors)

```
LIGHT → LENS → IMAGE SENSOR → DIGITAL DATA
         │
         ▼
    ┌────────────────────────────────┐
    │   CMOS IMAGE SENSOR            │
    │   (Grid of photodiodes)        │
    │                                │
    │   [█][█][█][█][█][█][█][█]    │ ← Each square = 1 pixel
    │   [█][█][█][█][█][█][█][█]    │   (photodiode + amplifier)
    │   [█][█][█][█][█][█][█][█]    │
    │   [█][█][█][█][█][█][█][█]    │   480x480 = 230,400 pixels
    │                                │
    └────────────────────────────────┘
              │
              ▼
    Photons hit silicon → Generate electrons
    More light → More electrons → Higher voltage
    ADC converts voltage → 8-bit number (0-255)
```

**Color Detection (Bayer Pattern)**:
```
    R G R G R G    ← Red and Green filters
    G B G B G B    ← Green and Blue filters
    R G R G R G
    G B G B G B
```
Each pixel only sees ONE color. Software interpolates the missing colors.

**Output Format**: YUV422 (Luminance + Chrominance)
- Y = Brightness (most data, eyes are sensitive to this)
- U/V = Color difference (less data, eyes less sensitive)

---

# PART 2: T5AI CHIP DEEP DIVE

## 2.1 T5AI Hardware Specifications

Based on the TuyaOpen repository, the T5AI is an ARM-based SoC (System on Chip):

```
┌─────────────────────────────────────────────────────────────────┐
│                        T5AI SoC                                 │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   │
│  │ ARM Cortex-M │   │    SRAM      │   │     PSRAM        │   │
│  │   CPU Core   │   │   (Fast)     │   │  (Large/Slow)    │   │
│  │   240 MHz    │   │   ~512KB     │   │    4-8 MB        │   │
│  └──────────────┘   └──────────────┘   └──────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    PERIPHERALS                          │   │
│  │                                                         │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────────┐  │   │
│  │  │  WiFi   │  │Bluetooth│  │  DMA2D  │  │  Camera  │  │   │
│  │  │ 2.4GHz  │  │  BLE 5  │  │  Accel  │  │Interface │  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └──────────┘  │   │
│  │                                                         │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────────┐  │   │
│  │  │   SPI   │  │  I2C    │  │  UART   │  │  LCD     │  │   │
│  │  │ Display │  │ Sensors │  │  Debug  │  │Controller│  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └──────────┘  │   │
│  │                                                         │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────────────────┐│   │
│  │  │  ADC    │  │   PWM   │  │      GPIO (40+ pins)    ││   │
│  │  │ (Audio) │  │  (LEDs) │  │                         ││   │
│  │  └─────────┘  └─────────┘  └─────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     FLASH MEMORY                        │   │
│  │              4-16 MB (Stores your code)                 │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 2.2 Key Hardware Features for Your Workout AI

| Feature | Capability | Why You Need It |
|---------|-----------|-----------------|
| **DMA2D Accelerator** | Hardware YUV→RGB conversion | Real-time video processing without CPU |
| **Camera Interface** | 480x480 @ 20 FPS JPEG/YUV | Capture workout video frames |
| **PSRAM** | 4-8 MB external RAM | Store video buffers (1.35 MB per double-buffer) |
| **WiFi** | 802.11 b/g/n 2.4 GHz | Stream to Gemini Live cloud |
| **LCD Controller** | SPI/Parallel displays | Show feedback to user |
| **Audio ADC/DAC** | 16kHz 16-bit mono | Voice feedback from AI |
| **GPIO** | 40+ configurable pins | Connect sensors, buttons, LEDs |

## 2.3 Available T5AI Board Variants

From the repository's `boards/T5AI/` directory:

```
boards/T5AI/
├── ATK_T5AI_MINI_BOARD     ← Compact form factor
├── T5AI_MINI               ← Minimal configuration
├── T5AI_MOJI_1_28          ← 1.28" display variant
├── T5AI_OTTO               ← Robot variant
├── TUYA_T5AI_BOARD         ← Reference design (RECOMMENDED)
├── TUYA_T5AI_CORE          ← Core module only
├── TUYA_T5AI_DESKTOP_ROBOT ← Desktop robot platform
├── TUYA_T5AI_EVB           ← Evaluation board (RECOMMENDED)
├── TUYA_T5AI_PIXEL         ← Pixel display variant
├── TUYA_T5AI_POCKET        ← Portable variant
└── WAVESHARE_T5AI_TOUCH_AMOLED_1_75  ← Touchscreen AMOLED
```

**For your workout AI, I recommend**: `TUYA_T5AI_EVB` (full evaluation board with camera + display)

---

# PART 3: TUYAOPEN SOFTWARE ARCHITECTURE

## 3.1 Repository Structure Overview

```
TuyaOpen/
├── apps/                    # Complete applications
│   └── tuya.ai/
│       ├── ai_components/   # Reusable AI modules
│       │   └── ai_audio/    # Audio processing (ASR, TTS)
│       ├── your_chat_bot/   # Voice AI chatbot
│       ├── duo_eye_mood/    # Vision emotion detection
│       └── your_otto_robot/ # Robotics example
│
├── boards/                  # Hardware configurations
│   └── T5AI/               # T5AI chip boards
│
├── examples/               # Learning examples
│   ├── graphics/          # Display/UI examples
│   │   └── lvgl2Camera/   # Camera→Display pipeline
│   ├── multimedia/        # Audio examples
│   └── wifi/              # Network examples
│
├── src/                   # Core SDK source
│   ├── tuya_ai_service/   # AI processing layer
│   │   ├── svc_ai_agent/  # AI agent framework
│   │   ├── svc_ai_codec/  # Audio/video codecs
│   │   └── wukong/        # MCP server (AI model integration)
│   ├── tuya_cloud_service/# Cloud connectivity
│   ├── tuya_p2p/          # P2P video streaming
│   ├── tal_*/             # Hardware abstraction layers
│   └── lib*/              # Libraries (LVGL, HTTP, MQTT, etc.)
│
└── platform/              # Chip-specific implementations
```

## 3.2 Software Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     YOUR APPLICATION                                │
│   (apps/tuya.ai/your_chat_bot/ or YOUR CUSTOM APP)                │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                    AI SERVICE LAYER                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐              │
│  │ ai_audio    │  │ svc_ai_agent │  │ wukong MCP    │              │
│  │ (ASR/TTS)   │  │ (Processing) │  │ (AI Models)   │              │
│  └─────────────┘  └──────────────┘  └───────────────┘              │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                   CLOUD SERVICE LAYER                               │
│  ┌─────────┐  ┌─────────┐  ┌────────┐  ┌───────────────────────┐  │
│  │  MQTT   │  │  HTTP   │  │  P2P   │  │  Tuya Cloud / Gemini  │  │
│  │ Realtime│  │  REST   │  │ Stream │  │  (External AI APIs)   │  │
│  └─────────┘  └─────────┘  └────────┘  └───────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│               TUYA ABSTRACTION LAYER (TAL)                          │
│  ┌──────────┐ ┌───────────┐ ┌────────────┐ ┌─────────────────────┐ │
│  │ tal_wifi │ │tal_driver │ │tal_network │ │ tal_security        │ │
│  └──────────┘ └───────────┘ └────────────┘ └─────────────────────┘ │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                  HARDWARE (T5AI CHIP)                               │
│  WiFi Radio │ Camera │ Display │ Audio │ GPIO │ Flash │ RAM        │
└─────────────────────────────────────────────────────────────────────┘
```

## 3.3 The Video Pipeline (What You Need for Workout AI)

From `examples/graphics/lvgl2Camera/`:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    VIDEO CAPTURE PIPELINE                           │
└─────────────────────────────────────────────────────────────────────┘

   CAMERA SENSOR                     DMA2D ACCELERATOR
        │                                  │
        ▼                                  ▼
┌──────────────────┐              ┌───────────────────┐
│  YUV422 Frame    │─────────────►│  YUV422 → RGB565  │
│  480x480 pixels  │   DMA        │  Hardware Convert │
│  ~460 KB         │   Transfer   │  (No CPU usage)   │
└──────────────────┘              └─────────┬─────────┘
                                            │
                                            ▼
                                  ┌───────────────────┐
                                  │  RGB565 Buffer    │
                                  │  sg_display_      │
                                  │  frame_buff[0/1]  │
                                  │  DOUBLE BUFFERED  │
                                  └─────────┬─────────┘
                                            │
        ┌───────────────────────────────────┴──────────────┐
        ▼                                                   ▼
┌───────────────────┐                            ┌──────────────────┐
│   LCD DISPLAY     │                            │   JPEG ENCODE    │
│   Local Preview   │                            │   For Cloud      │
│   20 FPS          │                            │   Streaming      │
└───────────────────┘                            └──────────────────┘
```

**Key Code from `lvgl2Camera`**:
```c
// Camera initialization (from app_camera.c)
cfg.width = 480;
cfg.height = 480;
cfg.fps = 20;
cfg.out_fmt = TDL_CAMERA_FMT_JPEG_YUV422_BOTH;  // Dual output!
cfg.jpg_quality_size_max = 25 * 1024;  // 25KB JPEG max

// Buffer allocation - CRITICAL for performance
sg_camera_frame_buff = (uint8_t *)tkl_system_psram_malloc(480*480*2);
sg_display_frame_buff[0] = (uint8_t *)tkl_system_psram_malloc(480*480*2);
sg_display_frame_buff[1] = (uint8_t *)tkl_system_psram_malloc(480*480*2);
// Total: ~1.35 MB in PSRAM
```

## 3.4 The Audio Pipeline

From `ai_components/ai_audio/`:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AUDIO PROCESSING PIPELINE                        │
└─────────────────────────────────────────────────────────────────────┘

     MICROPHONE                          SPEAKER
         │                                  ▲
         ▼                                  │
┌─────────────────┐                ┌────────────────┐
│  ADC (16kHz)    │                │    DAC         │
│  16-bit Mono    │                │   Audio Out    │
└────────┬────────┘                └────────▲───────┘
         │                                  │
         ▼                                  │
┌─────────────────┐                ┌────────────────┐
│  ai_audio_input │                │ ai_audio_player│
│  VAD Detection  │                │  TTS Playback  │
│  Buffer & Send  │                │  MP3 Decode    │
└────────┬────────┘                └────────▲───────┘
         │                                  │
         ▼                                  │
┌─────────────────┐                ┌────────────────┐
│ ai_audio_agent  │                │ AI Response    │
│ Upload to Cloud │───────────────►│ Stream Back    │
└────────┬────────┘                └────────▲───────┘
         │                                  │
         └──────────► TUYA CLOUD ───────────┘
                         │
                         ▼
                   ┌───────────┐
                   │ AI MODEL  │
                   │ (Gemini,  │
                   │  ChatGPT, │
                   │  Claude)  │
                   └───────────┘
```

## 3.5 The MCP (Model Context Protocol) - AI Integration

From `src/tuya_ai_service/wukong/wukong_ai_mcp_server.c`:

```c
// MCP enables AI models to call device functions
// This is how Gemini can control your device!

// Tool Registration Example:
WUKONG_MCP_TOOL_ADD(
    "device.camera.take_photo",           // Tool name AI can call
    "Capture a photo from device camera", // Description for AI
    __camera_take_photo_cb                // Your callback function
);

// When AI says "take a photo", this runs:
static int __camera_take_photo_cb(WUKONG_MCP_PROPERTY_T *args,
                                   WUKONG_MCP_RET_VAL_T *ret) {
    JPEG_FRAME_T *jpeg = app_camera_jpeg_capture(3000);  // 3s timeout
    if (jpeg) {
        ret->type = WUKONG_MCP_RET_TP_IMAGE_JPEG;
        ret->data.image.data = jpeg->data;
        ret->data.image.len = jpeg->len;
    }
    return 0;
}
```

---

# PART 4: END-TO-END DATA FLOW FOR YOUR WORKOUT AI

## 4.1 Complete System Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    YOUR WORKOUT AI DEVICE                           │
│                                                                     │
│   ┌─────────┐                              ┌──────────────────┐    │
│   │ CAMERA  │────► Video Frames ──────────►│  JPEG ENCODER    │    │
│   │ 480x480 │      (20 FPS)                │  (Hardware)      │    │
│   └─────────┘                              └────────┬─────────┘    │
│                                                     │              │
│   ┌─────────┐                                       │              │
│   │   MIC   │────► Audio Stream ──┐                 │              │
│   │ 16kHz   │      (VAD Active)   │                 │              │
│   └─────────┘                     │                 │              │
│                                   ▼                 ▼              │
│                          ┌─────────────────────────────┐           │
│                          │     MULTIMODAL BUFFER       │           │
│                          │   Video + Audio Aligned     │           │
│                          └──────────────┬──────────────┘           │
│                                         │                          │
│   ┌─────────┐                           │                          │
│   │ DISPLAY │◄───────────────┐          │                          │
│   │ 480x480 │                │          │                          │
│   │ AMOLED  │                │          │                          │
│   └─────────┘                │          │                          │
│                              │          │                          │
│   ┌─────────┐                │          ▼                          │
│   │ SPEAKER │◄───────────────┤  ┌─────────────────┐                │
│   │  Audio  │                │  │  WiFi TX/RX     │                │
│   └─────────┘                │  │  WebSocket or   │                │
│                              │  │  HTTP Stream    │                │
│                              │  └────────┬────────┘                │
└──────────────────────────────┼───────────┼─────────────────────────┘
                               │           │
                               │           ▼
                     ┌─────────┴───────────────────────────┐
                     │         INTERNET / CLOUD            │
                     │                                     │
                     │  ┌─────────────────────────────┐   │
                     │  │      GEMINI LIVE API        │   │
                     │  │                             │   │
                     │  │  Video → Pose Estimation    │   │
                     │  │  Audio → Voice Commands     │   │
                     │  │  Context → Workout State    │   │
                     │  │                             │   │
                     │  │  Output:                    │   │
                     │  │  - Form corrections         │   │
                     │  │  - Rep counting             │   │
                     │  │  - Voice feedback           │   │
                     │  │  - Visual overlays          │   │
                     │  └──────────────┬──────────────┘   │
                     │                 │                  │
                     └─────────────────┼──────────────────┘
                                       │
                                       ▼
                              Response Stream:
                              - Audio: "Keep your back straight!"
                              - Data: {rep_count: 5, form_score: 85%}
                              - Overlay: Pose skeleton corrections
```

## 4.2 Latency Analysis

| Stage | Time | Notes |
|-------|------|-------|
| Camera capture | 50ms | 20 FPS = 50ms per frame |
| JPEG encode | 5-10ms | Hardware accelerated |
| WiFi TX | 20-50ms | Depends on signal quality |
| Internet RTT | 50-150ms | To Gemini servers |
| AI Processing | 100-300ms | Gemini inference |
| WiFi RX | 20-50ms | Response stream |
| Audio playback | 10ms | Buffer delay |
| **Total RTT** | **~300-600ms** | Acceptable for workout feedback |

## 4.3 On-Device vs Cloud Processing Decision

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PROCESSING STRATEGY                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ON-DEVICE (T5AI):                                                  │
│  ✓ Video preview (local display)                                   │
│  ✓ Audio VAD (voice activity detection)                            │
│  ✓ JPEG encoding                                                    │
│  ✓ Basic motion detection (frame diff)                             │
│  ✓ UI rendering (LVGL)                                             │
│  ✗ Pose estimation (too compute intensive)                         │
│  ✗ Complex NLP (needs LLM)                                         │
│                                                                     │
│  CLOUD (Gemini Live):                                               │
│  ✓ Pose estimation from video                                      │
│  ✓ Form analysis                                                    │
│  ✓ Rep counting                                                     │
│  ✓ Natural language feedback                                        │
│  ✓ Context memory (workout history)                                │
│  ✓ Audio generation (TTS)                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

# PART 5: BUILDING YOUR WORKOUT AI COACH

## 5.1 Hardware Bill of Materials

| Component | Recommended | Price Est. | Purpose |
|-----------|-------------|------------|---------|
| **Main Board** | TUYA_T5AI_EVB | $50-80 | Core compute + WiFi + Camera |
| **Display** | 2.4" IPS 480x480 | Included | Visual feedback |
| **Camera** | GC0308 (included) | Included | Video capture |
| **Microphone** | MEMS PDM | Included | Voice input |
| **Speaker** | 1W driver | Included | Audio output |
| **Battery** | 3.7V 2000mAh LiPo | $10 | Portability |
| **Enclosure** | 3D printed | $5-20 | Protection |
| **Stand/Mount** | Tripod adapter | $10 | Positioning |

**Total: ~$100-150 for prototype**

## 5.2 Software Architecture for Workout AI

```c
// Main application structure

typedef enum {
    STATE_IDLE,           // Waiting for user
    STATE_WORKOUT_SETUP,  // Selecting exercise
    STATE_ACTIVE,         // Streaming + analyzing
    STATE_REST,           // Between sets
    STATE_SUMMARY         // Session complete
} workout_state_t;

typedef struct {
    workout_state_t state;
    char exercise_name[32];
    int rep_count;
    int set_count;
    float form_score;
    bool voice_enabled;
} workout_session_t;

// Key modules you need to implement:
// 1. Camera streaming to Gemini
// 2. Gemini response parsing
// 3. Display overlay rendering
// 4. Audio feedback playback
// 5. State machine management
```

## 5.3 Gemini Live Integration Strategy

**Option A: Via Tuya Cloud Proxy**
```
Device → Tuya Cloud → Gemini API → Tuya Cloud → Device
        (MQTT)        (REST)        (MQTT)
```
- Pros: Uses existing Tuya infrastructure
- Cons: Extra latency, Tuya dependency

**Option B: Direct Gemini Connection**
```
Device ──────────► Gemini Live API ──────────► Device
        (WebSocket + HTTP)          (Stream)
```
- Pros: Lower latency, full control
- Cons: Need to implement auth, handle API directly

**Recommended: Option B with fallback to Option A**

## 5.4 Gemini Live API Integration Code

```c
// gemini_client.h
typedef struct {
    void (*on_text)(const char *text);
    void (*on_audio)(const uint8_t *data, size_t len);
    void (*on_pose)(const pose_keypoints_t *pose);
    void (*on_error)(int code, const char *msg);
} gemini_callbacks_t;

OPERATE_RET gemini_init(const char *api_key, gemini_callbacks_t *cbs);
OPERATE_RET gemini_start_session(const char *system_prompt);
OPERATE_RET gemini_send_video_frame(const uint8_t *jpeg, size_t len);
OPERATE_RET gemini_send_audio(const uint8_t *pcm, size_t len);
OPERATE_RET gemini_end_session(void);

// System prompt for workout AI:
const char *WORKOUT_SYSTEM_PROMPT =
    "You are a personal fitness coach AI. "
    "Analyze the video stream for exercise form. "
    "Count repetitions accurately. "
    "Provide real-time voice corrections: "
    "- Back posture "
    "- Knee alignment "
    "- Range of motion "
    "- Tempo and control "
    "Be encouraging but precise. "
    "Respond with JSON when counting reps: "
    "{\"rep\": N, \"form_score\": 0-100, \"correction\": \"...\"}";
```

## 5.5 Complete Implementation Plan

### Phase 1: Basic Setup (Week 1-2)
```
□ Order TUYA_T5AI_EVB development board
□ Set up development environment:
  - Install TuyaOpen SDK
  - Configure toolchain
  - Flash test firmware
□ Run lvgl2Camera example
□ Run your_chat_bot example
□ Get Tuya Cloud license
□ Get Gemini API key
```

### Phase 2: Camera Integration (Week 3-4)
```
□ Modify lvgl2Camera for continuous streaming
□ Implement JPEG frame queue
□ Add frame timestamp synchronization
□ Test WiFi streaming bandwidth
□ Optimize JPEG quality vs size
```

### Phase 3: Gemini Integration (Week 5-6)
```
□ Implement HTTP client for Gemini API
□ Handle authentication
□ Create video upload endpoint
□ Parse Gemini responses (JSON)
□ Implement audio response playback
```

### Phase 4: UI Development (Week 7-8)
```
□ Design workout selection UI
□ Create exercise instruction screens
□ Implement rep counter display
□ Add form score visualization
□ Create rest timer UI
```

### Phase 5: Voice Integration (Week 9-10)
```
□ Integrate ai_audio component
□ Add voice command detection
□ Implement TTS for AI responses
□ Create audio feedback system
□ Add volume controls
```

### Phase 6: Polish & Testing (Week 11-12)
```
□ End-to-end testing
□ Latency optimization
□ Power consumption tuning
□ Enclosure design
□ User testing
```

## 5.6 Key Files to Study

| File | Purpose | Priority |
|------|---------|----------|
| `apps/tuya.ai/your_chat_bot/src/app_chat_bot.c` | Main AI app pattern | HIGH |
| `apps/tuya.ai/ai_components/ai_audio/src/*` | Audio processing | HIGH |
| `examples/graphics/lvgl2Camera/src/*` | Camera + display | HIGH |
| `src/tuya_ai_service/wukong/*` | MCP integration | MEDIUM |
| `src/tuya_cloud_service/cloud/*` | Cloud connectivity | MEDIUM |
| `boards/T5AI/TUYA_T5AI_EVB/*` | Board configuration | HIGH |

## 5.7 Build Commands

```bash
# Clone repository
git clone https://github.com/tuya/TuyaOpen.git
cd TuyaOpen
git checkout hackathon2026

# Set up environment
source export.sh

# Select board
tos config_choice
# → Select T5AI_EVB

# Configure (optional)
tos menuconfig

# Build your_chat_bot as reference
cd apps/tuya.ai/your_chat_bot
tos build

# Flash to device
tos flash
```

---

# PART 6: TROUBLESHOOTING & TIPS

## 6.1 Common Issues

**Camera not working:**
```c
// Check camera initialization in Kconfig
CONFIG_ENABLE_EX_MODULE_CAMERA=y

// Verify buffer allocation
if (sg_camera_frame_buff == NULL) {
    PR_ERR("PSRAM allocation failed - check memory config");
}
```

**WiFi connection drops:**
```c
// Implement reconnection logic
static void __network_status_cb(netmgr_status_e status) {
    if (status == NETMGR_LINK_DOWN) {
        // Stop streaming
        // Queue reconnection
    }
}
```

**Audio feedback delayed:**
```c
// Reduce audio buffer size for lower latency
cfg.buffer_size = 1024;  // Instead of default 4096
```

## 6.2 Performance Optimization

```c
// 1. Use DMA for all data transfers
// 2. Double-buffer everything
// 3. Process in work queues, not callbacks
// 4. Compress video aggressively (JPEG quality 60-70)
// 5. Use Opus for audio (better than MP3)
// 6. Keep display updates async from network
```

## 6.3 Power Consumption

| State | Current | Battery Life (2000mAh) |
|-------|---------|------------------------|
| Idle | 10 mA | 200 hours |
| WiFi Connected | 80 mA | 25 hours |
| Streaming Video | 200 mA | 10 hours |
| Full Active | 350 mA | 5.7 hours |

---

# SUMMARY

You now understand:

1. **Physics**: How electricity, semiconductors, and transistors make computation possible
2. **Hardware**: What the T5AI chip contains and how it processes video/audio
3. **Software**: The TuyaOpen layered architecture from HAL to applications
4. **Data Flow**: How video/audio travels from sensors to cloud AI and back
5. **Implementation**: Concrete steps to build your workout AI coach

**Next Steps:**
1. Order the TUYA_T5AI_EVB board
2. Set up development environment
3. Run the example applications
4. Start modifying for your workout AI use case

The TuyaOpen framework gives you 80% of the infrastructure you need. Your main work is:
- Integrating Gemini Live API directly
- Building the workout-specific UI
- Creating the exercise recognition prompts
- Testing with real workout scenarios

Good luck with your build!
