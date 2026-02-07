# A Gesture-Centered Workflow for Human–AI Co-Creation

This is an interactive creation system combining Hand Gesture Drawing and Generative AI (Stable Diffusion).

The project aims to explore the iterative creation process between human hand drawing and AI generation: starting from hand-drawn sketches, AI generates images, and then users "harvest" local textures from the generated images to convert them into new "brushes/stamps" for the next round of creation. Trying to find the language between AI and Human.

[![Watch the Demo](https://img.youtube.com/vi/sKjc2r9NwXE/maxresdefault.jpg)](https://youtu.be/sKjc2r9NwXE)

## Core Features

*   Dual Hand Control:
    *   Right Hand (Positioning): Controls the position of the brush or stamp.
    *   Left Hand (Action): Making a fist represents "draw/stamp", opening the palm represents "hover/move".
*   AI Image Generation: Integrates Replicate API (Stable Diffusion XL) to instantly convert hand-drawn lines into refined images.
*   Interactive Texture Harvesting: Users can select specific areas in the AI-generated image to convert them into exclusive "gesture textures".
*   Gesture Memory and Reuse: The system records the user's drawing trajectory. When a similar trajectory is drawn again, the corresponding texture stamp is automatically triggered.
*   Iterative Creation Loop: Draw -> AI Generate -> Harvest Texture -> Draw again with Texture -> Generate more complex images.

## System Requirements

*   Python 3.8+
*   Webcam (Must be connected)
*   Replicate API Token (For AI image generation)

## Installation and Setup

1.  Install Dependencies
    
    ```bash
    pip install -r requirements.txt
    ```

2.  Setup API Key
    Create a .env file in the project root directory and fill in your Replicate API Token:
    
    ```env
    REPLICATE_API_TOKEN=your_replicate_api_token_here
    ```

## Operation Instructions

### 1. Launch Program
Execute main.py:

```bash
python main.py
```

### 2. Hand Control
The system uses MediaPipe to detect both hands:

*   Right Hand (Cursor & Mode)
    *   Index Finger Tip: Controls the center point of the brush or stamp.
    *   Thumb Up: Switches to "Stamp Mode" (Requires previously learned gestures).
    *   Thumb Down: Switches to "Paint Mode" (Regular lines).

*   Left Hand (Trigger Action)
    *   Fist: Draw / Stamp.
    *   Open Palm: Pause / Hover (Moving at this time will not draw lines).

### 3. Creation Workflow (State Machine)

1.  Drawing Phase
    *   Use the right hand to draw lines.
    *   Can also use "Stamp Mode" to stamp learned textures.
    *   Click "Generate AI Image" when finished.

2.  Generating Phase
    *   AI generates an image based on the current canvas.
    *   After generation, click "Next: Annotate & Learn" to enter annotation mode.

3.  Annotation Phase
    *   Use the mouse to Drag & Drop to select an area you like on the screen.
    *   The system will automatically analyze your drawing trajectory in that area and bind the image texture of that area to the trajectory.
    *   Press "Start Round 2" (or next round) to return to the drawing phase.
    *   New Feature: Now this trajectory becomes your new brush! When you draw a similar shape with your right hand, it will stamp the pattern you just selected.

## Project Structure

*   main.py: Main program, handling Pygame window, MediaPipe detection, and state machine logic.
*   stroke_recorder.py: Responsible for recording stroke trajectories, calculating similarity, and saving/loading learned gesture databases.
*   sd_generator.py: Handles connection with Replicate API and image generation.
*   data/: Stores learned gesture data (JSON format) and texture images (PNG).

## Keyboard Shortcuts

*   Q: Quit Program
*   R: Reset/Clear Canvas
*   Enter: Confirm/Next Step (Depending on current state)

---
Created for Final Project: Inquiry to CD


