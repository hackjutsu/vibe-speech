# Voice-to-Text Coding Assistant: Proposed Architecture

## Overview

This project will enable hands-free coding by converting spoken language into text and automatically inputting that text into a focused terminal or editor window. The user can configure whether the text is simply transcribed or pre-processed (summarized/cleaned up) before being entered.

## Components

1. **Speech-to-Text Model:**
   - **Tool:** OpenAI Whisper (open-source, local)
   - **Purpose:** Listens to the microphone and converts speech into text.

2. **Text Input Automation:**
   - **macOS:** AppleScript or Hammerspoon to focus the terminal/editor and simulate typing.
   - **Windows:** AutoHotkey for similar automation.
   - **Cross-Platform:** Python with `pyautogui` to handle cross-OS text input.

3. **Configuration and Toggle:**
   - **Listening Toggle:** A single toggle (button or hotkey) to start/stop the transcription process.
   - **Mode Configuration:** A setting (in a config file or small UI) to choose between direct transcription or summarization/cleanup before input.

## Workflow

1. User turns on the listening toggle.
2. User selects the terminal or text editor window to focus.
3. The program listens and transcribes speech into text via Whisper.
4. Depending on the configuration, the text is either directly typed or processed (summarized/cleaned) before being typed into the focused window.

---

Feel free to copy this Markdown into your project notes!
