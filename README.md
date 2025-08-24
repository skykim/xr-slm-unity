# xr-smollm2-135m-instruct-unity

XR On-Device SLM: SmolLM2-135M-Instruct in Unity Inference Engine

## Overview

This repository provides a lightweight, cross-platform inference engine optimized for [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), a compact small language model (SLM) from Hugging Face's SmolLM2 series. SmolLM2-135M delivers exceptional performance in a compact size, making it perfect for on-device applications even on the Meta Quest 3.

## Features

- ✅ Support English
- ✅ GPT2Tokenizer implemented in C#
- ✅ On-device processing: No internet connection required
- ✅ Quantized model: Uint8 (164MB)

## Requirements

- **Unity**: `6000.2.0f1`
- **Inference Engine**: `2.3.0`

## Architecture

### 1. GPT2Tokenizer in C#

BPE-based tokenizer ported from HuggingFace [GPT2 Tokenizer](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gpt2) Python implementation to C#.

### 2. SmolLM2-135M-Instruct (ONNX)

You can download the KV-cache-optimized architecture using `optimum-cli`, or easily download a pre-built ONNX file from [onnx-community](https://huggingface.co/onnx-community/SmolLM2-135M-ONNX).

## Getting Started

### 1. Project Setup

- Clone or download this repository
- Unzip the provided [Model.zip](https://drive.google.com/file/d/1qspG-1Biwi3IZVaKM-6rTu0TBU364oEq/view?usp=drive_link) file and place its contents into the `/Assets/Model` directory in your project

### 2. Run the Demo Scene

- Open the `/Assets/Scenes/XRSLMScene.unity` scene in the Unity Editor
- Run the scene to see the SLM in action

## Demo

Experience smollm2-135m-instruct-unity in action! Check out our demo showcasing the model's capabilities:

[![SmolLM2-135M-Instruct Unity](https://img.youtube.com/vi/ByIGusacdi0/0.jpg)](https://www.youtube.com/watch?v=ByIGusacdi0)

## Links

- [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- [Onnx Community: SmolLM2-135M-Instruct](https://huggingface.co/onnx-community/SmolLM2-135M-ONNX)
- [GPT2 Tokenizer](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gpt2)

## License

SmolLM2 is licensed under the Apache 2.0 License.
