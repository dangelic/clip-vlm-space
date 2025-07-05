# CLIP Space Image Classifier

A Python tool for classifying space and astronomy images using OpenAI's CLIP model. No training required—just use the pretrained model to get started right away.

## Features

- Zero-shot classification with CLIP (no training needed)
- Recognizes 28+ astronomy and space categories
- Works with local images or URLs
- Visualization of classification results
- Batch filtering for datasets

## ⚠️ Python & Virtual Environment Setup (Important!)

**If you use Homebrew Python on macOS, or just want to avoid breaking your system Python, you must use a virtual environment.**

> **Why?**
> Homebrew and recent Python versions prevent installing packages globally to protect your system. This project will not install correctly unless you use a venv.

### Quick Setup

1. **Clone the repository:**
   ```sh
   git clone <repository-url>
   cd clip-vlm-space
   ```
2. **Create a virtual environment:**
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```
   - If you see `(venv)` at the start of your prompt, you're good to go.
3. **Install dependencies:**
   ```sh
   python install.py
   ```
   - The script will check your environment and help you if you missed a step.

---

## Usage

### Basic Classification

```python
from space_clip_classifier import SpaceCLIPClassifier

classifier = SpaceCLIPClassifier()
image_path = "path/to/your/space/image.jpg"
predictions = classifier.classify_image(image_path, top_k=5)

for category, confidence in predictions:
    print(f"{category}: {confidence:.3f}")
```

### Visualize Results

```python
classifier.visualize_classification(image_path, top_k=5)
```

### Run Demo

```sh
python demo.py
```

### Download NASA Space Images

```sh
python nasa_space_filter.py
```

**First time setup:**
1. Get a free NASA API key at https://api.nasa.gov/
2. Copy the sample config file:
   ```bash
   cp config.env.sample config.env
   ```
3. Edit `config.env` with your API key:
   ```bash
   # Replace DEMO_KEY with your actual API key
   NASA_API_KEY=your_api_key_here
   ```

---

## Space Categories

The classifier recognizes a range of space-related categories, including:
- galaxy, nebula, star cluster, planet, moon, asteroid, comet, black hole, supernova
- cosmic dust, solar system, constellation, meteor, aurora borealis
- satellite, space station, rocket, telescope, astronaut, space suit, spacecraft, mars rover
- hubble telescope, james webb telescope, ISS, space debris, milky way, andromeda galaxy, orion nebula

---

## How It Works

1. Loads the pretrained CLIP model from HuggingFace
2. Processes your image and compares it to a list of astronomy-related text prompts
3. Returns the most likely categories and confidence scores

---

## Example Output

```
=== Space Image Classification with CLIP ===

Image 1:
  1. galaxy: 0.234
  2. nebula: 0.189
  3. star cluster: 0.156

Image 2:
  1. stars: 0.312
  2. constellation: 0.245
  3. milky way: 0.198
```

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PIL/Pillow
- Matplotlib
- NumPy

All dependencies are listed in `requirements.txt` and will be installed by the setup script.

---

## Troubleshooting

- **Not in a venv?**
  - If you see errors about "externally managed environment" or "PEP 668", you need to activate your venv (see above).
- **First run is slow?**
  - The CLIP model will be downloaded the first time you use it (~500MB).
- **Want to use your own images?**
  - Just pass a local file path or URL to the classifier.

---

## License

MIT License. See `LICENSE` file for details. 