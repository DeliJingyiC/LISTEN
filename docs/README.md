# LISTEN GitHub Pages Website

This directory contains the GitHub Pages website for the LISTEN benchmark.

## Files

- `index.html` - Main website structure
- `styles.css` - Styling and layout
- `script.js` - Interactive leaderboard functionality
- `README.md` - This file

## Updating the Leaderboard

To update the leaderboard with your actual results, edit the `leaderboardData` array in `script.js` (starting at line 1).

### Data Format

Each model entry should have the following structure:

```javascript
{
    model: "Model Name",
    type: "proprietary" | "opensource" | "baseline",
    exp1_text: { accuracy: 0.0, weighted_accuracy: 0.0, uar: 0.0, macro_f1: 0.0, micro_f1: 0.0 },
    exp1_audio: { accuracy: 0.0, weighted_accuracy: 0.0, uar: 0.0, macro_f1: 0.0, micro_f1: 0.0 },
    exp1_both: { accuracy: 0.0, weighted_accuracy: 0.0, uar: 0.0, macro_f1: 0.0, micro_f1: 0.0 },
    exp2_text: { accuracy: 0.0, weighted_accuracy: 0.0, uar: 0.0, macro_f1: 0.0, micro_f1: 0.0 },
    exp2_audio: { accuracy: 0.0, weighted_accuracy: 0.0, uar: 0.0, macro_f1: 0.0, micro_f1: 0.0 },
    exp2_both: { accuracy: 0.0, weighted_accuracy: 0.0, uar: 0.0, macro_f1: 0.0, micro_f1: 0.0 },
    exp3_text: { accuracy: 0.0, weighted_accuracy: 0.0, uar: 0.0, macro_f1: 0.0, micro_f1: 0.0 },
    exp3_audio: { accuracy: 0.0, weighted_accuracy: 0.0, uar: 0.0, macro_f1: 0.0, micro_f1: 0.0 },
    exp3_both: { accuracy: 0.0, weighted_accuracy: 0.0, uar: 0.0, macro_f1: 0.0, micro_f1: 0.0 },
    exp4_audio: { accuracy: 0.0, weighted_accuracy: 0.0, uar: 0.0, macro_f1: 0.0, micro_f1: 0.0 }
}
```

### Extracting Data from Results

Your test scripts output comprehensive metrics in JSON format. To extract data:

1. Run your test scripts (they save results to `results/` directory)
2. Look for the `comprehensive_metrics` field in each result JSON file
3. Copy the metrics for each experiment variant

Example from a results JSON:

```json
{
  "experiment": "experiment1_text",
  "accuracy": 85.2,
  "comprehensive_metrics": {
    "weighted_accuracy": 84.8,
    "uar": 82.3,
    "macro_f1": 83.1,
    "micro_f1": 85.2
  }
}
```

### Helper Script

You can create a Python script to automatically extract and format the data:

```python
import json
import glob

# Read all result files
results = {}
for file in glob.glob('results/*.json'):
    with open(file) as f:
        data = json.load(f)
        model = data.get('model', 'Unknown')
        exp = data.get('experiment', '')
        
        if model not in results:
            results[model] = {}
        
        results[model][exp] = {
            'accuracy': data.get('accuracy', 0),
            'weighted_accuracy': data['comprehensive_metrics'].get('weighted_accuracy', 0),
            'uar': data['comprehensive_metrics'].get('uar', 0),
            'macro_f1': data['comprehensive_metrics'].get('macro_f1', 0),
            'micro_f1': data['comprehensive_metrics'].get('micro_f1', 0)
        }

# Print formatted JavaScript
print("const leaderboardData = [")
for model, exps in results.items():
    print(f"    {{")
    print(f"        model: \"{model}\",")
    print(f"        type: \"opensource\",  // Change as needed")
    for exp_name, metrics in exps.items():
        exp_key = exp_name.replace('experiment', 'exp').replace('_', '_')
        print(f"        {exp_key}: {{ accuracy: {metrics['accuracy']:.1f}, weighted_accuracy: {metrics['weighted_accuracy']:.1f}, uar: {metrics['uar']:.1f}, macro_f1: {metrics['macro_f1']:.1f}, micro_f1: {metrics['micro_f1']:.1f} }},")
    print(f"    }},")
print("];")
```

## Setting Up GitHub Pages

### Option 1: Using the `docs/` folder (Recommended)

1. Go to your GitHub repository settings
2. Navigate to "Pages" in the left sidebar
3. Under "Source", select "Deploy from a branch"
4. Under "Branch", select `main` and `/docs` folder
5. Click "Save"
6. Your site will be available at `https://yourusername.github.io/LISTEN/`

### Option 2: Using a separate `gh-pages` branch

1. Create a new branch called `gh-pages`:
   ```bash
   git checkout --orphan gh-pages
   git rm -rf .
   ```

2. Copy the contents of the `docs/` folder to the root:
   ```bash
   cp docs/* .
   ```

3. Commit and push:
   ```bash
   git add .
   git commit -m "Initial GitHub Pages setup"
   git push origin gh-pages
   ```

4. Go to repository settings → Pages
5. Select `gh-pages` branch and `/ (root)` folder
6. Click "Save"

## Customization

### Changing Colors

Edit the CSS variables in `styles.css`:

```css
:root {
    --primary-color: #2563eb;
    --secondary-color: #7c3aed;
    --accent-color: #06b6d4;
    /* ... */
}
```

### Adding More Sections

Add new sections in `index.html` following the existing pattern:

```html
<section id="your-section" class="section">
    <div class="container">
        <h2>Your Section Title</h2>
        <!-- Your content -->
    </div>
</section>
```

### Updating Citation

Edit the citation in `index.html` (search for `@misc{deli2025listen`).

## Testing Locally

To test the website locally:

1. Install a simple HTTP server:
   ```bash
   # Python 3
   python -m http.server 8000
   
   # Or Node.js
   npx http-server
   ```

2. Navigate to the `docs/` directory:
   ```bash
   cd docs/
   python -m http.server 8000
   ```

3. Open `http://localhost:8000` in your browser

## Features

- **Interactive Leaderboard**: Sortable by any column
- **Metric Filtering**: Switch between accuracy, weighted accuracy, UAR, macro F1, and micro F1
- **Experiment Filtering**: View results for specific experiments
- **Search**: Filter models by name
- **Responsive Design**: Works on mobile and desktop
- **Smooth Animations**: Professional transitions and hover effects
- **Copy Citation**: One-click BibTeX copying

## Troubleshooting

### Site not updating after push

- Clear your browser cache
- Wait a few minutes for GitHub Pages to rebuild
- Check the "Actions" tab in your repository for build status

### Custom domain

To use a custom domain:

1. Add a `CNAME` file to the `docs/` directory with your domain
2. Configure DNS settings with your domain provider
3. Enable HTTPS in repository settings

## Support

For issues or questions, please open an issue on the [GitHub repository](https://github.com/DeliJingyiC/LISTEN).

