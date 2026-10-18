# ML in Motion

An interactive website that teaches machine learning through animated 3D simulations built with
[Three.js](https://threejs.org/). Every lesson runs the real algorithm live in the browser, so you can
watch it learn, change its settings, and orbit around the scene.

## Lessons

| # | Topic | What you see |
|---|-------|--------------|
| 00 | What is Machine Learning? | A particle cloud morphing from noise into clusters, a surface and a helix |
| 01 | Linear Regression | A line fitted by gradient descent, with residuals shrinking |
| 02 | Gradient Descent | Two balls (plain GD and momentum) racing down a 3D loss landscape |
| 03 | Overfitting & Regularization | Polynomial fits from degree 0–14, a train/test error curve and ridge λ |
| 04 | Logistic Regression | A sigmoid probability sheet and decision boundary learning to separate classes |
| 05 | k-Nearest Neighbors | A moving query point, its k neighbours and a live decision map |
| 06 | Decision Trees | Gini-based splits rising as walls that partition the plane |
| 07 | Support Vector Machines | Max-margin training with support vectors, plus a 3D kernel-trick lift |
| 08 | k-Means Clustering | 3D assign/update iterations until convergence |
| 09 | PCA | Principal axes of a 3D cloud, and projection to 2D and 1D |
| 10 | Neural Networks & Backprop | Forward-pass pulses, backpropagated gradients and live weight updates |
| 11 | CNNs | A kernel sliding over an image to build a feature map, with optional ReLU |
| 12 | Attention & Transformers | Softmax attention arcs between tokens in a sentence |
| 13 | Q-Learning | An agent learning a grid world; values and the policy spread from the goal |

## Running locally

The site is plain HTML/CSS/JS ES modules with no build step. Three.js loads from a CDN through an
import map, so any static file server works:

```bash
python3 -m http.server 8000
# or
npx http-server -p 8000
```

Then open <http://localhost:8000>. Opening `index.html` directly from disk won't work because browsers
block ES module imports over `file://`.

## Deploying

Because it's fully static, you can publish the repository as-is with **GitHub Pages** (Settings → Pages →
deploy from branch, root folder), Netlify, Vercel, or any static host.

## Project structure

```
index.html            App shell and import map
css/style.css         Layout and theme
js/main.js            Renderer, camera, navigation, lesson UI builder
js/helpers.js         Shared Three.js utilities (instanced points, labels, colours, RNG)
js/lessons/*.js       One file per lesson
js/lessons/index.js   Lesson order shown in the sidebar
```

## Adding a lesson

Create `js/lessons/my-topic.js` that exports a lesson object, then add it to `js/lessons/index.js`:

```js
export default {
  id: 'my-topic',                 // used in the URL hash (#my-topic)
  title: 'My Topic',
  category: 'Supervised Learning',
  camera: { position: [0, 0, 16], target: [0, 0, 0] },
  formula: 'ŷ = f(x)',             // shown in the panel (HTML allowed)
  hint: 'One-line caption under the scene',
  info: '<h3>What it is</h3><p>…</p>',
  create({ scene, ui, setCamera, onClick }) {
    // Build Three.js objects into `scene`; add controls with
    // ui.slider / ui.button / ui.select / ui.toggle / ui.stat / ui.html.
    return {
      update(dt, elapsed) { /* advance the simulation each frame */ },
      dispose() { /* optional extra cleanup */ },
    };
  },
};
```

Scene objects are disposed automatically when you switch lessons.

## Controls

- Drag to orbit, scroll or pinch to zoom
- `Space` pauses or resumes, `←` / `→` switch lessons
- The **Speed** slider scales every simulation
