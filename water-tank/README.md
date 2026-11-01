# Water Tank Problem

A frontend web application built with **Vanilla JavaScript, HTML and CSS** that computes how many units of water are stored between blocks of given heights, and visualizes the result as **SVG** grids.

## Problem

Given an array of `n` block heights (each height is a whole number greater than -1), compute the units of water stored in between the blocks.

**Example**

```
Input:  [0,4,0,0,0,6,0,6,4,0]
Output: 18 Units
```

The 4-high block and the 6-high block hold 4 units over each of the 3 empty columns between them (12 units). The two 6-high blocks hold 6 units over the column between them (6 units). That makes 12 + 6 = **18 units**.

## Solution

### Approach

The water above any column `i` is limited by the shorter of the tallest wall on its left and the tallest wall on its right:

```
water[i] = min(maxLeft(i), maxRight(i)) - height[i]
```

#### Two-pointer algorithm, O(n) time and O(1) extra space

1. Put one pointer at each end of the array and keep a running `leftMax` and `rightMax`.
2. Move the pointer on the side with the **lower** height. That side's running max is the limiting wall: a wall at least as tall is known to exist on the other side, so the water level at that column is already fixed.
3. Add `runningMax - height` at that column to the total, then move the pointer inward.
4. Stop when the pointers meet.

```js
while (left < right) {
  if (heights[left] < heights[right]) {
    leftMax = Math.max(leftMax, heights[left]);
    water[left] = leftMax - heights[left];
    left++;
  } else {
    rightMax = Math.max(rightMax, heights[right]);
    water[right] = rightMax - heights[right];
    right--;
  }
}
```

The algorithm also records the water above each column (`water[i]`), which the UI uses to draw the visualization.

| Approach | Time | Extra space |
|---|---|---|
| Brute force (scan left/right for each column) | O(n²) | O(1) |
| Prefix and suffix max arrays | O(n) | O(n) |
| **Two pointers (used here)** | **O(n)** | **O(1)** |

## Features

- Accepts input as `[0,4,0,6]`, `0,4,0,6` or `0 4 0 6`.
- Validates input: only non-negative whole numbers are accepted, with clear error messages.
- **Input chart**: an SVG grid with blocks (yellow) and stored water (blue), matching the assignment diagram.
- **Output chart**: an SVG grid that shows only the stored water, with the units per column.
- One-click example inputs.
- Responsive layout. No frameworks, build step or dependencies.

## Project Structure

```
water-tank/
├── index.html          # Page markup
├── styles.css          # Styling
├── water.js            # Core algorithm (parseHeights, computeWater), shared by browser and Node
├── app.js              # UI logic and SVG rendering
├── tests/
│   └── water.test.js   # Unit tests (Node built-in test runner)
└── package.json        # npm test script
```

## Running Locally

There is no build step. Either:

- Open `index.html` directly in a browser, **or**
- Serve the folder:

  ```bash
  npx serve .
  # or
  python3 -m http.server 8000
  ```

  Then open http://localhost:8000.

## Running Tests

Requires Node.js 18 or later:

```bash
npm test
```

The tests cover the assignment example, well-known cases and edge cases (empty input, a single block, increasing and decreasing heights, all zeros). They also check the algorithm against a brute-force solution on 500 random inputs and test input parsing and validation.

## Snippets

**Input:** `[0,4,0,0,0,6,0,6,4,0]` → **Output:** `18 Units`

| Column | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Height | 0 | 4 | 0 | 0 | 0 | 6 | 0 | 6 | 4 | 0 |
| Water  | 0 | 0 | 4 | 4 | 4 | 0 | 6 | 0 | 0 | 0 |
