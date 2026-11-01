# Max Profit Problem

**Live demo:** https://himanshi-2602.github.io/gen-ai-engineer-assignments/max-profit/

## Problem

Mr. X owns a large strip of land in Mars Land, with unlimited space. On each parcel he can build a Theatre, a Pub or a Commercial Park:

| Building | Build time | Land | Earnings per unit of time while operational |
|---|---|---|---|
| Theatre (T) | 5 units | 2x1 | $1500 |
| Pub (P) | 4 units | 1x1 | $1000 |
| Commercial Park (C) | 10 units | 3x1 | $2000 |

- Each unit of time that a building is operational, it earns money.
- **Two properties cannot be under construction at the same time.**
- After `n` units of time (the input), he earns money based on the properties that have been developed.
- Output: the maximum earnings, and every mix of buildings that achieves it, written as `T: <count> P: <count> C: <count>`.

**Test cases**

| | Test Case 1 | Test Case 2 | Test Case 3 |
|---|---|---|---|
| Input | Time Unit: 7 | Time Unit: 8 | Time Unit: 13 |
| Earnings | $3000 | $4500 | $16500 |
| Solutions | 1. T: 1 P: 0 C: 0<br>2. T: 0 P: 1 C: 0 | 1. T: 1 P: 0 C: 0 | 1. T: 2 P: 0 C: 0 |

## Solution

### Understanding the earnings

Buildings are built one after another. A building that finishes at time `t` earns its rate for each of the `n − t` units left:

```
earnings of one building = rate × (n − finish time)
```

Test case 3 (n = 13), building Theatre then Theatre:

```
Theatre 1: built 0–5,  earns 1500 × (13 − 5)  = 12000
Theatre 2: built 5–10, earns 1500 × (13 − 10) =  4500
                                        Total = 16500 ✓
```

A building that finishes at or after `n` earns nothing, so it is never counted.

### Dynamic programming

Let `best[r]` be the maximum earnings when `r` units of time remain. Whatever we build first takes `b` units, then earns `rate` for each of the `r − b` units left, and we face the same problem with `r − b` units left:

```
best[r] = max over buildings with b < r of:  rate × (r − b) + best[r − b]
best[r] = 0  if no building can finish in time (r ≤ 4)
```

This is like the unbounded knapsack problem, except an item's value depends on the time left when it is added. The DP considers every build order, so no ordering argument is needed.

**Table for n = 13**

| r | Theatre: 1500·(r−5) + best[r−5] | Pub: 1000·(r−4) + best[r−4] | Commercial: 2000·(r−10) + best[r−10] | best[r] | Choice |
|---|---|---|---|---|---|
| 0–4 | – | – | – | 0 | – |
| 5 | – | 1000 + 0 | – | 1000 | P |
| 6 | 1500 + 0 | 2000 + 0 | – | 2000 | P |
| 7 | 3000 + 0 | 3000 + 0 | – | **3000** | T or P ✓ Test 1 |
| 8 | 4500 + 0 | 4000 + 0 | – | **4500** | T ✓ Test 2 |
| 9 | 6000 + 0 | 5000 + 1000 | – | 6000 | T or P |
| 10 | 7500 + 1000 | 6000 + 2000 | – | 8500 | T |
| 11 | 9000 + 2000 | 7000 + 3000 | 2000 + 0 | 11000 | T |
| 12 | 10500 + 3000 | 8000 + 4500 | 4000 + 0 | 13500 | T |
| 13 | 12000 + 4500 | 9000 + 6000 | 6000 + 0 | **16500** | T ✓ Test 3 |

**Recovering the mix.** We store the winning choice(s) for each `r` and trace back from `r = n`: 13 → T → 8 → T → 3 (nothing fits), giving `T: 2 P: 0 C: 0`. When choices tie, we follow every tied branch and keep the distinct `(T, P, C)` counts. That is how n = 7 produces both solutions.

**Build order for the timeline.** Swapping two adjacent buildings A and B changes the earnings by `rateA × timeB − rateB × timeA`. So the best order for a fixed mix is the highest earnings per unit of build time first: Theatre (300) → Pub (250) → Commercial (200). The UI draws each solution in that order.

```js
for (let r = 1; r <= n; r++) {
  BUILDINGS.forEach(({ time, rate }, k) => {
    if (time >= r) return;
    const value = rate * (r - time) + best[r - time];
    if (value > best[r]) { best[r] = value; choices[r] = [k]; }
    else if (value === best[r]) choices[r].push(k);
  });
}
```

### Complexity

| Approach | Time | Space |
|---|---|---|
| Try every build sequence | Exponential | O(n) |
| Try every mix of counts (with the T → P → C order) | O(n³) | O(1) |
| **Dynamic programming (used here)** | **O(n · k) = O(n)** | **O(n)** |
| DP, earnings only (keep only the last 10 values of `best`) | O(n) | O(1) |

`k = 3` building types. The table is kept, giving O(n) space, so that the mix can be traced back. Listing tied solutions adds only a little extra work, because the number of distinct tied mixes is small.

**Observation:** a Commercial Park is never part of the best mix. It has the lowest earnings per unit of build time, and its 10-unit build delays everything after it.

## Features

- Enter `n` and see the maximum earnings and **every** optimal mix, in the format the problem asks for.
- **SVG build timeline** for each solution, showing the construction period and the earning period of every building, with what each one earns.
- The **dynamic programming table** (`best[r]` and the best first build for each `r`) to show how the answer was reached.
- Input validation, one-click test cases, and a responsive layout. Vanilla JS, HTML and CSS, with no dependencies.

## Project Structure

```
max-profit/
├── index.html          # Page markup
├── styles.css          # Styling
├── profit.js           # Core algorithm (maxProfit, schedule, parseTime), shared by browser and Node
├── app.js              # UI logic, SVG timeline and DP table rendering
├── tests/
│   └── profit.test.js  # Unit tests (Node built-in test runner)
└── package.json        # npm test script
```

## Running Locally

No build step. Open `index.html` in a browser, or serve the folder:

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

## Running Tests

Requires Node.js 18 or later:

```bash
npm test
```

The tests cover the three test cases from the problem, ties (n = 9 → `T: 1` or `P: 2`), and the edge case n ≤ 4. They also check that each solution's schedule adds up to the reported earnings for n = 0..120. The DP is compared against a **brute-force search over every build sequence** for n = 0..40, matching both the earnings and the exact set of solutions.
