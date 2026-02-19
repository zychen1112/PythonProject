---
name: code-explainer
description: Explains code with clear examples and analogies. Use when the user asks about how code works or wants to understand a codebase.
license: MIT
metadata:
  author: pyagent
  version: "1.0"
  tags:
    - code
    - explanation
    - learning
allowed-tools: Read Grep Glob
---

# Code Explainer Skill

This skill helps explain code in a clear and understandable way.

## Instructions

When explaining code:

1. **Start with an overview**: Briefly describe what the code does overall
2. **Use analogies**: Compare the code to real-world concepts when helpful
3. **Break it down**: Explain each significant part step by step
4. **Highlight key concepts**: Point out important patterns or techniques
5. **Provide examples**: Show how the code would be used

## Code Analysis Process

1. Read the relevant files using the Read tool
2. Search for related code using Grep if needed
3. Understand the structure and flow
4. Explain in simple terms

## Example

When explaining a function:
```
This function `calculate_total` is like a cashier adding up items:
- It takes a list of prices (items)
- Adds them together (sums the values)
- Returns the total (gives you the final amount)
```
