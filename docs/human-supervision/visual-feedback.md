# Visual Feedback Loop

AgentCommander supports multimodal analysis.

## Mechanism
1.  **Generate**: Your code generates a plot (e.g., `plot.png`).
2.  **Feed Back**: In the workflow, reference the file using `@plot.png` in the prompt template.
3.  **Analyze**: Vision-capable models (Gemini Pro Vision, Qwen-VL) will see the image and analyze metrics like overfitting or bias.
