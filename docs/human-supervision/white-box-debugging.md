# White-box Debugging

AgentCommander allows you to intervene at any step.

## Resume Conversation (`gemini -r`)
Every experiment runs in its own folder. To debug:
1.  Open terminal.
2.  `cd` into the experiment folder.
3.  Run `gemini -r` (or `qwen -c`).
4.  You are now in the *exact context* of the agent. You can ask 'Why did you write this?' or manually guide it.
