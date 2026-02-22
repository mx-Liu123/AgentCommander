# Centralized Model Definitions for AgentCommander

MODEL_PRESETS = {
    'gemini': [
        'auto-gemini-3', 
        'gemini-3-pro-preview', 
        'gemini-3-flash-preview', 
        'gemini-2.0-flash-exp', 
        'gemini-1.5-pro-latest',
        'gemini-1.5-flash-latest'
    ],
    'qwen': [
        'qwen:coder-model',
        'qwen:vision-model'
    ],
    'claude': [
        'claude-cli'
    ],
    'opencode': [
        'opencode/big-pickle',
        'opencode/minimax-m2.5-free',
        'opencode/glm-5-free'
    ]
}

DEFAULT_MODEL = "auto-gemini-3"
