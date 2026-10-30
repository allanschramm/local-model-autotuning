CONFIG_OPTIONS = [
    {
        "flag": "--provider",
        "env": "DM_PROVIDER",
        "default": "deepseek",
        "description": "LLM provider name.",
    },
    {
        "flag": "--timeout",
        "env": "DM_TIMEOUT",
        "default": 120,
        "description": "Provider request timeout in seconds.",
    },
    {
        "flag": "--model",
        "env": "DM_MODEL",
        "default": "deepseek-chat",
        "description": "Model identifier passed to the provider.",
    },
    {
        "flag": "--retries",
        "env": "DM_RETRIES",
        "default": 2,
        "description": "Retry count for transient provider failures.",
    },
]


def render_config_table(options=None):
    options = options or CONFIG_OPTIONS
    visible = [item for item in options if item['flag'] in {'--provider', '--timeout'}]
    lines = [
        "| Option | Env | Default | Description |",
        "| --- | --- | --- | --- |",
    ]
    for item in visible:
        lines.append(
            f"| {item['flag']} | {item['env']} | {item['default']} | {item['description']} |"
        )
    return "\n".join(lines)
