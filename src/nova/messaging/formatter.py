"""Format NOVA responses for messaging platforms.

Each platform has different formatting support:
- Telegram: MarkdownV2 or HTML (we keep it simple)
- WhatsApp: *bold*, _italic_, ~strikethrough~, ```monospace```
"""


def format_for_telegram(text: str) -> str:
    """Format response for Telegram.

    Telegram supports MarkdownV2 or HTML. We keep formatting minimal
    since the LLM already produces clean text from RULES.md compliance.

    Args:
        text: Raw response text from the LLM.

    Returns:
        Formatted text suitable for Telegram.
    """
    return text


def format_for_whatsapp(text: str) -> str:
    """Format response for WhatsApp.

    WhatsApp supports limited formatting:
    - *bold*
    - _italic_
    - ~strikethrough~
    - ```monospace```

    Args:
        text: Raw response text from the LLM.

    Returns:
        Formatted text suitable for WhatsApp.
    """
    return text
