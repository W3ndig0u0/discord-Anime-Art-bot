def handle_response(message) -> str:
    p_message = message.lower()

    if p_message == "hi":
        return "Hello!"

    if p_message == "?help":
        return "This is a help message"

    return "I'm sorry, I didn't understand your message."
