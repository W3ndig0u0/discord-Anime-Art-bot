def handle_response(message):
    p_message = message.lower()

    if p_message == "hi":
        return "Hello!"

    if p_message == "!help":
        return "This is a help message"

    # Default response for other cases
    return "What????"
