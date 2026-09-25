# This is utiltiy for logging colors
# It's similar to agent of logging
# Foreground colors
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'
WHITE = '\033[37m'

# Background color
BG_BLACK = '\033[40m'
BG_BLUE = '\033[44m'

# Reset code to return to default color
RESET = '\033[0m'

mapper = {
    BG_BLACK+RED: "#dd0000",
    BG_BLACK+GREEN: "#00dd00",
    BG_BLACK+YELLOW: "#dddd00",
    BG_BLACK+BLUE: "#0000ee",
    BG_BLACK+MAGENTA: "#aa00dd",
    BG_BLACK+CYAN: "#00dddd",
    BG_BLACK+WHITE: "#87CEEB",
    BG_BLUE+WHITE: "#ff7800"
}

# Reformat message
def reformat(message):
    for key, value in mapper.items():
        # Convert color to html snippet
        message = message.replace(key, f'<span style="color: {value}">')
    # For reset case    
    message = message.replace(RESET, '</span>')
    return message
    
    