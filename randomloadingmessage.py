"""
Loading Screen Messages Generator

Generates humorous loading screen messages inspired by https://github.com/1egoman/funnies
"""

import random
from typing import Dict, List

# Define message categories with weights
MESSAGES: Dict[str, List[str]] = {
    "TECHNICAL": [
        "Reticulating splines...",
        "Swapping time and space...",
        "Spinning violently around the y-axis...",
        "Tokenizing real life...",
        "Bending the spoon...",
        "Filtering morale...",
        "We need a new fuse...",
        "Upgrading Windows, your PC will restart several times. Sit back and relax.",
        "The architects are still drafting.",
        "We're building the buildings as fast as we can.",
        "Please wait while the little elves draw your map.",
        "Don't worry - a few bits tried to escape, but we caught them.",
        "The server is powered by a lemon and two electrodes.",
        "Creating time-loop inversion field.",
        "Computing chance of success.",
        "All I really need is a kilobit.",
        "I feel like I'm supposed to be loading something...",
        "Should have used a compiled language...",
        "Is this Windows?",
        "Keeping all the 1's and removing all the 0's...",
        "Cracking military-grade encryption...",
        "Entangling superstrings...",
        "Dividing by zero...",
        "Installing dependencies.",
        "Switching to the latest JS framework...",
        "Ordering 1s and 0s...",
        "Updating dependencies...",
    ],
    "HUMOROUS": [
        "Have a good day.",
        "Go ahead -- hold your breath!",
        "...at least you're not on hold...",
        "We're testing your patience.",
        "As if you had any other choice.",
        "The bits are flowing slowly today.",
        "It's still faster than you could draw it.",
        "My other loading screen is much faster.",
        "(Insert quarter)",
        "Are we there yet?",
        "Just count to 10.",
        "Don't panic...",
        "We're making you a cookie.",
        "Don't break your screen yet!",
        "I swear it's almost done.",
        "Let's take a mindfulness minute...",
        "Listening for the sound of one hand clapping...",
        "We are not liable for any broken screens as a result of waiting.",
        "Where did all the internets go?",
        "Granting wishes...",
        "Time flies when you’re having fun.",
        "Get some coffee and come back in ten minutes...",
        "Stay awhile and listen...",
        "Convincing AI not to turn evil...",
        "How did you get here?",
        "Wait, do you smell something burning?",
        "Computing the secret to life, the universe, and everything.",
        "When nothing is going right, go left...",
        "I love my job only when I'm on vacation...",
        "Why are they called apartments if they are all stuck together?",
        "I’ve got a problem for your solution...",
        "Whenever I find the key to success, someone changes the lock.",
        "You don’t pay taxes—they take taxes.",
        "A commit a day keeps the mobs away.",
        "This is not a joke, it's a commit.",
        "Hello IT, have you tried turning it off and on again?",
        "Hello, IT... Have you tried forcing an unexpected reboot?",
        "I didn't choose the engineering life. The engineering life chose me.",
        "If I’m not back in five minutes, just wait longer.",
        "Web developers do it with <style>",
        "Looking for sense of humour, please hold on.",
        "A different error message? Finally, some progress!",
        "Please hold on as we reheat our coffee.",
        "Kindly hold on as we convert this bug to a feature...",
        "Kindly hold on as our intern quits vim...",
        "Winter is coming...",
        "Let's hope it's worth the wait.",
        "Aw, snap! Not...",
        "Please wait... Consulting the manual...",
        "Loading funny message...",
        "Feel free to spin in your chair.",
    ],
}

def get_random_message() -> str:
    """
    Select a random loading screen message from predefined categories.
    
    Returns:
        str: A randomly selected loading screen message.
    """
    # Combine all messages into a single list
    all_messages = [msg for category in MESSAGES.values() for msg in category]
    
    # Ensure there are messages to select from
    if not all_messages:
        return "No loading messages available."
    
    # Return a random message
    return random.choice(all_messages)

def generate_loading_messages(count: int = 1) -> None:
    """
    Generate and print random loading screen messages.
    
    Args:
        count (int): Number of messages to generate (default: 1).
    """
    for _ in range(count):
        print(get_random_message())

if __name__ == "__main__":
    # Generate 1 message by default (configurable via count parameter)
    generate_loading_messages(count=1)