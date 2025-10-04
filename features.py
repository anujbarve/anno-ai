import random
from memory import ConversationMemory

class FeatureManager:
    """Manages additional features and capabilities"""
    
    def __init__(self, memory: ConversationMemory):
        self.memory = memory
        self.active_timers = {}
        
    def set_reminder(self, content: str, time_str: str) -> str:
        """Set a reminder (simplified version)"""
        self.memory.add_memory(
            f"Reminder: {content} at {time_str}",
            importance=0.8,
            category="reminder"
        )
        return f"I'll remind you to {content} at {time_str}."
    
    def get_fun_fact(self) -> str:
        """Return a random fun fact"""
        facts = [
            "Did you know? The term 'computer bug' originated when a real moth was found in a Harvard computer in 1947.",
            "Fun fact: Honey never spoils. Archaeologists have found 3000-year-old honey that's still edible!",
            "Here's something interesting: Octopi have three hearts and blue blood.",
            "Did you know? The first computer programmer was Ada Lovelace in the 1840s.",
            "Mythology fact: In Greek mythology, Hades' three-headed dog Cerberus guards the underworld.",
            "The longest word you can type using only the left hand is 'stewardesses'.",
            "A group of flamingos is called a 'flamboyance'.",
            "The first oranges weren't orange - they were green!"
        ]
        return random.choice(facts)