
import dspy
from .diff_types import PRInfo
from .config import SUB_MODEL

class SuggestionSignature(dspy.Signature):
    """Generate 4-5 short, relevant follow-up questions or actions for the user."""
    
    pr_context: str = dspy.InputField(desc="Summary of the PR context")
    conversation: str = dspy.InputField(desc="Recent conversation history")
    last_answer: str = dspy.InputField(desc="The last answer provided by the assistant")
    
    suggestions: list[str] = dspy.OutputField(desc="List of 4-5 short suggestion strings (max 5 words each)")

class SuggestionGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use SUB_MODEL (likely a faster/cheaper model like Gemini Flash)
        self.generate = dspy.Predict(SuggestionSignature)
        
    def forward(self, pr_info: PRInfo, conversation: list[dict], last_answer: str):
        # Format inputs
        pr_context = f"PR: {pr_info.title}\n{pr_info.body[:500]}"
        
        # Get last few messages
        recent_msgs = conversation[-3:] if conversation else []
        conv_text = "\n".join([f"{m['role']}: {m['content'][:200]}" for m in recent_msgs])
        
        result = self.generate(
            pr_context=pr_context,
            conversation=conv_text,
            last_answer=last_answer[:500]
        )
        
        return result.suggestions

# Singleton instance
_generator = None

def get_suggestion_generator():
    global _generator
    if _generator is None:
        # Ensure dspy is configured (should be done by main app, but safe to check)
        # We assume dspy.configure is called elsewhere with the correct LMs
        _generator = SuggestionGenerator()
    return _generator
