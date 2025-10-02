import os
from groq import Groq

def enhance_conversation_ai(conversation_module):
    """Add Groq AI to conversation module"""
    
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    def groq_generate(text, context):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": f"You are {behavior_config.robot_name}, a helpful robot."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content
    
    # Replace the AI generator
    conversation_module._generate_ai_response = groq_generate