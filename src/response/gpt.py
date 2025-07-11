from openai import OpenAI
import json
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
from enum import Enum
from loguru import logger
from dotenv import load_dotenv
load_dotenv()

class RiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class EmotionalState(Enum):
    CALM = "calm"
    ANXIOUS = "anxious"
    DEPRESSED = "depressed"
    ANGRY = "angry"
    DISTRESSED = "distressed"
    CRISIS = "crisis"

class OmaniTherapistAI:
    def __init__(self, api_key: str = None):
        """
        Initialize the OMANI Therapist AI system
        
        Args:
            api_key: OpenAI API key (if not provided, will use environment variable)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Session management
        self.conversation_history = []
        self.user_profile = {}
        self.emotional_state = EmotionalState.CALM
        self.risk_level = RiskLevel.LOW
        
        # Crisis keywords in Arabic and English
        self.crisis_keywords = {
            'arabic': [
                'انتحار', 'أريد أن أموت', 'لا أستطيع المتابعة', 'أريد أن أؤذي نفسي',
                'أفكر في الموت', 'لا قيمة لي', 'أريد أن أنهي حياتي', 'لا أحد يهتم بي',
                'أشعر بالفراغ', 'أريد أن أختفي', 'أكره نفسي', 'أريد أن أؤذي شخصاً'
            ],
            'english': [
                'suicide', 'kill myself', 'end my life', 'hurt myself', 'self harm',
                'want to die', 'no point living', 'better off dead', 'hate myself',
                'worthless', 'hopeless', 'can\'t go on', 'want to hurt someone'
            ]
        }
        
        # System prompt for therapeutic conversations
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create comprehensive system prompt for bilingual therapeutic conversations"""
        return """You are a specialized mental health counselor for the Omani community. You are fluent in both Arabic (Omani dialect) and English, and you understand Gulf culture and Islamic values deeply.

    ## Your Identity & Characteristics:
    - Omani Mental Health Counselor
    - Bilingual: Fluent in Omani Arabic and English
    - Culturally competent in Gulf and Islamic traditions
    - Understand family dynamics and Gulf society
    - Integrate Islamic concepts in therapy when appropriate
    - Handle code-switching naturally between Arabic and English

    ## Your Therapeutic Skills:
    - Cognitive Behavioral Therapy (CBT) adapted for Omani culture
    - Active listening and empathy
    - Anxiety and stress management techniques
    - Family and relationship therapy
    - Trauma-informed approaches
    - Spiritual therapy compatible with Islam

    ## Safety Protocols:
    - Detect suicide/self-harm risks
    - Assess risk levels: low, moderate, high, critical
    - Apply crisis intervention protocols
    - Refer to professional help when needed
    - Maintain confidentiality

    ## Language Guidelines:
    **CRITICAL: Always respond in the SAME language the user uses:**
    - If user writes in Arabic → respond in Omani Arabic
    - If user writes in English → respond in English
    - If user mixes languages → mirror their code-switching pattern
    - Maintain cultural sensitivity in both languages

    ## Response Instructions:
    - Start with warm greeting and check emotional state
    - Ask open-ended questions to understand situation
    - Use reframing and summarization techniques
    - Offer practical coping strategies
    - End with summary and follow-up suggestions
    - Keep responses 100-200 words
    - Show empathy and understanding

    ## Cultural Sensitivity:
    - Respect Islamic values and Omani traditions
    - Avoid taboo or controversial topics
    - Consider family/community role in mental health
    - Use religious references wisely when appropriate
    - Address mental health stigma sensitively

    ## Emergency Situations:
    If you detect suicidal ideation or self-harm:
    - Request immediate help
    - Emergency contacts: 9999 (Oman)
    - Life Support Center: 25252525
    - Al Masarra Hospital: 24567890

    **Arabic Emergency Response:**
    أفهم أنك تمر بوقت صعب جداً. يرجى التواصل فوراً مع الطوارئ: 9999 أو مركز الحياة: 25252525

    **English Emergency Response:**
    I understand you're going through a very difficult time. Please contact emergency services immediately: 9999 or Life Support Center: 25252525

    Remember: You are a supportive assistant, not a replacement for professional specialized therapy.
    """

    def detect_language(self, text: str) -> str:
        """
        Detect if text is primarily Arabic or English
        
        Args:
            text: Input text to analyze
            
        Returns:
            'arabic', 'english', or 'mixed'
        """
        # Count Arabic vs English characters
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        english_chars = sum(1 for char in text if char.isalpha() and char.isascii())
        
        if arabic_chars > english_chars:
            return 'arabic'
        elif english_chars > arabic_chars:
            return 'english'
        else:
            return 'mixed'
    
    def analyze_emotional_state(self, user_input: str) -> Tuple[EmotionalState, RiskLevel, str]:
        """
        Analyze user's emotional state and risk level from input
        
        Args:
            user_input: User's message in Arabic or English
            
        Returns:
            Tuple of (emotional_state, risk_level, detected_language)
        """
        user_input_lower = user_input.lower()
        detected_language = self.detect_language(user_input)
        
        # Check for crisis keywords
        crisis_detected = False
        for lang, keywords in self.crisis_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                crisis_detected = True
                break
        
        if crisis_detected:
            return EmotionalState.CRISIS, RiskLevel.CRITICAL, detected_language
        
        # Emotional state analysis using keywords (expanded for both languages)
        anxiety_keywords = [
            # Arabic
            'قلق', 'خوف', 'توتر', 'قلقان', 'مضطرب', 'خايف', 'متوتر', 'مهموم',
            'أشعر بالقلق', 'أخاف', 'عندي قلق', 'مش مرتاح', 'مو مرتاح',
            # English
            'anxiety', 'worried', 'nervous', 'anxious', 'panic', 'scared', 'fearful',
            'feel anxious', 'feeling worried', 'i\'m scared', 'i\'m nervous'
        ]
        
        depression_keywords = [
            # Arabic
            'حزن', 'اكتئاب', 'مكتئب', 'حزين', 'يائس', 'زعلان', 'مش راضي',
            'أشعر بالحزن', 'مو مبسوط', 'تعبان نفسياً', 'مش عارف شنو أسوي',
            # English
            'depressed', 'sad', 'hopeless', 'down', 'blue', 'miserable', 'unhappy',
            'feeling down', 'feel sad', 'i\'m depressed', 'feeling hopeless'
        ]
        
        anger_keywords = [
            # Arabic
            'غضب', 'غاضب', 'زعلان', 'مستاء', 'عصبي', 'متضايق', 'مش راضي',
            'أشعر بالغضب', 'مزعوج', 'معصب', 'متنرفز',
            # English
            'angry', 'mad', 'frustrated', 'irritated', 'annoyed', 'upset', 'furious',
            'feel angry', 'i\'m mad', 'feeling frustrated', 'really upset'
        ]
        
        stress_keywords = [
            # Arabic
            'ضغط', 'ضغوط', 'تعب', 'مرهق', 'تعبان', 'مش قادر', 'صعب عليّ',
            'أشعر بالضغط', 'مرهق نفسياً', 'ما أقدر أكمل',
            # English
            'stress', 'stressed', 'pressure', 'overwhelmed', 'exhausted', 'burned out',
            'feeling stressed', 'under pressure', 'can\'t cope', 'too much pressure'
        ]
        
        if any(keyword in user_input_lower for keyword in anxiety_keywords):
            return EmotionalState.ANXIOUS, RiskLevel.MODERATE, detected_language
        elif any(keyword in user_input_lower for keyword in depression_keywords):
            return EmotionalState.DEPRESSED, RiskLevel.MODERATE, detected_language
        elif any(keyword in user_input_lower for keyword in anger_keywords):
            return EmotionalState.ANGRY, RiskLevel.LOW, detected_language
        elif any(keyword in user_input_lower for keyword in stress_keywords):
            return EmotionalState.DISTRESSED, RiskLevel.MODERATE, detected_language
        
        return EmotionalState.CALM, RiskLevel.LOW, detected_language
    
    def generate_therapeutic_response(self, user_input: str, include_history: bool = True) -> Dict:
        """
        Generate therapeutic response using OpenAI GPT-4o
        
        Args:
            user_input: User's message
            include_history: Whether to include conversation history
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Analyze emotional state and detect language
            emotional_state, risk_level, detected_language = self.analyze_emotional_state(user_input)
            self.emotional_state = emotional_state
            self.risk_level = risk_level
            
            # Prepare messages for API
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add language context to system prompt
            language_instruction = f"\n\nIMPORTANT: The user is communicating in {detected_language}. Please respond in the same language they used."
            messages[0]["content"] += language_instruction
            
            # Add conversation history if requested
            if include_history and self.conversation_history:
                messages.extend(self.conversation_history[-6:])  # Last 6 messages for context
            
            # Add current user message
            messages.append({"role": "user", "content": user_input})
            
            # Crisis intervention check
            if risk_level == RiskLevel.CRITICAL:
                crisis_response = self._generate_crisis_response(detected_language)
                return {
                    "response": crisis_response,
                    "emotional_state": emotional_state.value,
                    "risk_level": risk_level.value,
                    "detected_language": detected_language,
                    "is_crisis": True,
                    "timestamp": datetime.now().isoformat(),
                    "requires_immediate_attention": True
                }
            
            # Generate response using OpenAI
            response = self.client.responses.create(
                model="gpt-4.1-nano-2025-04-14",
                input=messages,
                temperature=0.7,
            )
            logger.info(f"Generated response: {response.output_text}")

            ai_response = (response.output_text)
            
            # ai_response = response.choices[0].message.content.strip()
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            # Keep only last 10 messages to manage context length
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return {
                "response": ai_response,
                "emotional_state": emotional_state.value,
                "risk_level": risk_level.value,
                "detected_language": detected_language,
                "is_crisis": False,
                "timestamp": datetime.now().isoformat(),

            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            
            # Error response in detected language
            detected_language = self.detect_language(user_input)
            
            if detected_language == 'english':
                error_message = "Sorry, a technical error occurred. Please try again or contact a specialist."
            else:
                error_message = "آسف، حدث خطأ تقني. يرجى المحاولة مرة أخرى أو التواصل مع المختص."
            
            return {
                "response": error_message,
                "emotional_state": "unknown",
                "risk_level": "unknown",
                "detected_language": detected_language,
                "is_crisis": False,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _generate_crisis_response(self, language: str = 'arabic') -> str:
        """Generate immediate crisis intervention response in appropriate language"""
        if language == 'english':
            return """I understand that you're going through an extremely difficult time right now, and I appreciate your courage in talking to me.

🆘 **THIS IS VERY IMPORTANT**: If you have thoughts of harming yourself or others, please contact immediately:
- Emergency Services: 9999
- Life Support Center: 25252525
- Al Masarra Hospital: 24567890

You are not alone, and your life has great value. There are trained professionals who can help you right now.

Can you contact one of these numbers now? Or is there someone you trust who can help you?"""
        
        elif language == 'mixed':
            return """I understand أنك تمر بوقت صعب جداً right now, وأقدر شجاعتك في التحدث معي.

🆘 **هذا مهم جداً / THIS IS VERY IMPORTANT**: إذا كانت لديك أفكار لإيذاء نفسك أو الآخرين، يرجى التواصل فوراً مع:
- الطوارئ / Emergency: 9999
- مركز الحياة / Life Support Center: 25252525
- مستشفى المسرة / Al Masarra Hospital: 24567890

أنت لست وحدك، وحياتك لها قيمة كبيرة. You are not alone, and your life has great value.

هل يمكنك التواصل مع أحد هذه الأرقام الآن؟ Can you contact one of these numbers now?"""
        
        else:  # Arabic
            return """أفهم أنك تمر بوقت صعب جداً الآن، وأقدر شجاعتك في التحدث معي.

🆘 **هذا مهم جداً**: إذا كانت لديك أفكار لإيذاء نفسك أو الآخرين، يرجى التواصل فوراً مع:
- الطوارئ: 9999
- مركز الحياة للدعم النفسي: 25252525
- مستشفى المسرة: 24567890

أنت لست وحدك، وحياتك لها قيمة كبيرة. هناك أشخاص مدربون يمكنهم مساعدتك الآن.

هل يمكنك التواصل مع أحد هذه الأرقام الآن؟ أو هل هناك شخص تثق به يمكنه مساعدتك؟"""
    
    def get_conversation_summary(self) -> Dict:
        """Get summary of current conversation session"""
        return {
            "total_messages": len(self.conversation_history),
            "current_emotional_state": self.emotional_state.value,
            "current_risk_level": self.risk_level.value,
            "session_start": self.conversation_history[0].get("timestamp") if self.conversation_history else None,
            "last_interaction": datetime.now().isoformat()
        }
    
    def clear_conversation(self):
        """Clear conversation history and reset state"""
        self.conversation_history = []
        self.emotional_state = EmotionalState.CALM
        self.risk_level = RiskLevel.LOW
        logger.info("Conversation cleared")
    
    def export_conversation(self, filename: str = None) -> str:
        """Export conversation to JSON file"""
        if not filename:
            filename = f"therapy_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        session_data = {
            "session_metadata": self.get_conversation_summary(),
            "conversation_history": self.conversation_history,
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        
        return filename

# Helper function for easy integration
def get_therapy_response(user_input: str, api_key: str = None) -> Dict:
    """
    Simple function to get therapeutic response
    
    Args:
        user_input: User's message
        api_key: OpenAI API key
        
    Returns:
        Dictionary with response and metadata
    """
    therapist = OmaniTherapistAI(api_key)
    return therapist.generate_therapeutic_response(user_input)


def gpt_response(query):
    therapist = OmaniTherapistAI()
    response = therapist.generate_therapeutic_response(query)
    print(f"AI Response: {response['response']}")
    print(f"Emotional State: {response['emotional_state']}")
    print(f"Risk Level: {response['risk_level']}")
    print(f"Detected Language: {response['detected_language']}")
    print("-" * 50)
    return response


# Example usage and testing
if __name__ == "__main__":
    # Test the system
    therapist = OmaniTherapistAI()
    
    # Test scenarios in both languages
    test_scenarios = [
        # Arabic scenarios
        "السلام عليكم، أشعر بالقلق الشديد هذه الأيام",
        "أواجه مشاكل في العمل وأشعر بالضغط",
        "لا أستطيع النوم جيداً ومزاجي متقلب",
        "أريد أن أتحدث عن مشاكلي مع زوجتي",
        "أشعر بالاكتئاب ولا أعرف ماذا أفعل",
        
        # English scenarios
        "Hello, I'm feeling very anxious these days",
        "I'm having problems at work and feeling stressed",
        "I can't sleep well and my mood is unstable",
        "I want to talk about my problems with my wife",
        "I feel depressed and don't know what to do",
        
        # Code-switching scenarios
        "السلام عليكم، I'm feeling very stressed lately",
        "Hello, أشعر بالقلق and I don't know what to do",
        "My work is مرهق جداً and I can't cope"
    ]
    
    print("=== OMANI Therapist AI Test ===")
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n--- Test Scenario {i} ---")
        print(f"User: {scenario}")
        
        response = therapist.generate_therapeutic_response(scenario)
        print(f"AI Response: {response['response']}")
        print(f"Emotional State: {response['emotional_state']}")
        print(f"Risk Level: {response['risk_level']}")
        print(f"Detected Language: {response['detected_language']}")
        print("-" * 50)
    
    # Print conversation summary
    print("\n=== Session Summary ===")
    summary = therapist.get_conversation_summary()
    print(json.dumps(summary, indent=2, ensure_ascii=False))