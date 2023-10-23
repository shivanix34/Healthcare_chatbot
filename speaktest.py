import pyttsx3

def Say(Text):
    engine = pyttsx3.init("sapi5")
    
    # Handle voice selection
    voices = engine.getProperty('voices')
    if voices:
        engine.setProperty('voice', voices[0].id)
    
    # Set speech rate
    engine.setProperty('rate', 170)
    
    try:
        print("  ")
        print(f"MediBot: {Text}")
        engine.say(text=Text)
        engine.runAndWait()
        print("  ")
    except Exception as e:
        print(f"Error in speech synthesis: {str(e)}")


