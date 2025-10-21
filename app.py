"""
app.py - League of Legends Counter-Pick AI Agent

This program uses OpenAI's API to recommend champion counter-picks
based on enemy team composition.
"""

from pydantic import BaseModel, Field
from typing import List
from helpers import structured_generator

# ============================================================================
# DATA MODELS - Define the structure of AI responses
# ============================================================================

class ChampionRecommendations(BaseModel):
    """
    This model defines what data we expect back from the AI.
    Pydantic validates that the AI returns data in this exact format.
    """
    champions: List[str] = Field(
        description="List of 3-5 recommended champion names",
        min_items=3,
        max_items=5
    )
    reasoning: str = Field(
        description="Detailed explanation of why these champions counter the enemy team"
    )
    key_threats: List[str] = Field(
        description="Main threats from enemy team you need to watch out for"
    )

# ============================================================================
# USER INPUT - Collect enemy team composition
# ============================================================================

def get_user_input():
    """
    Collects champion picks from the user.
    Returns a dictionary with all the inputs.
    """
    print("=" * 60)
    print("    LEAGUE OF LEGENDS COUNTER-PICK AI AGENT")
    print("=" * 60)
    print("\nEnter enemy team champions (or press Enter to skip):\n")
    
    # Collect enemy team info
    enemy_top = input("Enemy Top: ").strip() or "Unknown"
    enemy_jgl = input("Enemy Jungle: ").strip() or "Unknown"
    enemy_mid = input("Enemy Mid: ").strip() or "Unknown"
    enemy_adc = input("Enemy ADC: ").strip() or "Unknown"
    enemy_support = input("Enemy Support: ").strip() or "Unknown"
    
    # Collect your role
    print("\nYour role:")
    your_role = input("Role (Top/Jungle/Mid/ADC/Support): ").strip().capitalize()
    
    return {
        "top": enemy_top,
        "jungle": enemy_jgl,
        "mid": enemy_mid,
        "adc": enemy_adc,
        "support": enemy_support,
        "your_role": your_role
    }

# ============================================================================
# PROMPT ENGINEERING - Create detailed instructions for the AI
# ============================================================================

def create_prompt(enemy_team: dict, your_role: str) -> str:
    """
    Creates a detailed prompt that gives the AI context about League of Legends
    and what makes a good counter-pick.
    
    The more specific and detailed your prompt, the better the AI's recommendations.
    """
    
    prompt = f"""You are an expert League of Legends analyst specializing in champion select strategy.

ENEMY TEAM COMPOSITION:
- Top: {enemy_team['top']}
- Jungle: {enemy_team['jungle']}
- Mid: {enemy_team['mid']}
- ADC: {enemy_team['adc']}
- Support: {enemy_team['support']}

YOUR ROLE: {your_role}

TASK:
Recommend 3-5 champions for the {your_role} role that counter this enemy team composition.

ANALYSIS CRITERIA:
1. **Lane Matchup**: How well does the champion perform in direct lane matchups?
2. **Damage Type**: Consider AP/AD balance (avoid same damage type as team)
3. **Team Composition**: Does the enemy have tanks, assassins, poke, engage, disengage?
4. **Win Conditions**: Identify what the enemy wants to do and how to stop it
5. **Scaling**: Consider early/mid/late game power spikes
6. **Crowd Control**: Does your team need more CC or do you need to avoid CC?
7. **Mobility**: Can you match enemy mobility or do you need to punish immobile enemies?

Provide champions that synergize well against THIS specific enemy composition, not just general strong picks."""

    return prompt

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function that orchestrates the entire program:
    1. Get user input
    2. Create the prompt
    3. Call the AI
    4. Display results
    """
    
    # Step 1: Collect information from user
    user_data = get_user_input()
    
    # Step 2: Build the AI prompt with detailed context
    prompt = create_prompt(
        enemy_team={
            "top": user_data["top"],
            "jungle": user_data["jungle"],
            "mid": user_data["mid"],
            "adc": user_data["adc"],
            "support": user_data["support"]
        },
        your_role=user_data["your_role"]
    )
    
    # Step 3: Configure the AI model
    # Options: "gpt-4-turbo" (best), "gpt-4" (good), "gpt-3.5-turbo" (cheaper)
    openai_model = "gpt-5"
    
    # Step 4: Call the AI and get structured recommendations
    print("\n🤖 Analyzing enemy team composition...\n")
    
    try:
        # This calls your helper function that connects to OpenAI
        result = structured_generator(openai_model, prompt, ChampionRecommendations)
        
        # Step 5: Display the results in a nice format
        print("=" * 60)
        print(f"  RECOMMENDED CHAMPIONS FOR {user_data['your_role'].upper()}")
        print("=" * 60)
        
        for i, champion in enumerate(result.champions, 1):
            print(f"{i}. {champion}")
        
        print("\n" + "=" * 60)
        print("  KEY THREATS TO WATCH")
        print("=" * 60)
        for threat in result.key_threats:
            print(f"⚠️  {threat}")
        
        print("\n" + "=" * 60)
        print("  REASONING")
        print("=" * 60)
        print(result.reasoning)
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. Your .env file exists with OPENAI_API_KEY set")
        print("2. You have installed required packages: pip install openai pydantic python-dotenv")
        print("3. You have API credits available in your OpenAI account")

# ============================================================================
# RUN THE PROGRAM
# ============================================================================

if __name__ == "__main__":
    main()
