"""~80 hand-labeled tasks against memories.py's large personal memory
bank. Real natural-language questions a personal assistant would
actually get asked -- recall here means "did the router surface the
memory a human actually wanted," not "did it find the nearest vector."

About a third of these are direct paraphrases; a large minority are
DELIBERATE DISTRACTOR-DISAMBIGUATION tasks -- pairs of memories that are
close in vector space (two birthdays days apart, two different
allergies, two different "reset X" procedures, two coffee orders) where
only reading the actual content tells you which one the question is
really about. Those are the tasks that make recall < 1.00 possible, and
therefore make this eval able to show a real quality gap between judges
instead of every strategy hitting a 1.00 ceiling.

Each task has exactly one expected memory id -- kept unambiguous on
purpose so ground truth stays defensible by inspection, not a judgment
call.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    query: str
    expected_id: str


TASKS: list[Task] = [
    # --- direct: semantic ---
    Task("Do I have any food allergies?", "sem-1"),
    Task("Am I vegetarian or do I eat meat?", "sem-2"),
    Task("What's my blood type?", "sem-3"),
    Task("What's my eyeglasses prescription?", "sem-4"),
    Task("Can I have dairy?", "sem-5"),
    Task("What race am I training for?", "sem-6"),
    Task("Do I prefer lifting weights or running?", "sem-7"),
    Task("How many hours of sleep am I aiming for?", "sem-8"),
    Task("What budgeting method do I use?", "sem-9"),
    Task("Do I pick individual stocks or index funds?", "sem-10"),
    Task("How many months of expenses should my emergency fund cover?", "sem-11"),
    Task("Which streaming subscriptions did I keep?", "sem-12"),
    Task("Which credit card do I use for travel?", "sem-13"),
    Task("What's my rule for how much rent I can afford?", "sem-14"),
    Task("How do my partner and I split shared expenses?", "sem-15"),
    Task("What percentage of my freelance income goes to taxes?", "sem-16"),
    Task("Am I planning to buy a home soon?", "sem-17"),
    Task("Would I rather have a shorter commute or cheaper rent?", "sem-18"),
    Task("Is my smart home set up with Google or Apple?", "sem-19"),
    Task("Why did I choose fiber over cable internet?", "sem-20"),
    Task("What's my preferred career level goal?", "sem-23"),
    Task("Do I like frontend or backend work more?", "sem-24"),
    Task("Did I negotiate remote work?", "sem-25"),
    Task("Do I prefer meetings or async messages?", "sem-26"),
    Task("Did I take the management track offer?", "sem-27"),
    Task("What language am I learning right now?", "sem-37"),
    Task("How many books am I trying to read this year?", "sem-40"),
    Task("Do I read fiction as an audiobook or a physical book?", "sem-41"),
    Task("What seat do I always book on flights?", "sem-43"),
    Task("When does my passport expire?", "sem-44"),
    Task("Do I have Global Entry?", "sem-45"),
    Task("At what trip cost do I always buy travel insurance?", "sem-47"),
    # --- distractor-disambiguation: semantic ---
    Task("When's my partner's birthday?", "sem-30"),
    Task("When's my best friend's birthday?", "sem-31"),
    Task("When's my parents' anniversary?", "sem-32"),
    Task("Does my nephew have any allergies?", "sem-33"),
    Task("Is my best friend vegetarian too, or something stricter?", "sem-34"),
    Task("What does my partner order at a coffee shop?", "sem-35"),
    Task("What does my best friend order at a coffee shop?", "sem-36"),
    Task("How do I take my own coffee?", "sem-50"),
    Task("Why don't I have a cat or dog?", "sem-22"),
    # --- direct: procedural ---
    Task("How do I renew my passport?", "proc-2"),
    Task("How do I recover my email if I lose 2FA access?", "proc-3"),
    Task("What do I do on the 1st of every month for subscriptions?", "proc-4"),
    Task("How do I handle my quarterly freelance taxes?", "proc-5"),
    Task("How often do I rotate my password manager's master password?", "proc-6"),
    Task("What's my morning routine before work?", "proc-8"),
    Task("What's my weekly workout split?", "proc-9"),
    Task("How do I meal prep?", "proc-10"),
    Task("How often do I go to the dentist?", "proc-12"),
    Task("Do I wear sunscreen every day?", "proc-13"),
    Task("How is my half marathon training structured?", "proc-14"),
    Task("How do I bleed the radiators at home?", "proc-16"),
    Task("How often do I change my HVAC filter?", "proc-17"),
    Task("Where's the main water shutoff valve?", "proc-19"),
    Task("When do I replace smoke detector batteries?", "proc-20"),
    Task("How do I dispute a credit card charge?", "proc-22"),
    Task("How do my partner and I settle up shared expenses in practice?", "proc-23"),
    Task("How does my automatic investing work?", "proc-24"),
    Task("How often should I change my car's oil?", "proc-26"),
    Task("How do I jump-start my car?", "proc-28"),
    Task("How do I practice Spanish?", "proc-30"),
    Task("What's my guitar practice schedule?", "proc-31"),
    Task("What's my pre-trip checklist?", "proc-34"),
    Task("How do I pack for a trip?", "proc-35"),
    Task("When should I renew TSA PreCheck?", "proc-36"),
    Task("What order do I update my address in after moving?", "proc-39"),
    Task("How do I back up my phone photos?", "proc-42"),
    # --- distractor-disambiguation: procedural ---
    Task("How do I reset my wifi router?", "proc-1"),
    Task("How do I reset my cable modem?", "proc-44"),
    Task("A guest can't get on wifi -- how do I find the password without resetting anything?", "proc-7"),
    Task("How do I take care of my fish tank week to week?", "proc-18"),
    Task("How do I take care of my fish tank while I'm traveling?", "proc-37"),
    Task("How do I make coffee in the morning?", "proc-43"),
    # --- direct: episodic ---
    Task("Did I ever get food poisoning while traveling?", "epi-1"),
    Task("Have I ever sprained anything while hiking?", "epi-2"),
    Task("Have I had a bad reaction to an allergy medication before?", "epi-3"),
    Task("Did last year's flu shot cause any side effects?", "epi-4"),
    Task("Have I hurt myself bouldering?", "epi-5"),
    Task("What was my 10k race time?", "epi-6"),
    Task("Did the dentist find any cavities recently?", "epi-7"),
    Task("Did I ever get an overdraft fee refunded?", "epi-8"),
    Task("Did I get a raise this year?", "epi-9"),
    Task("Has my credit card ever been compromised?", "epi-11"),
    Task("Have I ever paid off my credit card in full?", "epi-14"),
    Task("Has a flight of mine ever been cancelled?", "epi-15"),
    Task("Have I ever lost my luggage?", "epi-16"),
    Task("Have I ever been upgraded on a flight?", "epi-17"),
    Task("Have I ever missed a connecting flight?", "epi-18"),
    Task("Did I have any passport issues at customs?", "epi-19"),
    Task("Did I use Global Entry on the Japan trip?", "epi-21"),
    Task("Did I ever miss a deadline because of another team?", "epi-22"),
    Task("What positive feedback did I get in my last review?", "epi-23"),
    Task("Why did I leave my previous job?", "epi-24"),
    Task("What was my conference talk about?", "epi-25"),
    Task("Did anything go wrong at home during my conference talk?", "epi-26"),
    Task("Has a pipe ever burst at home?", "epi-29"),
    Task("Has a package ever been stolen from my porch?", "epi-30"),
    Task("Has a neighbor ever complained about noise?", "epi-31"),
    Task("Has the fish tank had any problems?", "epi-33"),
    Task("Why did the smoke detector go off that one time?", "epi-34"),
    Task("Did I ever let the HVAC filter go too long?", "epi-35"),
    Task("Did I ever forget my partner's birthday?", "epi-39"),
    Task("Was there ever an allergy scare with my nephew?", "epi-40"),
    Task("Did I return a laptop before?", "epi-43"),
    Task("Did I negotiate the price when I bought my car?", "epi-44"),
    Task("Did I regret any impulse purchase?", "epi-45"),
    Task("Did travel insurance ever actually pay off?", "epi-48"),
    # --- paraphrases (same target as an earlier task, different wording) ---
    Task("What should I avoid eating because of an allergy?", "sem-1"),
    Task("Is dairy off the table for me?", "sem-5"),
    Task("What's my plan for retirement/investing?", "sem-10"),
    Task("What's the maximum I should spend on rent?", "sem-14"),
    Task("Am I remote, hybrid, or in-office?", "sem-25"),
    Task("What programming languages do I reach for?", "sem-29"),
    Task("What am I learning for the trip I'm planning?", "sem-37"),
    Task("Which of my two closest people has the birthday earlier in March?", "sem-30"),
    Task("What should I get my best friend for coffee if I'm buying?", "sem-36"),
    Task("What's the very first thing I should check if the internet is down?", "proc-1"),
    Task("What's my routine right when I wake up?", "proc-8"),
    Task("How do I train for a long run?", "proc-14"),
    Task("What's the first step if I smell gas or the power trips?", "proc-15"),
    Task("Give me the steps to undo a bad credit card charge.", "proc-22"),
    Task("What do I check before I leave for a trip?", "proc-34"),
    Task("Was there ever a problem with my luggage?", "epi-16"),
    Task("What happened the last time I flew business class?", "epi-17"),
    Task("Did I ever have a scare related to my nephew's allergy?", "epi-40"),
]
