"""A large, hand-written PERSONAL memory bank -- facts/preferences,
routines, and life events a personal AI assistant might actually
accumulate over months of real use. Synthetic, but deliberately built
like real personal-assistant data: many topic clusters contain a
near-duplicate PAIR (two birthdays close together, two different
allergies in the family, two different "reset X" procedures, two
coffee orders) specifically so a judge that only looks at vector-space
similarity, not content, has a real chance to pick the wrong one.
tasks.py's ground truth depends on every memory here being
distinguishable enough that "the right answer" is unambiguous to a
human reader.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Memory:
    id: str
    shard: str  # "semantic" | "procedural" | "episodic"
    content: str


MEMORIES: list[Memory] = [
    # ============================================================
    # semantic: facts, preferences, and standing decisions
    # ============================================================
    # --- health & body ---
    Memory("sem-1", "semantic", "Has a severe peanut allergy -- carries an EpiPen at all times."),
    Memory("sem-2", "semantic", "Is pescatarian since 2023: no meat, but eats fish occasionally."),
    Memory("sem-3", "semantic", "Blood type is O negative."),
    Memory("sem-4", "semantic", "Is nearsighted, -3.5 diopters in both eyes, wears daily contact lenses."),
    Memory("sem-5", "semantic", "Is lactose intolerant but can handle aged hard cheeses like parmesan."),
    Memory("sem-6", "semantic", "Is training for a half marathon in November."),
    Memory("sem-7", "semantic", "Prefers strength training over cardio when short on time."),
    Memory("sem-8", "semantic", "Sleep target is 7.5 hours; uses a blue-light filter on all screens after 9pm."),
    # --- finance ---
    Memory("sem-9", "semantic", "Follows the 50/30/20 rule for budgeting: needs/wants/savings."),
    Memory("sem-10", "semantic", "Only invests in low-cost index funds -- no individual stock picking."),
    Memory("sem-11", "semantic", "Emergency fund target is 6 months of living expenses."),
    Memory("sem-12", "semantic", "Cancelled the Netflix subscription but kept Spotify and the gym membership."),
    Memory("sem-13", "semantic", "Uses the Chase Sapphire card for all travel purchases, for the points."),
    Memory("sem-14", "semantic", "Personal rule: rent should never exceed 30% of take-home pay."),
    Memory("sem-15", "semantic", "Splits shared household expenses with their partner 60/40 based on income."),
    Memory("sem-16", "semantic", "Freelances on weekends and sets aside 25% of that income for taxes."),
    # --- home ---
    Memory("sem-17", "semantic", "Prefers renting over buying for at least the next 5 years, for flexibility."),
    Memory("sem-18", "semantic", "Prioritizes a short commute over cheaper rent farther from downtown."),
    Memory("sem-19", "semantic", "Smart home ecosystem is Google Home, not Apple HomeKit."),
    Memory("sem-20", "semantic", "Chose fiber internet over cable specifically for the upload speed."),
    Memory("sem-21", "semantic", "Prefers a corner desk layout for the home office."),
    Memory("sem-22", "semantic", "Allergic to cat dander, so no furred pets -- keeps a freshwater fish tank instead."),
    # --- work ---
    Memory("sem-23", "semantic", "Career goal is to reach staff engineer within 3 years."),
    Memory("sem-24", "semantic", "Prefers backend work over frontend."),
    Memory("sem-25", "semantic", "Negotiated a fully remote arrangement with no relocation requirement."),
    Memory("sem-26", "semantic", "Prefers async communication over meetings whenever possible."),
    Memory("sem-27", "semantic", "Turned down a management-track offer to stay an individual contributor."),
    Memory("sem-28", "semantic", "Personal goal: give at least one conference talk per year."),
    Memory("sem-29", "semantic", "Prefers Python for scripting and Go for production services."),
    # --- relationships (deliberate near-duplicate pairs) ---
    Memory("sem-30", "semantic", "Partner's birthday is March 3rd."),
    Memory("sem-31", "semantic", "Best friend's birthday is March 9th."),
    Memory("sem-32", "semantic", "Parents' wedding anniversary is June 14th."),
    Memory("sem-33", "semantic", "Nephew (sister's kid) has a tree-nut allergy."),
    Memory("sem-34", "semantic", "Best friend is fully vegan, unlike their own pescatarian diet."),
    Memory("sem-35", "semantic", "Partner's coffee order: oat milk latte, one sugar."),
    Memory("sem-36", "semantic", "Best friend's coffee order: black coffee, no sugar."),
    # --- learning & hobbies ---
    Memory("sem-37", "semantic", "Decided to learn Spanish, not French, ahead of an upcoming trip."),
    Memory("sem-38", "semantic", "Chose a self-paced app over a bootcamp for learning Spanish."),
    Memory("sem-39", "semantic", "Practices guitar three times a week."),
    Memory("sem-40", "semantic", "Reading goal for the year is 20 books."),
    Memory("sem-41", "semantic", "Prefers audiobooks for fiction, physical books for technical material."),
    Memory("sem-42", "semantic", "Took up bouldering as a hobby in 2025."),
    # --- travel ---
    Memory("sem-43", "semantic", "Always books a window seat, never an aisle."),
    Memory("sem-44", "semantic", "Passport expires August 2027."),
    Memory("sem-45", "semantic", "Has TSA PreCheck but decided not to renew Global Entry."),
    Memory("sem-46", "semantic", "Prefers a direct flight even when a layover option is cheaper."),
    Memory("sem-47", "semantic", "Personal rule: always buy travel insurance for any trip over $1,000."),
    Memory("sem-48", "semantic", "Prefers a carry-on-only packing style, even for weeklong trips."),
    Memory("sem-49", "semantic", "Grocery preference: buys organic produce but not organic packaged goods."),
    Memory("sem-50", "semantic", "Morning coffee preference: pour-over, no sugar, no milk."),
    # ============================================================
    # procedural: routines and how-tos
    # ============================================================
    # --- accounts & security ---
    Memory(
        "proc-1",
        "procedural",
        "To reset the home wifi router: hold the reset button 10 seconds, "
        "reconnect via the ISP's app, and re-enter the PPPoE credentials.",
    ),
    Memory(
        "proc-2",
        "procedural",
        "To renew a passport: fill out form DS-82, mail it with the old passport "
        "and a new photo; turnaround is about 6-8 weeks.",
    ),
    Memory(
        "proc-3",
        "procedural",
        "For email 2FA recovery, use the backup codes stored in the password "
        "manager's secure note, not the phone number recovery option.",
    ),
    Memory(
        "proc-4",
        "procedural",
        "Monthly routine: review every subscription on the 1st and cancel anything unused.",
    ),
    Memory(
        "proc-5",
        "procedural",
        "Quarterly freelance tax routine: export invoices, run them through the "
        "self-employed tax software, and pay estimated tax by the 15th.",
    ),
    Memory(
        "proc-6",
        "procedural",
        "Password manager master password gets rotated every 6 months, updated "
        "on every device the same day.",
    ),
    Memory(
        "proc-7",
        "procedural",
        "To reset a shared guest wifi password, look it up in the router app "
        "under connected devices rather than doing a full router reset.",
    ),
    # --- health routine ---
    Memory("proc-8", "procedural", "Morning routine: allergy medication with breakfast, then a 20-minute walk."),
    Memory("proc-9", "procedural", "Workout split is push/pull/legs, 3x a week, rest days Wednesday and Sunday."),
    Memory("proc-10", "procedural", "Meal prep routine: cook proteins on Sunday, portion into 5 containers."),
    Memory(
        "proc-11",
        "procedural",
        "Contact lens routine: replace the monthly lenses on the 1st, always "
        "carry backup glasses in the bag.",
    ),
    Memory("proc-12", "procedural", "Dentist checkup routine: every 6 months, booked right after the previous visit."),
    Memory("proc-13", "procedural", "Skincare routine: SPF every morning, regardless of weather."),
    Memory(
        "proc-14",
        "procedural",
        "Half-marathon training plan: long run every Saturday, distance "
        "increasing by 1 mile every 2 weeks.",
    ),
    # --- home maintenance ---
    Memory(
        "proc-15",
        "procedural",
        "To reset the circuit breaker: the main panel is in the hallway closet, "
        "flip the tripped switch fully off then back on.",
    ),
    Memory(
        "proc-16",
        "procedural",
        "To bleed the radiators: bleed key is in the kitchen drawer, turn "
        "counterclockwise until the hiss stops, then until water appears.",
    ),
    Memory("proc-17", "procedural", "HVAC filter gets replaced every 3 months, size 16x25x1."),
    Memory(
        "proc-18",
        "procedural",
        "Fish tank maintenance: 20% water change every week, rinse the filter "
        "media once a month.",
    ),
    Memory("proc-19", "procedural", "To shut off the main water valve, it's under the kitchen sink, turn clockwise."),
    Memory(
        "proc-20",
        "procedural",
        "Smoke detector batteries get replaced every 6 months, on the daylight-saving-time changes.",
    ),
    # --- financial routine ---
    Memory(
        "proc-21",
        "procedural",
        "Monthly budget review happens the first Sunday of the month, using the 50/30/20 spreadsheet.",
    ),
    Memory(
        "proc-22",
        "procedural",
        "To dispute a credit card charge, call the number on the back of the card within 60 days.",
    ),
    Memory(
        "proc-23",
        "procedural",
        "Shared expenses with the partner go through the Splitwise app, settled up monthly.",
    ),
    Memory(
        "proc-24",
        "procedural",
        "Investment routine: automatic transfer into the index fund on payday, no manual trading.",
    ),
    Memory(
        "proc-25",
        "procedural",
        "Quarterly estimated tax payment goes through IRS Direct Pay, before the 15th.",
    ),
    # --- car maintenance ---
    Memory("proc-26", "procedural", "Oil change interval is every 5,000 miles or 6 months, whichever comes first."),
    Memory("proc-27", "procedural", "Tire rotation happens every other oil change."),
    Memory(
        "proc-28",
        "procedural",
        "To jump-start the car: cables are in the trunk, connect red-to-red, black-to-ground.",
    ),
    Memory("proc-29", "procedural", "Car registration renews online through the DMV site every April."),
    # --- learning routine ---
    Memory("proc-30", "procedural", "Spanish practice is 20 minutes on the app every morning before work."),
    Memory("proc-31", "procedural", "Guitar practice is 30 minutes, Mon/Wed/Fri: scales first, then a song."),
    Memory("proc-32", "procedural", "Reading routine: 30 minutes before bed, physical book only, no phone."),
    Memory("proc-33", "procedural", "Bouldering is a gym session every Tuesday evening with a climbing partner."),
    # --- travel prep ---
    Memory(
        "proc-34",
        "procedural",
        "Pre-trip checklist: check passport expiration, notify the bank of "
        "travel, download offline maps.",
    ),
    Memory(
        "proc-35",
        "procedural",
        "Packing method: rolling clothes, always a change of clothes in the carry-on.",
    ),
    Memory(
        "proc-36",
        "procedural",
        "TSA PreCheck renewal gets requested online 4 months before it expires.",
    ),
    Memory(
        "proc-37",
        "procedural",
        "Fish tank care while traveling: auto-feeder set for twice daily, "
        "neighbor checks in once a week.",
    ),
    Memory(
        "proc-38",
        "procedural",
        "Travel insurance gets compared on a broker site, always with medical evacuation coverage.",
    ),
    # --- misc life admin ---
    Memory(
        "proc-39",
        "procedural",
        "After moving, update the address in this order: DMV, bank, employer HR, then USPS mail forwarding.",
    ),
    Memory(
        "proc-40",
        "procedural",
        "A subscription found during the monthly review gets cancelled in-app "
        "first, by email only if there's no in-app option.",
    ),
    Memory("proc-41", "procedural", "Grocery delivery gets ordered every Sunday for the week's meal prep."),
    Memory(
        "proc-42",
        "procedural",
        "Phone photos back up automatically to the cloud, plus a manual local backup every quarter.",
    ),
    Memory("proc-43", "procedural", "Morning coffee routine: grind the beans fresh, pour-over, no sugar."),
    Memory(
        "proc-44",
        "procedural",
        "To reset the cable modem (separate from the wifi router), unplug it "
        "for 30 seconds before plugging back in.",
    ),
    # ============================================================
    # episodic: things that actually happened
    # ============================================================
    # --- health events ---
    Memory(
        "epi-1",
        "episodic",
        "Got food poisoning during the 2025 Mexico trip, spent a day in bed, "
        "has avoided street tacos since.",
    ),
    Memory("epi-2", "episodic", "Sprained an ankle on a Colorado hiking trip, wore a brace for 3 weeks."),
    Memory("epi-3", "episodic", "Had a bad reaction to a new allergy medication, switched back to the old one."),
    Memory("epi-4", "episodic", "Last year's flu shot caused a sore arm for two days, otherwise no issues."),
    Memory("epi-5", "episodic", "Twisted a wrist bouldering, took two weeks off the climbing gym."),
    Memory("epi-6", "episodic", "Ran a first 10k race in under 55 minutes."),
    Memory("epi-7", "episodic", "Dentist found a small cavity at the 6-month checkup and filled it the same day."),
    # --- financial events ---
    Memory("epi-8", "episodic", "Disputed an overdraft fee with the bank and got it reversed."),
    Memory("epi-9", "episodic", "Got an 8% raise after the annual performance review."),
    Memory("epi-10", "episodic", "Lost a friendly bet on a football game, paid up with dinner."),
    Memory(
        "epi-11",
        "episodic",
        "Credit card got skimmed at a gas station -- the bank caught the fraud "
        "and issued a new card.",
    ),
    Memory("epi-12", "episodic", "Sold an old laptop online for more than expected."),
    Memory("epi-13", "episodic", "Accidentally overdrew checking while transferring to savings, fixed same day."),
    Memory("epi-14", "episodic", "Paid off the last credit card balance in full for the first time in years."),
    # --- travel events ---
    Memory("epi-15", "episodic", "A flight to Chicago got cancelled; got rebooked on the next morning's flight."),
    Memory("epi-16", "episodic", "Lost luggage on a trip to Rome -- it showed up three days later."),
    Memory("epi-17", "episodic", "Got upgraded to business class on a long-haul flight using points."),
    Memory("epi-18", "episodic", "Missed a connecting flight in Denver because the first leg was delayed."),
    Memory(
        "epi-19",
        "episodic",
        "Passport got briefly held at customs over a visa mix-up, resolved within an hour.",
    ),
    Memory("epi-20", "episodic", "A rental car had a flat tire on the way to the airport, changed it curbside."),
    Memory(
        "epi-21",
        "episodic",
        "The 2025 Japan trip was the first (and only) time Global Entry lines were actually used.",
    ),
    # --- work events ---
    Memory("epi-22", "episodic", "Missed a project deadline because a dependency team delivered their part late."),
    Memory(
        "epi-23",
        "episodic",
        "Got specific positive feedback in the performance review about mentoring juniors.",
    ),
    Memory("epi-24", "episodic", "Quit the previous job after a new manager changed the team's remote policy."),
    Memory("epi-25", "episodic", "Gave a first conference talk this year, on distributed tracing."),
    Memory(
        "epi-26",
        "episodic",
        "A production incident during that conference talk's week got resolved "
        "without needing to leave the stage.",
    ),
    Memory("epi-27", "episodic", "Turned down the management-track offer in writing during Q2."),
    Memory(
        "epi-28",
        "episodic",
        "Onboarded a new teammate and wrote the ramp-up doc that others later reused.",
    ),
    # --- home events ---
    Memory("epi-29", "episodic", "A pipe burst under the kitchen sink -- the plumber fixed it the same day."),
    Memory("epi-30", "episodic", "A package got stolen from the porch, filed a claim and got a replacement."),
    Memory(
        "epi-31",
        "episodic",
        "A neighbor complained about noise from a Saturday gathering, resolved by inviting them next time.",
    ),
    Memory("epi-32", "episodic", "The wifi router failed the week after a storm and was replaced under warranty."),
    Memory("epi-33", "episodic", "The fish tank heater died overnight -- lost two fish before noticing."),
    Memory("epi-34", "episodic", "The smoke detector went off from burnt toast at 6am, no actual fire."),
    Memory(
        "epi-35",
        "episodic",
        "The HVAC filter was overdue by two months and the airflow noticeably "
        "dropped before it finally got replaced.",
    ),
    # --- relationship events ---
    Memory(
        "epi-36",
        "episodic",
        "Had an argument with the partner about the shared expense split, "
        "resolved by adjusting the ratio.",
    ),
    Memory(
        "epi-37",
        "episodic",
        "Surprised the best friend for their birthday with a dinner at their favorite restaurant.",
    ),
    Memory("epi-38", "episodic", "Reconnected with an old college friend at a conference after 5 years."),
    Memory(
        "epi-39",
        "episodic",
        "Forgot the partner's birthday plans initially, recovered by planning a weekend trip.",
    ),
    Memory(
        "epi-40",
        "episodic",
        "Helped the nephew through an allergic-reaction scare at a family dinner -- close, but the EpiPen wasn't needed.",
    ),
    Memory("epi-41", "episodic", "Went to the parents' anniversary dinner and gave a toast."),
    Memory("epi-42", "episodic", "Had a falling-out with a coworker-turned-friend over a cancelled trip, later reconciled."),
    # --- purchase events ---
    Memory("epi-43", "episodic", "Returned a defective laptop within the return window, got a full refund."),
    Memory("epi-44", "episodic", "Bought a car after months of comparison shopping, negotiated $1,500 off sticker."),
    Memory("epi-45", "episodic", "Regretted an impulse-bought treadmill that's now mostly used as a clothes rack."),
    Memory("epi-46", "episodic", "Got the Chase Sapphire card specifically for one trip's flight points."),
    Memory("epi-47", "episodic", "Upgraded the home office to the corner-desk layout."),
    Memory(
        "epi-48",
        "episodic",
        "Bought travel insurance for the Japan trip and ended up actually using it for a lost-luggage claim.",
    ),
    Memory("epi-49", "episodic", "Bought a bulk pack of contact lenses during a sale."),
]
