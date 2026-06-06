# core/casting/mood_pack.py
"""The canonical 8-read Mood Pack. Every recorded voice is captured on these
exact reads, same booth/mic/distance/session, ~20-35s each."""

MOODS = ["neutral", "fired", "serious", "amused", "thoughtful", "reactions", "wry", "intimate"]

MOOD_PACK = {
    "neutral": {
        "label": "Neutral (warm baseline / anchor)",
        "direction": "Read it warm and conversational, like talking to one person.",
        "script": (
            "Right now, somewhere out there, somebody just turned this on for the first time. "
            "Maybe you're driving, maybe you're up late, maybe you're just getting started. "
            "Wherever you are, you found us, and I'm glad you did. Settle in, get comfortable, "
            "and let's spend some time together. This is your station, and this is exactly where "
            "you're supposed to be."
        ),
    },
    "fired": {
        "label": "Fired up (high energy)",
        "direction": "Big energy, fast, excited. Like the best part of the day is RIGHT NOW.",
        "script": (
            "Okay, stop what you're doing, because you are not ready for this one. I've been waiting "
            "all day to get on this mic, and we are going in. Turn it up, roll the windows down, tell "
            "somebody to come listen. This is the part of the day you've been waiting for, and trust "
            "me, it is about to be electric."
        ),
    },
    "serious": {
        "label": "Serious (grounded, sincere)",
        "direction": "Slow down. Sincere, grounded, no performance. Mean it.",
        "script": (
            "Let me slow it down for a second, because this matters. We don't always say the real "
            "thing out loud, but I'm going to. Times are heavy for a lot of people right now, and if "
            "that's you, I want you to know you're not alone in it. Take a breath. We're going to get "
            "through the hard stuff together. I mean that."
        ),
    },
    "amused": {
        "label": "Amused (light, lands on a laugh)",
        "direction": "Light and playful, smiling through it, end on a genuine laugh.",
        "script": (
            "So I have to tell you what just happened, because I still can't believe it. You know that "
            "feeling when something's so ridiculous you can't even be mad, you just have to laugh? Yeah. "
            "That was my whole morning. Ha. You can't make this stuff up. I wish I could."
        ),
    },
    "thoughtful": {
        "label": "Thoughtful (reflective, slower, lower)",
        "direction": "Reflective and unhurried, slightly lower, like thinking out loud.",
        "script": (
            "You ever stop and really think about how you got here? Not the big stuff, just the little "
            "turns. A conversation you almost didn't have. A day you almost stayed home. Funny how the "
            "smallest moments end up shaping everything. I've been sitting with that lately, and maybe "
            "you should too."
        ),
    },
    "reactions": {
        "label": "Reactions (short ad-libs + the laugh)",
        "direction": "Read each as a separate beat with a pause between. Natural, varied.",
        "script": (
            "Mm. ... Right. ... Come on, now. ... No way. ... Ha, okay, okay. ... Whew. ... I hear you. "
            "... That's wild. ... Let's get into it."
        ),
    },
    "wry": {
        "label": "Wry / Sarcastic (dry, deadpan)",
        "direction": "Dry, deadpan, eyebrow up. Understated, not mean.",
        "script": (
            "Oh, no, this is great. This is exactly how I pictured my day going. You ever notice how the "
            "people with the most to say are usually the ones who've done the least? No? Just me? Look, "
            "I'm not saying I told you so. But I did tell you so. Anyway. Let's all act surprised."
        ),
    },
    "intimate": {
        "label": "Intimate / Late-night (hushed, close-mic)",
        "direction": "Hushed, close to the mic, slow and warm. Just you and one listener.",
        "script": (
            "Hey. It's just us now. The rest of the world's gone quiet, and that's kind of the best part, "
            "isn't it? No rush tonight. Pull the covers up, turn the lights down low, and let me keep you "
            "company for a while. Whatever today took out of you, set it down right here. I'm not going "
            "anywhere. Just stay with me."
        ),
    },
}
