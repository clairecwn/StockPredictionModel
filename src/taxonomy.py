"""Maps raw events (Capital IQ event types, headlines, 8-K items) to broad 'noise' classes.
Used by both the historical event builder and the live tracker so both speak the same language."""
import re

# Calendar / housekeeping items that are not news shocks: dropped
ROUTINE = re.compile(r"conference|presentation|earnings call|analyst/investor day|shareholder/analyst call|"
                     r"release date|ex-div|dividend affirm|annual general meeting|shareholders meeting|board meeting|"
                     r"fiscal year end|name change|ticker change|address change|exchange change|"
                     r"buyback tranche|buyback transaction|seeking acquisitions|seeking to sell|index constituent add", re.I)

# Order matters: first match wins, most serious first
TYPE_RULES = [
    ("accounting_fraud", r"restatement|auditor going concern|auditor change|delayed sec filing|impairment|write.?off"),
    ("credit_distress", r"bankruptcy|default|credit rating.*(downgrade|watch)|delisting|index constituent drop"),
    ("legal_regulatory", r"lawsuit|legal|regulatory|enforcement|fda|investigation|sanction|settlement"),
    ("activist_short", r"activist|proxy|shareholder proposal"),
    ("leadership", r"executive|board change|ceo|cfo|director"),
    ("operational", r"recall|halt|resume of operations|discontinued operations|downsiz|labor|strike|cyber|unusual event|disaster"),
    ("earnings_guidance", r"earnings|guidance|operating results|sales results|production results"),
    ("ma_strategy", r"m&a|merger|acquisition|divestiture|spin.?off|strategic alternative|alliance|joint venture|"
                    r"business expansion|product|client announcement|business reorganization"),
    ("capital", r"buyback|offering|private placement|debt financing|dividend (increase|decrease)|special dividend|stock split"),
]
TYPE_RULES = [(c, re.compile(p, re.I)) for c, p in TYPE_RULES]

# Headline overrides: political / policy news is not a Capital IQ event type, so detect it from text
POLITICS = re.compile(r"tariff|white house|president|congress|senat|election|government|beijing|china|taiwan|"
                      r"export (control|ban|curb)|sanction|subsid|chips act|executive order|geopolit", re.I)
SCANDAL = re.compile(r"affair|misconduct|harass|spy|espionage|fraud|bribe|probe|scandal|whistleblow", re.I)

# Headline-only rules for live news (no event type available)
HEADLINE_RULES = [
    ("accounting_fraud", r"restat|fraud|accounting (irregular|probe)|sec charges|misstat|whistleblow"),
    ("legal_regulatory", r"lawsuit|sued|sues|antitrust|regulator|probe|investigat|subpoena|indict|settle|\bfine[sd]?\b|ftc|doj"),
    ("activist_short", r"short.?seller|hindenburg|muddy waters|activist|proxy fight"),
    ("leadership", r"\bceo\b|chief executive|founder|chairman|\bcfo\b|resign|steps? down|ousted|succession|affair|misconduct"),
    ("politics_policy", POLITICS.pattern),
    ("operational", r"recall|outage|breach|hack|cyberattack|explosion|grounded|strike|shortage|plant (fire|shut)"),
    ("credit_distress", r"bankrupt|default|downgrade[sd]? .*(credit|rating)|junk"),
    ("analyst", r"downgrade|upgrade|price target|initiat\w* coverage|overweight|underweight"),
    ("earnings_guidance", r"earnings|quarterly results|guidance|revenue|\beps\b|outlook|forecast"),
    ("ma_strategy", r"acquisition|acquire|merger|deal|partnership|contract|launch|unveil"),
]
HEADLINE_RULES = [(c, re.compile(p, re.I)) for c, p in HEADLINE_RULES]

CLASSES = ["accounting_fraud", "credit_distress", "legal_regulatory", "activist_short", "leadership",
           "politics_policy", "operational", "analyst", "earnings_guidance", "ma_strategy", "capital", "other"]
PRIORITY = {c: i for i, c in enumerate(CLASSES)}

def classify_keydev(eventtype, headline=""):
    et, hl = str(eventtype or ""), str(headline or "")
    if ROUTINE.search(et):
        return None
    if POLITICS.search(hl) and not re.search(r"earnings|results", et, re.I):
        return "politics_policy"
    for c, rx in TYPE_RULES:
        if rx.search(et):
            return c
    return "other"

def classify_headline(headline):
    for c, rx in HEADLINE_RULES:
        if rx.search(str(headline or "")):
            return c
    return "other"

def is_scandal(headline):
    return int(bool(SCANDAL.search(str(headline or ""))))

if __name__ == "__main__":
    tests = [("Executive/Board Changes - Other", "Nvidia board opens probe into CEO over foreign intelligence effort"),
             ("Lawsuits & Legal Issues", "Shareholder sues Nvidia over disclosures"),
             ("Product-Related Announcements", "US tightens export controls on AI chips to China"),
             ("Earnings Calls", "Q3 2026 earnings call")]
    for et, hl in tests:
        print(f"{classify_keydev(et, hl)!s:18} | {et} | {hl}")
    print(classify_headline("Tesla recalls 2 million vehicles"), classify_headline("Hindenburg shorts Adani"))
