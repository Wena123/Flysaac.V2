# ============================================================
# FLYISAAC V3.6.2 — DOPAMINE SETTINGS
# ============================================================
#
# TO JEST GŁÓWNY PLIK DO EDYCJI DOPAMINY.
# Nie musisz grzebać w dopamine.py ani play_isaac_v3.py.
#
# Najważniejsza sekcja to EVENT_DOPAMINE poniżej.
#
# Wartość dodatnia  -> pozytywny sygnał (PAM-like)
# Wartość ujemna    -> negatywny sygnał (PPL-like)
# 0.0               -> ignoruj zdarzenie dla dopaminy
# None              -> użyj oryginalnego reward z eventu
#
# UWAGA:
# Te wartości sterują TYLKO systemem dopaminy V3.6.x.
# Nie zmieniają rewardów starego systemu Isaaca / restartu / logiki gry.
# ============================================================


# ============================================================
# 1. DOPAMINA ZA KONKRETNE ZDARZENIA
# ============================================================

USE_CUSTOM_EVENT_VALUES = True

# Jeżeli eventu NIE ma w EVENT_DOPAMINE:
# True  = użyj jego oryginalnego event["reward"]
# False = ignoruj go dla dopaminy
USE_ORIGINAL_REWARD_FOR_UNLISTED_EVENTS = True

# Wypisuj do konsoli np.:
# DOPAMINE EVENT | PLAYER_HIT | raw=-5.00 -> DA=-3.00
PRINT_DOPAMINE_EVENTS = True

# Globalne skalowanie po wybraniu wartości eventu.
GLOBAL_EVENT_MULTIPLIER = 1.0
POSITIVE_EVENT_MULTIPLIER = 1.0
NEGATIVE_EVENT_MULTIPLIER = 1.0

# Edytuj po prostu liczby po prawej stronie.
# Jeśli jakiegoś eventu nie masz w swojej wersji bridge'a, nic się nie stanie.
EVENT_DOPAMINE = {
    # ---------------- GOOD / REWARD ----------------
    "NEW_ROOM":          +2.00,
    "ROOM_CLEAR":        +100.00,
    "SHOT_HIT":          +1.00,
    "ENEMY_KILL":        +5.00,
    "BOSS_KILL":         +30.00,
    "ITEM_PICKUP":       +3.00,
    "PICKUP":            +1.25,
    "WALL_ESCAPE":       +0.30,
    "RUN_START":          0.00,

    # ---------------- BAD / PUNISHMENT ----------------
    "PLAYER_HIT":        -3.00,
    "PLAYER_DEATH":     -10.00,
    "SHOT_MISS":         -0.30,
    "NO_ENEMY_SHOT":     -0.15,
    "WALL_STUCK":        -0.50,
    "ROOM_STALL":        -0.50,
    "PROJECTILE_TOWARD": -0.20,
    "BOMB_WASTED":       -0.30,

    # Przykład wyłączenia danego eventu:
    # "SOME_EVENT":       0.00,

    # Przykład użycia oryginalnego reward tylko dla tego eventu:
    # "SOME_OTHER_EVENT": None,
}


# ============================================================
# 2. JAK DŁUGO UTRZYMUJE SIĘ SYGNAŁ
# ============================================================

# Im bliżej 1.0, tym dłużej utrzymuje się dopamina po zdarzeniu.
DOPAMINE_DECAY = 0.965

# Przed tanh reward jest dzielony przez tę wartość.
# Większa liczba = łagodniejsze impulsy dla dużych rewardów.
DOPAMINE_REWARD_SCALE = 5.0

DOPAMINE_POSITIVE_GAIN = 1.00
DOPAMINE_NEGATIVE_GAIN = 1.00


# ============================================================
# 3. PAM / PPL -> MALECNS
# ============================================================

DOPAMINE_INJECTION_GAIN = 0.13
DOPAMINE_MAX_DRIVE = 0.14
DOPAMINE_MAX_CELLS_PER_VALENCE = 128


# ============================================================
# 4. ONLINE POLICY LEARNING / ELIGIBILITY TRACE
# ============================================================

# Jak długo agent pamięta, które ostatnie decyzje mogły odpowiadać za reward.
DOPAMINE_ELIGIBILITY_DECAY = 0.94

# Jak szybko reward zmienia małe move/shoot biasy.
# Nie ustawiaj od razu bardzo wysoko — 0.012 jest celowo małe.
DOPAMINE_POLICY_LR = 0.012

# Maksymalny zapisany bias dla jednej klasy akcji.
DOPAMINE_POLICY_BIAS_CLIP = 0.30

# Jak mocno zapisany bias wpływa na finalne probability.
DOPAMINE_POLICY_STRENGTH = 0.65

# Po otrzymaniu rewardu eligibility trace jest mnożony przez tę liczbę.
DOPAMINE_POST_REWARD_ELIGIBILITY = 0.45


# ============================================================
# 5. ZAPIS PAMIĘCI
# ============================================================

DOPAMINE_AUTOSAVE_SECONDS = 30.0
DOPAMINE_POLICY_FILE = "isaac_dopamine_policy_v36.npz"
