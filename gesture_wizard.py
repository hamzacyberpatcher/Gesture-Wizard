import cv2
import numpy as np
import time
import threading
import random
from ultralytics import YOLO

# -----------------------------
# GLOBAL SHARED DATA
# -----------------------------
latest_gesture = None
frame_lock = threading.Lock()
running = True

# -----------------------------
# YOLO THREAD
# -----------------------------
def yolo_thread():
    global latest_gesture, running

    model = YOLO("best.pt")
    cap = cv2.VideoCapture(0)

    history = []

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, imgsz=384, conf=0.5)
        gesture = None

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best = int(np.argmax(boxes.conf))
            if boxes.conf[best] > 0.5:
                cls_id = int(boxes.cls[best])
                gesture = model.names[cls_id].lower()

        history.append(gesture)
        if len(history) > 5:
            history.pop(0)

        stable = max(set(history), key=history.count) if history else None

        with frame_lock:
            latest_gesture = stable

    cap.release()


thread = threading.Thread(target=yolo_thread, daemon=True)
thread.start()

# -----------------------------
# CONSTANTS
# -----------------------------
WIDTH, HEIGHT   = 1280, 720
GROUND_Y        = HEIGHT // 2 - 20   # baseline enemies + player stand on
PLAYER_X        = 80
GAME_SPEED      = 1.0
ENEMY_SPEED     = 3.5
SPAWN_INTERVAL  = 1.8                 # seconds between spawns (decreases over time)
FIREBALL_SPEED  = 16
FIREBALL_COST   = 6
ULTIMATE_COST   = 100

# -----------------------------
# LOAD + SCALE SPRITES
# -----------------------------
def load_sprite(path, w, h, crop_alpha=False):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        img = np.zeros((h, w, 4), dtype=np.uint8)
        img[:, :] = (180, 100, 200, 255)
        return img
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    if crop_alpha:
        # crop transparent border so all goblins appear the same visual size
        alpha = img[:, :, 3]
        rows = np.any(alpha > 10, axis=1)
        cols = np.any(alpha > 10, axis=0)
        if rows.any() and cols.any():
            r0, r1 = np.where(rows)[0][[0, -1]]
            c0, c1 = np.where(cols)[0][[0, -1]]
            img = img[r0:r1+1, c0:c1+1]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

SCALE = 2

spr_idle    = load_sprite("wizard.png",      int(80*SCALE), int(80*SCALE))
spr_charge  = load_sprite("charge.png",      int(80*SCALE), int(80*SCALE))
GOBLIN_W    = int(55*SCALE)
GOBLIN_H_PX = int(55*SCALE)
# crop_alpha trims padding so all three fill their canvas equally
spr_gob_r   = load_sprite("redgoblin.png",   GOBLIN_W, GOBLIN_H_PX, crop_alpha=True)
spr_gob_b   = load_sprite("bluegoblin.png",  GOBLIN_W, GOBLIN_H_PX, crop_alpha=True)
spr_gob_g   = load_sprite("greengoblin.png", GOBLIN_W, GOBLIN_H_PX, crop_alpha=True)

GOBLIN_MAP = {"red": spr_gob_r, "blue": spr_gob_b, "green": spr_gob_g}

# -----------------------------
# WALKING ANIMATION (4-frame bob cycle)
# -----------------------------
def make_walk_frames(spr):
    h, w = spr.shape[:2]
    frames = []
    for bob in [0, -4, 0, 4]:
        canvas = np.zeros((h, w, 4), dtype=np.uint8)
        scale_y = 1.0 + bob * 0.015
        new_h   = max(1, int(h * scale_y))
        resized = cv2.resize(spr, (w, new_h), interpolation=cv2.INTER_NEAREST)
        if new_h <= h:
            canvas[h - new_h:h, :] = resized   # anchor feet to bottom
        else:
            canvas = resized[:h, :]
        frames.append(canvas)
    return frames

walk_frames = {
    "red":   make_walk_frames(spr_gob_r),
    "blue":  make_walk_frames(spr_gob_b),
    "green": make_walk_frames(spr_gob_g),
}
WALK_FPS   = 8
walk_clock = time.time()

# -----------------------------
# DRAW PNG (alpha blend, clamped)
# -----------------------------
def draw_png(frame, img, x, y):
    if img is None:
        return
    h, w = img.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, WIDTH), min(y + h, HEIGHT)
    if x2 <= x1 or y2 <= y1:
        return
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    region = img[sy1:sy2, sx1:sx2]
    if region.shape[2] == 4:
        alpha = region[:, :, 3:4] / 255.0
        overlay = region[:, :, :3]
    else:
        alpha = 1.0
        overlay = region
    frame[y1:y2, x1:x2] = (alpha * overlay + (1 - alpha) * frame[y1:y2, x1:x2]).astype(np.uint8)

# -----------------------------
# HELPERS
# -----------------------------
def draw_centered(img, text, y, font, scale, color, thickness):
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(img, text, ((WIDTH - tw) // 2, y), font, scale, color, thickness, cv2.LINE_AA)

def draw_rounded_rect(img, x1, y1, x2, y2, r, color, thickness=-1):
    cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, thickness)
    for cx, cy, angle in [(x1+r, y1+r, 180), (x2-r, y1+r, 270), (x1+r, y2-r, 90), (x2-r, y2-r, 0)]:
        cv2.ellipse(img, (cx, cy), (r, r), angle, 0, 90, color, thickness)

# -----------------------------
# GAME STATE
# -----------------------------
game_state   = "start"
score        = 0
high_score   = 0
spawn_timer  = 0.0
wave_start   = 0.0

player = {
    "energy":   0,
    "max_energy": 100,
    "spell":    "red",
    "cooldown": 0.0,
    "charging": False,
}

# sprite y so feet land on GROUND_Y
WIZARD_H    = spr_idle.shape[0]
GOBLIN_H    = GOBLIN_H_PX
WIZARD_Y    = GROUND_Y - WIZARD_H
GOBLIN_Y    = GROUND_Y - GOBLIN_H

SPELL_COLORS = {
    "red":      (60,  60,  220),
    "blue":     (220, 80,   40),
    "green":    (40,  200,  60),
    "ultimate": (255,  255, 255),
}

enemies     = []
projectiles = []
particles   = []   # death burst particles

# -----------------------------
# CLASSES
# -----------------------------
class Fireball:
    def __init__(self, x, y, spell):
        self.x     = float(x)
        self.y     = float(y)
        self.spell = spell
        self.life  = 100

    def update(self):
        self.x   += FIREBALL_SPEED * GAME_SPEED
        self.life -= 1

class Particle:
    def __init__(self, x, y, color):
        self.x   = float(x)
        self.y   = float(y)
        self.vx  = random.uniform(-4, 4)
        self.vy  = random.uniform(-6, 0)
        self.life = random.randint(15, 35)
        self.color = color

    def update(self):
        self.x   += self.vx
        self.y   += self.vy
        self.vy  += 0.4   # gravity
        self.life -= 1

# -----------------------------
# SPAWN
# -----------------------------
def spawn_enemy():
    color = random.choice(["red", "blue", "green"])
    return {"x": float(WIDTH + random.randint(50, 250)), "color": color}

def reset_game():
    global enemies, projectiles, particles, score, spawn_timer, wave_start
    player["energy"]   = 0
    player["spell"]    = "red"
    player["cooldown"] = 0.0
    player["charging"] = False
    enemies     = [spawn_enemy() for _ in range(4)]
    projectiles = []
    particles   = []
    score       = 0
    spawn_timer = time.time()
    wave_start  = time.time()

# -----------------------------
# CAST SPELL
# -----------------------------
def cast_spell():
    # fire from the wizard's right hand area
    fx = PLAYER_X + spr_idle.shape[1] - 10
    fy = WIZARD_Y + WIZARD_H // 2
    projectiles.append(Fireball(fx, fy, player["spell"]))

    if player["spell"] == "ultimate":
        # wide spread of 5 balls
        for offset in [-60, -30, 30, 60]:
            projectiles.append(Fireball(fx, fy + offset, "ultimate"))

# -----------------------------
# DRAW ENERGY BAR
# -----------------------------
def draw_hud(screen):
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    # energy bar
    ratio = player["energy"] / player["max_energy"]
    cv2.rectangle(screen, (30, 155), (330, 180), (30, 30, 30), -1)
    fill_color = SPELL_COLORS.get(player["spell"], (255, 100, 0))
    cv2.rectangle(screen, (30, 155), (30 + int(300 * ratio), 180), fill_color, -1)
    cv2.rectangle(screen, (30, 155), (330, 180), (200, 200, 200), 2)
    cv2.putText(screen, "ENERGY", (30, 150), FONT, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    # spell indicator — two-part so color pops without overflow
    spell_label = player["spell"].upper()
    spell_color = SPELL_COLORS.get(player["spell"], (255,255,255))
    cv2.putText(screen, "SPELL", (30, 130), FONT, 0.6, (180,180,180), 1, cv2.LINE_AA)
    cv2.putText(screen, spell_label, (105, 130), FONT, 0.6, spell_color, 2, cv2.LINE_AA)

    # score
    cv2.putText(screen, f"SCORE  {score}", (30, 55), FONT, 1.0, (255, 220, 60), 2, cv2.LINE_AA)
    cv2.putText(screen, f"BEST   {high_score}", (30, 90), FONT, 0.65, (160, 160, 160), 1, cv2.LINE_AA)

    # gesture debug (bottom right)
    with frame_lock:
        g = latest_gesture
    cv2.putText(screen, f"gesture: {g if g else 'none'}", (WIDTH - 310, HEIGHT - 18),
                FONT, 0.55, (80, 80, 80), 1, cv2.LINE_AA)

# -----------------------------
# DRAW GROUND LINE
# -----------------------------
def draw_ground(screen):
    cv2.line(screen, (0, GROUND_Y + 2), (WIDTH, GROUND_Y + 2), (60, 50, 80), 2)


def draw_game_screen(screen):
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    spells = [
        ("ONE",   (60,60,220),  "Switch to  RED   spell",  None),
        ("TWO", (200,80,40),  "Switch to  BLUE  spell",  None),
        ("THREE", (40,180,60),  "Switch to  GREEN spell",  None),
        ("FOUR",  (255,255,255),  "Switch to  ULTIMATE",     "( requires full 100 energy )"),
    ]
    for i, (sign, dot, label, sublabel) in enumerate(spells):
        y = 500 + i * 58
        cv2.circle(screen, (233, y-10), 11, dot, -1)
        cv2.circle(screen, (233, y-10), 11, (255,255,255), 1)
        cv2.putText(screen, f"{sign:<8} ->  {label}", (258, y), FONT, 0.64, (215,215,215), 1, cv2.LINE_AA)
        if sublabel:
            cv2.putText(screen, sublabel, (258, y + 18), FONT, 0.50, (150,150,150), 1, cv2.LINE_AA)

    



# -----------------------------
# START SCREEN
# -----------------------------
def draw_start(screen, gesture):
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    for row in range(HEIGHT):
        t = row / HEIGHT
        screen[row, :] = (int(40*(1-t)), int(10*(1-t)), int(70*(1-t)))

    draw_centered(screen, "GESTURE  WIZARD", 113, FONT, 2.8, (80, 40, 140), 8)
    draw_centered(screen, "GESTURE  WIZARD", 110, FONT, 2.8, (255, 220, 60), 6)
    draw_centered(screen, "GESTURE  WIZARD", 110, FONT, 2.8, (255, 255, 255), 2)

    draw_rounded_rect(screen, 180, 145, 1100, 570, 18, (25, 20, 50), -1)
    draw_rounded_rect(screen, 180, 145, 1100, 570, 18, (110, 80, 200), 2)

    cv2.putText(screen, "SPELLS", (215, 198), FONT, 0.85, (180,140,255), 2, cv2.LINE_AA)
    spells = [
        ("ONE",   (60,60,220),  "Switch to  RED   spell",  None),
        ("TWO", (200,80,40),  "Switch to  BLUE  spell",  None),
        ("THREE", (40,180,60),  "Switch to  GREEN spell",  None),
        ("FOUR",  (255,255,255),  "Switch to  ULTIMATE",     "( requires full 100 energy )"),
    ]
    for i, (sign, dot, label, sublabel) in enumerate(spells):
        y = 243 + i * 58
        cv2.circle(screen, (233, y-10), 11, dot, -1)
        cv2.circle(screen, (233, y-10), 11, (255,255,255), 1)
        cv2.putText(screen, f"{sign:<8} ->  {label}", (258, y), FONT, 0.64, (215,215,215), 1, cv2.LINE_AA)
        if sublabel:
            cv2.putText(screen, sublabel, (258, y + 18), FONT, 0.50, (150,150,150), 1, cv2.LINE_AA)

    cv2.line(screen, (215, 478), (648, 478), (80,60,140), 1)
    cv2.putText(screen, "ACTIONS", (215, 510), FONT, 0.85, (180,140,255), 2, cv2.LINE_AA)
    cv2.putText(screen, "PALM  ->  hold to charge energy bar", (258, 543), FONT, 0.64, (215,215,215), 1, cv2.LINE_AA)
    cv2.putText(screen, "FIST  ->  fire current spell  ( costs 6 energy )", (258, 568), FONT, 0.60, (215,215,215), 1, cv2.LINE_AA)

    cv2.line(screen, (678, 160), (678, 558), (80,60,140), 1)
    cv2.putText(screen, "HOW TO WIN", (705, 198), FONT, 0.85, (180,140,255), 2, cv2.LINE_AA)
    tips = [
        "Goblins march from the right.",
        "Match your spell COLOR to the",
        "goblin COLOR to destroy it.",
        "If any goblin reaches you —",
        "it's GAME OVER.",
        "",
        "Waves get faster over time.",
        "Chase your high score!",
    ]
    for i, tip in enumerate(tips):
        cv2.putText(screen, tip, (705, 238 + i*42), FONT, 0.6, (195,195,195), 1, cv2.LINE_AA)

    pulse = int(abs(np.sin(time.time() * 3)) * 80 + 170)
    if gesture == "palm":
        draw_centered(screen, "PALM DETECTED  —  starting...", 648, FONT, 1.05, (80,255,120), 2)
    else:
        draw_centered(screen, "Show  PALM  to begin", 648, FONT, 1.1, (pulse,pulse,pulse), 2)

    cv2.putText(screen, f"Detected: {gesture if gesture else 'none'}", (WIDTH-310, HEIGHT-18),
                FONT, 0.55, (80,80,80), 1, cv2.LINE_AA)

# -----------------------------
# GAME OVER SCREEN
# -----------------------------
def draw_gameover(screen, gesture):
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    for row in range(HEIGHT):
        t = row / HEIGHT
        screen[row, :] = (int(10*(1-t)), int(5*(1-t)), int(20*(1-t)))

    draw_centered(screen, "GAME  OVER", 200, FONT, 3.0, (40, 0, 80), 10)
    draw_centered(screen, "GAME  OVER", 197, FONT, 3.0, (0, 0, 180), 8)
    draw_centered(screen, "GAME  OVER", 195, FONT, 3.0, (60, 60, 255), 3)

    draw_rounded_rect(screen, 390, 250, 890, 430, 16, (20,15,40), -1)
    draw_rounded_rect(screen, 390, 250, 890, 430, 16, (80,60,160), 2)

    draw_centered(screen, f"SCORE      {score}",      320, FONT, 1.4, (255,220,60),  3)
    draw_centered(screen, f"BEST       {high_score}", 385, FONT, 1.0, (160,160,160), 2)

    pulse = int(abs(np.sin(time.time() * 3)) * 80 + 170)
    if gesture == "palm":
        draw_centered(screen, "PALM DETECTED  —  restarting...", 510, FONT, 1.0, (80,255,120), 2)
    else:
        draw_centered(screen, "Show  PALM  to play again", 510, FONT, 1.0, (pulse,pulse,pulse), 2)

    cv2.putText(screen, f"Detected: {gesture if gesture else 'none'}", (WIDTH-310, HEIGHT-18),
                FONT, 0.55, (80,80,80), 1, cv2.LINE_AA)

# ==============================
# MAIN LOOP
# ==============================
print("Press Q to quit")

while True:
    screen = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    with frame_lock:
        gesture = latest_gesture

    # ──────────────────────────
    # START SCREEN
    # ──────────────────────────
    if game_state == "start":
        draw_start(screen, gesture)
        if gesture == "palm":
            time.sleep(0.5)
            reset_game()
            game_state = "play"
        cv2.imshow("Game", screen)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break
        continue

    # ──────────────────────────
    # GAME OVER SCREEN
    # ──────────────────────────
    if game_state == "gameover":
        draw_gameover(screen, gesture)
        if gesture == "palm":
            time.sleep(0.5)
            reset_game()
            game_state = "play"
        cv2.imshow("Game", screen)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break
        continue

    # ──────────────────────────
    # CONTROLS
    # ──────────────────────────
    player["charging"] = False

    if gesture == "palm":
        player["energy"] = min(player["max_energy"], player["energy"] + 1)
        player["charging"] = True

    elif gesture == "one":
        player["spell"] = "red"
    elif gesture == "peace":
        player["spell"] = "blue"
    elif gesture == "three" or gesture == "three2":
        player["spell"] = "green"
    elif gesture == "four":
        player["spell"] = "ultimate"

    elif gesture == "stop":
        if time.time() > player["cooldown"]:
            if player["spell"] == "ultimate":
                if player["energy"] >= ULTIMATE_COST:
                    player["energy"] = 0
                    cast_spell()
                    player["cooldown"] = time.time() + 0.8
            else:
                if player["energy"] >= FIREBALL_COST:
                    player["energy"] -= FIREBALL_COST
                    cast_spell()
                    player["cooldown"] = time.time() + 0.35

    # ──────────────────────────
    # SPAWN ENEMIES
    # (interval shrinks with score — gets harder)
    # ──────────────────────────
    interval = max(0.6, SPAWN_INTERVAL - score * 0.015)
    if time.time() - spawn_timer > interval:
        enemies.append(spawn_enemy())
        spawn_timer = time.time()

    # ──────────────────────────
    # UPDATE ENEMIES
    # ──────────────────────────
    speed = ENEMY_SPEED + score * 0.04   # gradually speeds up
    for e in enemies:
        e["x"] -= speed * GAME_SPEED

    # ──────────────────────────
    # COLLISION: enemy reaches player
    # ──────────────────────────
    for e in enemies:
        if e["x"] < PLAYER_X + spr_idle.shape[1] * 0.5:
            high_score = max(high_score, score)
            game_state = "gameover"
            break

    if game_state == "gameover":
        cv2.imshow("Game", screen)
        cv2.waitKey(1)
        continue


    # ──────────────────────────
    # UPDATE PROJECTILES
    # ──────────────────────────
    for p in projectiles:
        p.update()
    projectiles = [p for p in projectiles if p.life > 0]

    # ──────────────────────────
    # COLLISION: fireball vs enemy
    # ──────────────────────────
    hit_enemies     = set()
    hit_projectiles = set()

    for pi, p in enumerate(projectiles):
        for ei, e in enumerate(enemies):
            if ei in hit_enemies:
                continue
            if abs(p.x - (e["x"] + spr_gob_r.shape[1]//2)) < 55:
                # ultimate hits ALL colors; others must match
                if p.spell == "ultimate" or p.spell == e["color"]:
                    hit_enemies.add(ei)
                    hit_projectiles.add(pi)
                    score += 5
                    # spawn death particles
                    dot = SPELL_COLORS.get(e["color"], (255,255,255))
                    for _ in range(12):
                        particles.append(Particle(e["x"] + 60, GOBLIN_Y + GOBLIN_H//2, dot))
                    break

    enemies     = [e for i, e in enumerate(enemies)     if i not in hit_enemies]
    projectiles = [p for i, p in enumerate(projectiles) if i not in hit_projectiles]

    # remove off-screen enemies (shouldn't happen before player, but safety)
    enemies = [e for e in enemies if e["x"] > -200]

    # ──────────────────────────
    # UPDATE PARTICLES
    # ──────────────────────────
    for pt in particles:
        pt.update()
    particles = [pt for pt in particles if pt.life > 0]

    # ══════════════════════════
    # RENDER
    # ══════════════════════════

    # background gradient
    for row in range(HEIGHT):
        t = row / HEIGHT
        screen[row, :] = (int(30*(1-t)+5*t), int(10*(1-t)+5*t), int(50*(1-t)+10*t))

    draw_ground(screen)
    draw_game_screen(screen)

    # particles (behind sprites)
    for pt in particles:
        alpha = pt.life / 35.0
        color = tuple(int(c * alpha) for c in pt.color)
        cv2.circle(screen, (int(pt.x), int(pt.y)), 4, color, -1)

    # enemies — animated walk cycle
    walk_frame_idx = int((time.time() - walk_clock) * WALK_FPS) % 4
    for e in enemies:
        spr = walk_frames[e["color"]][walk_frame_idx]
        draw_png(screen, spr, int(e["x"]), GOBLIN_Y)

    # player
    spr = spr_charge if player["charging"] else spr_idle
    draw_png(screen, spr, PLAYER_X, WIZARD_Y)

    # projectiles
    for p in projectiles:
        base_color = SPELL_COLORS.get(p.spell, (255,255,255))
        # glow: draw 3 concentric circles
        cv2.circle(screen, (int(p.x), int(p.y)), 18, tuple(c//3 for c in base_color), -1)
        cv2.circle(screen, (int(p.x), int(p.y)), 12, tuple(c//2 for c in base_color), -1)
        cv2.circle(screen, (int(p.x), int(p.y)),  7, base_color, -1)

    # HUD on top
    draw_hud(screen)

    cv2.imshow("Game", screen)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

cv2.destroyAllWindows()
