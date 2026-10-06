"""
Hill Climb Racing clone controlled by hand gestures.

    pip install pygame pymunk opencv-python "mediapipe==0.10.14" numpy
    python hill_climb_hands.py          # webcam hand control
    python hill_climb_hands.py --keys   # keyboard fallback (arrows + A/D)

Gestures (one hand, palm facing the camera):
    open palm (3-4 fingers up)  -> GAS
    fist (0-1 fingers up)       -> BRAKE
    2 fingers up                -> coast
    tilt hand left/right        -> rotate the car in the air
R = restart, ESC = quit
"""
import sys, math, threading
import pygame, pymunk

W, H = 1000, 600
KEYBOARD = "--keys" in sys.argv


# ---------------------------------------------------------------- hand input
class HandInput(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        import cv2, mediapipe as mp
        self.cv2, self.mp = cv2, mp
        self.gas = self.brake = False
        self.tilt = 0.0
        self.thumb = None
        self.alive = True

    def run(self):
        cv2, mp = self.cv2, self.mp
        cap = cv2.VideoCapture(0)
        hands = mp.solutions.hands.Hands(
            max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)
        draw = mp.solutions.drawing_utils
        while self.alive:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            gas = brake = False
            tilt = 0.0
            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                lm = hand.landmark
                d = lambda a, b: math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)
                # finger is "up" if its tip is farther from the wrist than its middle joint
                up = sum(d(t, 0) > d(p, 0) * 1.1 for t, p in ((8, 6), (12, 10), (16, 14), (20, 18)))
                gas, brake = up >= 3, up <= 1
                ang = math.degrees(math.atan2(lm[9].x - lm[0].x, lm[0].y - lm[9].y))
                tilt = max(-1.0, min(1.0, ang / 45))
                draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            self.gas, self.brake, self.tilt = gas, brake, tilt
            self.thumb = cv2.cvtColor(cv2.resize(frame, (240, 180)), cv2.COLOR_BGR2RGB)
        cap.release()


# ---------------------------------------------------------------- game world
class Game:
    def __init__(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, -900)
        self.space.iterations = 20
        self.pts, self.segs = [], []
        self.tx = -300
        self.extend_terrain(1800)
        self.make_car(100, 220)
        self.over = False
        self.best = 0.0

    def height(self, x):
        k = min(1.0, max(0.0, (x - 200) / 500))          # flat start, then ramps up
        amp = min(40 + x * 0.04, 200)
        return 100 + k * (0.5 * amp * math.sin(x / 260) + 0.3 * amp * math.sin(x / 97 + 1.3))

    def extend_terrain(self, x_end):
        while self.tx < x_end:
            a, b = (self.tx, self.height(self.tx)), (self.tx + 20, self.height(self.tx + 20))
            seg = pymunk.Segment(self.space.static_body, a, b, 4)
            seg.friction = 1.5
            self.space.add(seg)
            self.segs.append(seg)
            self.pts.append(a)
            self.tx += 20

    def prune_terrain(self, x_min):
        while self.segs and self.segs[0].b.x < x_min:
            self.space.remove(self.segs.pop(0))
            self.pts.pop(0)

    def make_car(self, x, y):
        sp = self.space
        self.body = pymunk.Body()
        self.body.position = (x, y)
        self.chassis = pymunk.Poly.create_box(self.body, (100, 20))
        self.chassis.mass = 6
        self.chassis.friction = 0.5
        self.chassis.filter = pymunk.ShapeFilter(group=1)
        sp.add(self.body, self.chassis)
        self.wheels, self.motors = [], []
        for dx in (-38, 38):
            w = pymunk.Body()
            w.position = (x + dx, y - 30)
            c = pymunk.Circle(w, 18)
            c.mass, c.friction = 2, 1.6
            c.filter = pymunk.ShapeFilter(group=1)
            groove = pymunk.GrooveJoint(self.body, w, (dx, -10), (dx, -45), (0, 0))
            spring = pymunk.DampedSpring(self.body, w, (dx, -10), (0, 0), 35, 8000, 300)
            motor = pymunk.SimpleMotor(w, self.body, 0)
            motor.max_force = 0
            sp.add(w, c, groove, spring, motor)
            self.wheels.append(w)
            self.motors.append(motor)

    def head_pos(self):
        return self.body.local_to_world((5, 24))

    def step(self, gas, brake, tilt):
        if self.over:
            return
        for m in self.motors:
            if gas:
                m.rate, m.max_force = -30, 60000
            elif brake:
                m.rate, m.max_force = 0, 80000
            else:
                m.rate, m.max_force = 0, 0
        for _ in range(2):
            self.body.torque = -tilt * 100000
            self.space.step(1 / 120)
        hx, hy = self.head_pos()
        if hy - 9 < self.height(hx):
            self.over = True
        self.best = max(self.best, (self.body.position.x - 100) / 50)
        self.extend_terrain(self.body.position.x + 1500)
        self.prune_terrain(self.body.position.x - 900)


# ---------------------------------------------------------------- rendering
def draw(screen, g, font, inp):
    cx, cy = g.body.position.x - 300, g.body.position.y
    S = lambda p: (p[0] - cx, H * 0.6 - (p[1] - cy))
    screen.fill((135, 200, 235))
    vis = [S(p) for p in g.pts if -40 < p[0] - cx < W + 40]
    if len(vis) > 1:
        pygame.draw.polygon(screen, (95, 70, 45), vis + [(vis[-1][0], H), (vis[0][0], H)])
        pygame.draw.lines(screen, (70, 160, 60), False, vis, 8)
    verts = [S(g.body.local_to_world(v)) for v in g.chassis.get_vertices()]
    pygame.draw.polygon(screen, (200, 40, 40), verts)
    for w in g.wheels:
        c = S(w.position)
        pygame.draw.circle(screen, (25, 25, 25), c, 18)
        a = w.angle
        pygame.draw.line(screen, (200, 200, 200), c, (c[0] + 14 * math.cos(-a), c[1] + 14 * math.sin(-a)), 3)
    pygame.draw.circle(screen, (250, 210, 160), S(g.head_pos()), 9)

    status = "GAS" if inp[0] else "BRAKE" if inp[1] else "COAST"
    screen.blit(font.render(f"{(g.body.position.x - 100) / 50:.0f} m   best {g.best:.0f} m   [{status}]",
                            True, (20, 20, 20)), (15, 12))
    if g.over:
        t = font.render("CRASHED - press R", True, (180, 0, 0))
        screen.blit(t, (W // 2 - t.get_width() // 2, H // 2 - 60))


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Hand Climb Racing")
    font = pygame.font.SysFont("arial", 28, bold=True)
    clock = pygame.time.Clock()
    hand = None
    if not KEYBOARD:
        hand = HandInput()
        hand.start()
    g = Game()
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                if hand:
                    hand.alive = False
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                best = g.best
                g = Game()
                g.best = best
        if hand:
            gas, brake, tilt = hand.gas, hand.brake, hand.tilt
        else:
            k = pygame.key.get_pressed()
            gas, brake = k[pygame.K_RIGHT], k[pygame.K_LEFT]
            tilt = float(k[pygame.K_d]) - float(k[pygame.K_a])
        g.step(gas, brake, tilt)
        draw(screen, g, font, (gas, brake))
        if hand and hand.thumb is not None:
            surf = pygame.image.frombuffer(hand.thumb.tobytes(), (240, 180), "RGB")
            screen.blit(surf, (W - 250, 10))
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
