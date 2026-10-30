"""Arm-lead ECG placement + the wrist band / finger clip build, as one picture."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon, Ellipse

HERE = os.path.dirname(os.path.abspath(__file__))
INK = '#16202a'
SKIN = '#f2dccb'
fig, ax = plt.subplots(figsize=(16, 9.6))
ax.set_xlim(0, 64); ax.set_ylim(0, 38.4); ax.set_aspect('equal'); ax.axis('off')
ax.text(0.5, 37.3, 'Arm placement and the wearable band (seen from above, arms resting on a table, palms up)',
        fontsize=15, weight='bold', color=INK)


def forearm(x_wrist, x_elbow, y, hand_left):
    # forearm as a tapered band, hand at the wrist end
    w_w, w_e = 2.6, 3.6
    ax.add_patch(Polygon([(x_wrist, y - w_w / 2), (x_elbow, y - w_e / 2), (x_elbow, y + w_e / 2), (x_wrist, y + w_w / 2)],
                         closed=True, fc=SKIN, ec=INK, lw=1.4))
    s = -1 if hand_left else 1
    palm_x = x_wrist + s * 2.0
    ax.add_patch(Ellipse((palm_x, y), 4.2, 3.6, fc=SKIN, ec=INK, lw=1.4))
    fingers = []
    for k, dy in enumerate((-1.25, -0.42, 0.42, 1.25)):
        fx0 = palm_x + s * 1.8
        L = (2.6, 3.1, 2.9, 2.3)[k]
        ax.add_patch(FancyBboxPatch((min(fx0, fx0 + s * L), y + dy - 0.33), L, 0.66,
                                    boxstyle='round,pad=0,rounding_size=0.3', fc=SKIN, ec=INK, lw=1.2))
        fingers.append((fx0 + s * (L - 0.7), y + dy))
    # thumb
    ax.add_patch(FancyBboxPatch((palm_x - 0.4, y + 1.4), 0.8, 2.1, boxstyle='round,pad=0,rounding_size=0.35',
                                fc=SKIN, ec=INK, lw=1.2))
    return fingers


# person's RIGHT arm on the viewer's left (palms up, facing you), LEFT arm on the right
yarm = 21.5
fR = forearm(13.0, 26.0, yarm, hand_left=True)     # right arm: hand points left
fL = forearm(51.0, 38.0, yarm, hand_left=False)
ax.text(19.5, yarm - 3.4, "person's RIGHT forearm", ha='center', fontsize=10, color='#4a5560')
ax.text(44.5, yarm - 3.4, "person's LEFT forearm", ha='center', fontsize=10, color='#4a5560')


def electrode(x, y, lab, c):
    ax.add_patch(Circle((x, y), 0.85, fc='white', ec='#888', lw=1.0, zorder=5))
    ax.add_patch(Circle((x, y), 0.55, fc=c, ec=INK, lw=1.1, zorder=6))
    ax.text(x, y, lab, ha='center', va='center', fontsize=8, color='white', weight='bold', zorder=7)


electrode(15.6, yarm + 0.3, 'RA', '#d32f2f')
electrode(23.6, yarm - 0.6, 'RL', '#2e9b47')
electrode(45.4, yarm + 0.3, 'LA', '#e0a100')

# wrist band on the LEFT wrist: strap + MLX in a spacer cap, facing the inner wrist
bx = 46.2
ax.add_patch(Rectangle((bx - 0.45 + 2.2, yarm - 2.0), 0.9, 4.0, fc='#3b4a5a', ec=INK, lw=1.2, zorder=4))
ax.add_patch(Circle((bx + 2.65, yarm), 0.95, fc='#fff5e0', ec=INK, lw=1.3, zorder=6))
ax.text(bx + 2.65, yarm, 'T', ha='center', va='center', fontsize=9, weight='bold', zorder=7)
# finger clip on the left index finger (second finger from thumb side)
fx, fy = fL[1]
ax.add_patch(FancyBboxPatch((fx - 0.9, fy - 0.75), 1.8, 1.5, boxstyle='round,pad=0,rounding_size=0.3',
                            fc='#fdecef', ec=INK, lw=1.3, zorder=6))
ax.text(fx, fy, 'O₂', ha='center', va='center', fontsize=8.5, weight='bold', zorder=7)

# cables to the table electronics
ax.plot([bx + 2.65, bx + 2.65], [yarm - 1.0, 16.4], color='#e0a100', lw=2.0, ls=(0, (5, 2)))
ax.plot([fx, fx], [fy - 0.8, 16.4], color='#c2185b', lw=2.0, ls=(0, (5, 2)))
for (x, y) in ((15.6, yarm + 0.3), (23.6, yarm - 0.6), (45.4, yarm + 0.3)):
    ax.plot([x, x], [y - 0.9, 11.6], color='#777', lw=1.4)
ax.plot([15.6, 45.4], [11.6, 11.6], color='#777', lw=1.4)
ax.add_patch(FancyBboxPatch((48.5, 13.0), 14, 3.4, boxstyle='round,pad=0.1,rounding_size=0.4', fc='#e8eef5', ec=INK, lw=1.3))
ax.text(55.5, 14.7, 'breadboard + ESP32\non the table', ha='center', va='center', fontsize=9.5, weight='bold')
ax.plot([45.4, 50.0, 50.0], [11.6, 11.6, 13.0], color='#777', lw=1.4)
ax.text(28, 12.0, 'ECG cable (3 clips) to the AD8232', fontsize=9, color='#4a5560', ha='center')

# callouts
def callout(x, y, tx, ty, text):
    ax.annotate(text, xy=(x, y), xytext=(tx, ty), fontsize=9.6, color=INK, va='center',
                arrowprops=dict(arrowstyle='-', color='#666', lw=1.0))


callout(15.6, yarm + 1.1, 6.0, 31.5, 'RA: inside of the right forearm,\n3-4 cm above the wrist crease')
callout(23.6, yarm - 1.4, 18.5, 15.2, 'RL: inside of the right forearm,\nnear the elbow (reference)')
callout(45.4, yarm + 1.1, 31.5, 31.5, 'LA: inside of the left forearm,\njust above the band')
callout(bx + 2.65, yarm + 0.95, 45.5, 31.8, 'T = MLX90614 in a spacer cap,\nfacing the inner wrist (fixed 1 cm gap)')
callout(fx, fy + 0.8, 54.0, 27.0, 'O₂ = MAX30102 clip on the index\nfingertip, covered from light')

# band cross-section detail
ax.text(1.0, 9.6, 'Wrist band, side view', fontsize=11.5, weight='bold', color=INK)
ax.add_patch(Rectangle((1.5, 4.0), 13, 1.2, fc=SKIN, ec=INK, lw=1.2))
ax.text(8.0, 4.6, 'inner wrist skin', ha='center', va='center', fontsize=8.5, color='#6b5547')
ax.add_patch(Rectangle((5.2, 5.2), 0.35, 1.6, fc='#555', ec=INK, lw=0.8))
ax.add_patch(Rectangle((10.45, 5.2), 0.35, 1.6, fc='#555', ec=INK, lw=0.8))
ax.add_patch(Rectangle((5.2, 6.8), 5.6, 0.45, fc='#fff5e0', ec=INK, lw=1.0))
ax.add_patch(Rectangle((7.2, 6.55), 1.6, 0.25, fc='#333', ec=INK, lw=0.6))
ax.text(15.0, 6.95, 'GY-906 board\n(sensor faces down)', ha='left', va='center', fontsize=8.2)
ax.add_patch(Rectangle((1.5, 7.25), 13, 0.5, fc='#3b4a5a', ec=INK, lw=1.0))
ax.text(15.0, 7.55, 'Velcro strap', ha='left', va='bottom', fontsize=8.2, color='#3b4a5a')
ax.text(8.0, 3.3, 'spacer: bottle cap or 1 cm cardboard/foam ring\nkeeps the distance fixed and blocks moving air',
        ha='center', va='top', fontsize=8.4, color='#4a5560')

ax.text(22.0, 9.6, 'Finger clip, side view', fontsize=11.5, weight='bold', color=INK)
ax.add_patch(FancyBboxPatch((23.0, 4.6), 9.5, 1.7, boxstyle='round,pad=0,rounding_size=0.8', fc=SKIN, ec=INK, lw=1.2))
ax.text(27.5, 5.45, 'index fingertip (pad facing down)', ha='center', va='center', fontsize=8.3, color='#6b5547')
ax.add_patch(Rectangle((25.2, 3.6), 4.6, 1.0, fc='#fdecef', ec=INK, lw=1.0))
ax.add_patch(Rectangle((27.0, 4.35), 1.0, 0.25, fc='#333'))
ax.text(30.8, 3.95, 'MAX30102\n(window up)', ha='left', va='center', fontsize=8.2)
ax.add_patch(Rectangle((24.6, 6.3), 5.8, 0.45, fc='#222', ec=INK, lw=0.8))
ax.add_patch(Rectangle((24.6, 3.15), 5.8, 0.45, fc='#222', ec=INK, lw=0.8))
ax.text(27.5, 7.3, 'black tape / Velcro wrap\n(light shield, light pressure)', ha='center', va='bottom', fontsize=8.2, color='#333')

ax.text(36.0, 9.6, 'Rules for clean signals', fontsize=11.5, weight='bold', color=INK)
ax.text(36.0, 8.7, '- Arms rest still on the table, palms up, relaxed.\n'
        '- Stickers on the forearm, never on the palm or fingers.\n'
        '- Finger clip snug but not tight: pressing hard kills the pulse.\n'
        '- Keep the 3 sensor leads under ~50 cm; twist each set of 4 wires.\n'
        '- Laptop on battery while the stickers are on.',
        fontsize=9.3, color=INK, va='top')
fig.savefig(os.path.join(HERE, 'wearable_placement.png'), dpi=170, bbox_inches='tight', facecolor='white')
print('ok')
