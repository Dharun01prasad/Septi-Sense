"""Single-breadboard 'hub' layout: the breadboard's 4 rails are the shared wires; every part plugs in with jumpers."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
INK = '#16202a'
C = {'3V3': '#d32f2f', 'GND': '#212121', 'SDA': '#1e5bd8', 'SCL': '#e0a100', 'OUT': '#2e9b47',
     'LO+': '#8e44ad', 'LO-': '#c2185b', 'SDN': '#ef6c00'}

fig, ax = plt.subplots(figsize=(15, 10.5))
ax.set_xlim(0, 60); ax.set_ylim(0, 42); ax.set_aspect('equal'); ax.axis('off')
ax.text(0.5, 41, 'One breadboard as a wiring hub (top view)', fontsize=16, weight='bold', color=INK)
ax.text(0.5, 39.6, 'Its 4 long rails are the shared wires: top pair = power, bottom pair = I2C. Label the rails with tape. '
        'Every part connects with jumper wires.', fontsize=10, color='#4a5560')

# breadboard
bx0, bx1, by0, by1 = 4, 44, 14, 25
ax.add_patch(FancyBboxPatch((bx0, by0), bx1 - bx0, by1 - by0, boxstyle='round,pad=0.2,rounding_size=0.6',
                            fc='#f4f1ea', ec='#8c8579', lw=1.5))
rails = {'3V3': 24.0, 'GND': 23.2, 'SDA': 15.8, 'SCL': 15.0}
for name, y in rails.items():
    ax.plot([bx0 + 0.8, bx1 - 0.8], [y, y], color=C[name], lw=4, alpha=0.85, solid_capstyle='round')
    ax.text(bx0 - 0.3, y, name, fontsize=9, color=C[name], weight='bold', va='center', ha='right')
for x in range(5, 44):
    for y in (17.2, 17.8, 18.4, 19.0, 19.6, 20.6, 21.2, 21.8):
        ax.add_patch(Rectangle((x - 0.08, y - 0.08), 0.16, 0.16, fc='#b9b2a5', ec='none'))
ax.text(24, 19.95, 'middle area free (or hold the OLED here)', fontsize=8.5, color='#9b9385', ha='center', va='center')


def part(x, y, w, h, title, sub, pins, side, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05,rounding_size=0.3', fc=fc, ec=INK, lw=1.4))
    ty = y + h - 0.7 if side != 'T' else y + h * 0.42
    ax.text(x + w / 2, ty, title, ha='center', va='center', fontsize=10.5, weight='bold')
    if sub:
        ax.text(x + w / 2, ty - 0.8, sub, ha='center', va='center', fontsize=7.8, color='#4a5560')
    out = {}
    n = len(pins)
    for k, p in enumerate(pins):
        if side == 'B':
            px, py = x + (k + 0.5) * w / n, y
            ax.text(px, py + 0.55, p, ha='center', va='bottom', fontsize=7.5, family='DejaVu Sans Mono', rotation=90)
        elif side == 'T':
            px, py = x + (k + 0.5) * w / n, y + h
            ax.text(px, py - 0.55, p, ha='center', va='top', fontsize=7.5, family='DejaVu Sans Mono', rotation=90)
        else:  # 'L'
            px, py = x, y + h - 2.4 - k * (h - 2.8) / max(1, n - 1)
            ax.text(px + 0.4, py, p, ha='left', va='center', fontsize=7.5, family='DejaVu Sans Mono')
        ax.add_patch(Circle((px, py), 0.22, fc='#c9a227', ec=INK, lw=0.8, zorder=6))
        out[p] = (px, py)
    return out


def drop(p, y, color, lw=2.4):
    x, y0 = p
    ax.plot([x, x], [y0, y], color=color, lw=lw, zorder=4, solid_capstyle='round')
    ax.add_patch(Circle((x, y), 0.2, fc=color, ec='white', lw=0.6, zorder=7))


# three I2C modules above the breadboard, pins facing down
mods = [
    part(5, 28, 7.5, 6.5, 'OLED', '0x3C', ['GND', 'VCC', 'SCL', 'SDA'], 'B', '#eceaf9'),
    part(15, 28, 7.5, 6.5, 'MLX90614', 'IR temp  0x5A', ['VIN', 'GND', 'SCL', 'SDA'], 'B', '#fff5e0'),
    part(25, 28, 11, 6.5, 'MAX30102', 'pulse + SpO2  0x57', ['VIN', 'SDA', 'SCL', 'INT', 'GND'], 'B', '#fdecef'),
]
for m in mods:
    for pn, p in m.items():
        if pn in ('VCC', 'VIN'):
            drop(p, rails['3V3'], C['3V3'])
        elif pn == 'GND':
            drop(p, rails['GND'], C['GND'])
        elif pn == 'SDA':
            drop(p, rails['SDA'], C['SDA'])
        elif pn == 'SCL':
            drop(p, rails['SCL'], C['SCL'])
ax.text(30.5, 27.3, 'INT: leave free', fontsize=7.5, color='#777', ha='center')

# ESP32 below the breadboard, pins facing up
esp = part(8, 2, 34, 8, 'ESP32 DevKit', 'sits beside the breadboard; female-to-male jumpers onto its pins  |  USB to laptop',
           ['3V3', 'GND', 'D21', 'D22', 'D34', 'D35', 'D32'], 'T', '#e8eef5')
drop(esp['3V3'], rails['3V3'], C['3V3'])
drop(esp['GND'], rails['GND'], C['GND'])
drop(esp['D21'], rails['SDA'], C['SDA'])
drop(esp['D22'], rails['SCL'], C['SCL'])

# AD8232 to the right
ad = part(48, 25, 10, 11, 'AD8232', 'ECG  |  3.3 V only', ['3.3V', 'GND', 'OUTPUT', 'LO+', 'LO-', 'SDN'], 'L', '#e9f6ec')
for pn, rn in (('3.3V', '3V3'), ('GND', 'GND')):
    x, y = ad[pn]
    ax.plot([x, 46.0 + (0.6 if pn == 'GND' else 0), 46.0 + (0.6 if pn == 'GND' else 0)], [y, y, rails[rn]],
            color=C[rn], lw=2.4, zorder=4)
    ax.plot([46.0 + (0.6 if pn == 'GND' else 0), bx1 - 1.0], [rails[rn], rails[rn]], color=C[rn], lw=2.4, zorder=4)
    ax.add_patch(Circle((bx1 - 1.0, rails[rn]), 0.2, fc=C[rn], ec='white', lw=0.6, zorder=7))
for k, (pn, ep) in enumerate((('OUTPUT', 'D34'), ('LO+', 'D35'), ('LO-', 'D32'))):
    x, y = ad[pn]
    xl = 47.0 + k * 0.0
    ex, ey = esp[ep]
    lane = 12.2 - k * 0.7
    ax.plot([x, 45.6 - k * 0.6, 45.6 - k * 0.6, ex, ex], [y, y, lane, lane, ey], color=C[pn if pn != 'OUTPUT' else 'OUT'],
            lw=2.4, zorder=4)
xs, ys = ad['SDN']
x3, y3 = ad['3.3V']
ax.plot([xs, xs - 0.9, xs - 0.9, x3], [ys, ys, y3 - 0.3, y3 - 0.3], color=C['SDN'], lw=1.8, ls=(0, (3, 2)), zorder=5)
ax.text(52.5, 24.2, 'SDN: jumper to its own 3.3V', fontsize=8, color=C['SDN'], ha='center')
ax.text(53, 37.0, 'ECG cable into the jack', fontsize=8.5, color='#4a5560', ha='center')
ax.text(46.5, 11.0, 'OUTPUT / LO+ / LO- go straight\nto the ESP32 (no breadboard)', fontsize=8.5, color='#4a5560', va='top')

ax.text(0.5, 0.4, 'Pin order on your modules may differ from this picture: always match the names printed on each board.',
        fontsize=9.5, color='#4a5560')
fig.savefig(os.path.join(HERE, 'breadboard_layout.png'), dpi=170, bbox_inches='tight', facecolor='white')
print('ok')
